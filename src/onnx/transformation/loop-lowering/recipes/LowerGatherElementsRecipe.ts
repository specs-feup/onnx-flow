import type OnnxGraph from "../../../ir/OnnxGraph.js";
import type OperationNode from "../../../ir/OperationNode.js";
import {
    type ValueNode,
    type ConcreteValueNode,
    type KnownShape,
    DataType,
} from "../../../ir/OnnxTypes.js";
import {
    getIntAttr,
    int64Vec,
    normalizeAxis,
    scalarInt64,
    toStaticShape,
} from "../../../utils/index.js";
import type {
    LoopCarryContext,
    LoopLoweringRecipe,
    RecipeApplyResult,
} from "../LoopLoweringRecipe.js";
import {
    buildAguPointer,
    buildLinearIndex,
    buildMixedStrides,
    decodeMixedRadix,
    ensureFlatInput,
    ensureInnerValueNode,
    extractMixedDims,
    gatherScalar,
    resolveRecipeInput,
} from "../RecipeUtils.js";
import { GraphBuilder } from "../../../ir/GraphBuilder.js";
import { IndexingMode } from "../../../../DecompositionOptions.js";

export class LowerGatherElementsRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerGatherElements";
    public readonly targetOp = "GatherElements";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["indexing", "fuseable"] as const;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "ReduceProd",
        "Reshape",
        "Gather",
        "Unsqueeze",
        "Squeeze",
        "Add",
        "Mul",
        "Less",
        "Equal",
        "And",
        "Where",
        "Cast",
        "Div",
        "Mod",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "GatherElements") return false;
        const ins = op.getInputs();
        if (!ins || ins.length < 2) return false;
        const dataDims = toStaticShape(ins[0].shape);
        const idxDims = toStaticShape(ins[1].shape);
        return dataDims.length > 0 && dataDims.length === idxDims.length;
    }

    getLoopBounds(
        op: OperationNode.Class,
        _outShape: KnownShape,
    ): {
        totalIters: number | ConcreteValueNode;
        carryShape: number[] | ConcreteValueNode;
        targetShape?: ConcreteValueNode;
    } {
        const idx = op.getInputs()![1];
        const idxDims = toStaticShape(idx.shape);
        if (!idxDims.includes(-1)) {
            const total = idxDims.reduce((a, b) => a * b, 1);
            return { totalIters: total, carryShape: idxDims };
        }
        const builder = new GraphBuilder(op.graph as OnnxGraph.Class, `ge_bounds_${op.id}`);
        const ax0 = builder.createConstant(`geb_ax0_${op.id}`, int64Vec([0]));
        const [targetShape] = builder.createOp("Shape", [idx], {}, [
            { type: DataType.INT64, shape: [idxDims.length] },
        ]);
        const [totalIters] = builder.createOp("ReduceProd", [targetShape, ax0], { keepdims: 0 });
        const [carryShape] = builder.createOp("Unsqueeze", [totalIters, ax0]);
        return { totalIters, carryShape, targetShape };
    }

    apply(
        op: OperationNode.Class,
        body: OnnxGraph.Class,
        valueMap: Map<string, ValueNode>,
        iter: ConcreteValueNode,
        axes: ConcreteValueNode,
        outShape: KnownShape,
        _carryNode: ConcreteValueNode,
        targetShapeNode?: ValueNode,
        loopCtx?: LoopCarryContext,
    ): RecipeApplyResult {
        const builder = new GraphBuilder(body, `ge_${op.id}`);
        const ins = op.getInputs()!;
        const rawDataDims = toStaticShape(ins[0].shape);
        const rawOutDims = toStaticShape(ins[1].shape);
        const axis = normalizeAxis(getIntAttr(op, "axis", 0), rawDataDims.length);

        const rawData = resolveRecipeInput(
            builder,
            ins[0],
            valueMap,
            iter,
            axes,
            outShape,
            false,
            false,
            targetShapeNode,
            loopCtx,
        );
        const dataDims = extractMixedDims(builder, rawData, rawDataDims, `ge_dd_${op.id}`);
        const dataStrides = buildMixedStrides(builder, dataDims);
        const axisDim = dataDims[axis];
        const axisStride = dataStrides[axis];
        const flatData = ensureFlatInput(builder, rawData);

        let outDims: (number | ConcreteValueNode)[];
        if (!rawOutDims.includes(-1)) {
            outDims = [...rawOutDims];
        } else if (targetShapeNode) {
            const innerShape = ensureInnerValueNode(builder, targetShapeNode);
            outDims = rawOutDims.map((d, i) => {
                if (d !== -1) return d;
                const dIdx = builder.createConstant(`ge_od_${op.id}_${i}`, scalarInt64(i));
                return gatherScalar(builder, innerShape, dIdx);
            });
        } else {
            const rawIdx = ensureInnerValueNode(builder, ins[1]);
            outDims = extractMixedDims(builder, rawIdx, rawOutDims, `ge_od_${op.id}`);
        }

        // Read scalar index value for this output position
        let idxVal = resolveRecipeInput(
            builder,
            ins[1],
            valueMap,
            iter,
            axes,
            rawOutDims,
            true,
            true,
            targetShapeNode,
            loopCtx,
        ) as ConcreteValueNode;

        if (ins[1].literalType !== DataType.INT64) {
            [idxVal] = builder.createOp("Cast", [idxVal], { to: DataType.INT64 });
        }

        // Wrap negative indices: idx < 0 ? idx + dataDims[axis] : idx
        const zeroI64 = builder.createConstant(`ge_z64_${op.id}`, scalarInt64(0));
        const axisDimNode =
            typeof axisDim === "number"
                ? builder.createConstant(`ge_adim_${op.id}`, scalarInt64(axisDim))
                : axisDim;
        const [isNeg] = builder.createOp("Less", [idxVal, zeroI64]);
        const [wrapped] = builder.createOp("Add", [idxVal, axisDimNode]);
        const [normIdx] = builder.createOp("Where", [isNeg, wrapped, idxVal]);

        // Base data pointer for all non-axis coordinates
        const baseStrides: (number | ConcreteValueNode)[] = [...dataStrides];
        baseStrides[axis] = 0;

        let basePtr: ConcreteValueNode;
        if (loopCtx && loopCtx.indexingMode === IndexingMode.Pointer) {
            basePtr = buildAguPointer(
                builder,
                loopCtx,
                valueMap,
                outDims,
                baseStrides,
                `ge_base_${op.id}`,
            );
        } else {
            const coords = decodeMixedRadix(builder, iter, outDims, `ge_dec_${op.id}`);
            basePtr = buildLinearIndex(builder, coords, baseStrides, `ge_lin_${op.id}`);
        }

        // flatDataIdx = basePtr + normIdx * axisStride
        let axisOff: ConcreteValueNode = normIdx;
        if (typeof axisStride !== "number") {
            [axisOff] = builder.createOp("Mul", [normIdx, axisStride]);
        } else if (axisStride !== 1) {
            const sConst = builder.createConstant(`ge_astr_${op.id}`, scalarInt64(axisStride));
            [axisOff] = builder.createOp("Mul", [normIdx, sConst]);
        }
        const [flatDataIdx] = builder.createOp("Add", [basePtr, axisOff]);

        return gatherScalar(builder, flatData, flatDataIdx);
    }
}
