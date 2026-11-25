import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import {
    type ValueNode,
    type ConcreteValueNode,
    type KnownShape,
    DataType,
} from "../../../OnnxTypes.js";
import {
    computeStrides,
    getIntAttr,
    normalizeAxis,
    scalarInt64,
    toStaticShape,
} from "../../../Utils.js";
import type {
    LoopCarryContext,
    LoopLoweringRecipe,
    RecipeApplyResult,
} from "../LoopLoweringRecipe.js";
import {
    buildAguPointer,
    buildLinearIndex,
    decodeMixedRadix,
    ensureFlatInput,
    gatherScalar,
    resolveRecipeInput,
} from "../RecipeUtils.js";
import { GraphBuilder } from "../../../GraphBuilder.js";
import { IndexingMode } from "../../../../DecompositionOptions.js";

export class LowerGatherElementsRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerGatherElements";
    public readonly targetOp = "GatherElements";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly priority = 50;
    public readonly shapeSupport = "static" as const;
    public readonly tags = ["indexing", "fuseable"] as const;
    public readonly producedOps = [
        "Loop",
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
        return (
            dataDims.length > 0 &&
            dataDims.length === idxDims.length &&
            !dataDims.includes(-1) &&
            !idxDims.includes(-1)
        );
    }

    getLoopBounds(
        op: OperationNode.Class,
        _outShape: KnownShape,
    ): { totalIters: number; carryShape: number[] } {
        const idxDims = toStaticShape(op.getInputs()![1].shape);
        const total = idxDims.reduce((a, b) => a * b, 1);
        return { totalIters: total, carryShape: idxDims };
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
        const dataDims = toStaticShape(ins[0].shape);
        const outDims = toStaticShape(ins[1].shape);
        const axis = normalizeAxis(getIntAttr(op, "axis", 0), dataDims.length);

        const dataStrides = computeStrides(dataDims);
        const axisDim = dataDims[axis];
        const axisStride = dataStrides[axis];

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
        const flatData = ensureFlatInput(builder, rawData);

        // Read scalar index value for this output position
        let idxVal = resolveRecipeInput(
            builder,
            ins[1],
            valueMap,
            iter,
            axes,
            outDims,
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
        const axisDimConst = builder.createConstant(`ge_adim_${op.id}`, scalarInt64(axisDim));
        const [isNeg] = builder.createOp("Less", [idxVal, zeroI64]);
        const [wrapped] = builder.createOp("Add", [idxVal, axisDimConst]);
        const [normIdx] = builder.createOp("Where", [isNeg, wrapped, idxVal]);

        // Base data pointer for all non-axis coordinates
        const baseStrides = [...dataStrides];
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
        if (axisStride !== 1) {
            const sConst = builder.createConstant(`ge_astr_${op.id}`, scalarInt64(axisStride));
            [axisOff] = builder.createOp("Mul", [normIdx, sConst]);
        }
        const [flatDataIdx] = builder.createOp("Add", [basePtr, axisOff]);

        return gatherScalar(builder, flatData, flatDataIdx);
    }
}
