import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import {
    type ValueNode,
    type ConcreteValueNode,
    type KnownShape,
    DataType,
} from "../../../OnnxTypes.js";
import {
    asStaticDims,
    broadcastShapes,
    scalarInt64,
    computeStrides,
    int64Vec,
    UNKNOWN_SHAPE,
} from "../../../Utils.js";
import type {
    LoopCarryContext,
    LoopLoweringRecipe,
    RecipeApplyResult,
} from "../LoopLoweringRecipe.js";
import {
    resolveRecipeInput,
    buildLinearIndex,
    decodeMixedRadix,
    squeezeIfLen1,
    buildAguPointer,
    getOrCreateIterationSpaceCounters,
} from "../RecipeUtils.js";
import { GraphBuilder } from "../../../GraphBuilder.js";
import { IndexingMode } from "../../../../DecompositionOptions.js";

export class LowerMatMulRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerMatMul";
    public readonly targetOp = "MatMul";
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "Size",
        "Gather",
        "Unsqueeze",
        "Squeeze",
        "Mul",
        "Div",
        "Mod",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "MatMul") return false;
        const inputs = op.getInputs();
        return (
            !!inputs &&
            inputs.length >= 2 &&
            inputs[0].shape.length > 0 &&
            inputs[1].shape.length > 0
        );
    }

    getLoopBounds(
        op: OperationNode.Class,
        _outShape: KnownShape,
    ): {
        totalIters: number | ConcreteValueNode;
        carryShape: number[] | ConcreteValueNode;
        targetShape?: number[] | ConcreteValueNode;
    } {
        const inputs = op.getInputs()!;

        // 1. Static Case
        if (
            !inputs[0].shape.includes(-1) &&
            !inputs[1].shape.includes(-1) &&
            inputs[0].shape.length > 0 &&
            inputs[1].shape.length > 0
        ) {
            let sA = asStaticDims(inputs[0].shape),
                sB = asStaticDims(inputs[1].shape);
            if (sA.length === 1) sA = [1, sA[0]];
            if (sB.length === 1) sB = [sB[0], 1];

            const M = sA[sA.length - 2],
                N = sB[sB.length - 1];
            const batchOut = broadcastShapes(...[sA.slice(0, -2), sB.slice(0, -2)]);
            const prodBatch = batchOut.reduce((a, b) => a * b, 1);

            const totalIters = prodBatch * M * N;
            return { totalIters, carryShape: [totalIters] };
        }

        // 2. Dynamic Case
        const builder = new GraphBuilder(op.graph as OnnxGraph.Class, `mm_bounds_${op.id}`);
        const axes0 = builder.createConstant(`axes0_${op.id}`, int64Vec([0]));

        const [shapeA] = builder.createOp("Shape", [inputs[0]]);
        const [shapeB] = builder.createOp("Shape", [inputs[1]]);

        const expectedCoS = [{ type: DataType.FLOAT, shape: UNKNOWN_SHAPE }];
        const [dummyA] = builder.createOp("ConstantOfShape", [shapeA], {}, expectedCoS);
        const [dummyB] = builder.createOp("ConstantOfShape", [shapeB], {}, expectedCoS);

        const expectedMM = [{ type: DataType.FLOAT, shape: UNKNOWN_SHAPE }];
        const [dummyOut] = builder.createOp("MatMul", [dummyA, dummyB], {}, expectedMM);

        const [targetShapeNode] = builder.createOp("Shape", [dummyOut]);
        const [totalIters] = builder.createOp("ReduceProd", [targetShapeNode, axes0], {
            keepdims: 0,
        });
        const [carryShape] = builder.createOp("Unsqueeze", [totalIters, axes0]);

        return { totalIters, carryShape, targetShape: targetShapeNode };
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
        const builder = new GraphBuilder(body, `matmul_${op.id}`);
        const inputs = op.getInputs()!;

        // 1. Capture inputs into the loop body first so dynamic Shape ops stay in scope
        const tInnerA = resolveRecipeInput(
            builder,
            inputs[0],
            valueMap,
            iter,
            axes,
            outShape,
            false,
            false,
            targetShapeNode,
            loopCtx,
        );
        const tInnerB = resolveRecipeInput(
            builder,
            inputs[1],
            valueMap,
            iter,
            axes,
            outShape,
            false,
            false,
            targetShapeNode,
            loopCtx,
        );

        let staticShapeA = asStaticDims(inputs[0].shape);
        let staticShapeB = asStaticDims(inputs[1].shape);

        if (staticShapeA.length === 1) staticShapeA = [1, staticShapeA[0]];
        if (staticShapeB.length === 1) staticShapeB = [staticShapeB[0], 1];

        // 2. Extract M, K, N (statically or dynamically)
        let M: number | ValueNode, K: number | ValueNode, N: number | ValueNode;
        const isDynamicA = inputs[0].shape.includes(-1) || inputs[0].shape.length === 0;
        const isDynamicB = inputs[1].shape.includes(-1) || inputs[1].shape.length === 0;

        if (isDynamicA || isDynamicB) {
            const getDimNode = (
                tensor: ValueNode,
                rankNode: ValueNode,
                offsetFromEnd: number,
                tag: string,
            ) => {
                const [shapeNode] = builder.createOp("Shape", [tensor]);
                const offsetNode = builder.createConstant(
                    `${tag}_offset`,
                    scalarInt64(offsetFromEnd),
                );
                const [targetAxis] = builder.createOp("Add", [rankNode, offsetNode]);
                const [dimRaw] = builder.createOp(
                    "Gather",
                    [shapeNode, builder.createOp("Unsqueeze", [targetAxis, axes])[0]],
                    { axis: 0 },
                );
                return squeezeIfLen1(builder, dimRaw, axes, `${tag}_sq`);
            };

            const [rankA] = builder.createOp("Size", [builder.createOp("Shape", [tInnerA])[0]]);
            const [rankB] = builder.createOp("Size", [builder.createOp("Shape", [tInnerB])[0]]);

            M = isDynamicA
                ? getDimNode(tInnerA, rankA, -2, "M")
                : staticShapeA[staticShapeA.length - 2];
            K = isDynamicA
                ? getDimNode(tInnerA, rankA, -1, "K")
                : staticShapeA[staticShapeA.length - 1];
            N = isDynamicB
                ? getDimNode(tInnerB, rankB, -1, "N")
                : staticShapeB[staticShapeB.length - 1];
        } else {
            M = staticShapeA[staticShapeA.length - 2];
            K = staticShapeA[staticShapeA.length - 1];
            N = staticShapeB[staticShapeB.length - 1];
        }

        const batchA = staticShapeA.slice(0, -2);
        const batchB = staticShapeB.slice(0, -2);
        const batchOut = broadcastShapes(...[batchA as number[], batchB as number[]]);

        let iIdx: ConcreteValueNode;
        let jIdx: ConcreteValueNode;
        let bOffsetA: ConcreteValueNode;
        let bOffsetB: ConcreteValueNode;

        // 3. Address Generation: Pointer Mode (AGU) vs Radix Mode (Div/Mod)
        if (loopCtx && loopCtx.indexingMode === IndexingMode.Pointer) {
            const iterDims: (number | ValueNode)[] = [...batchOut, M, N];
            const { counters } = getOrCreateIterationSpaceCounters(
                builder,
                loopCtx,
                valueMap,
                iterDims,
                `mm_space_${op.id}`,
            );

            iIdx = counters[counters.length - 2];
            jIdx = counters[counters.length - 1];

            const buildBatchAgu = (targetBatch: number[], tag: string): ConcreteValueNode => {
                if (targetBatch.length === 0) {
                    return builder.createConstant(`${tag}_zero`, scalarInt64(0));
                }
                const rawStrides = computeStrides(targetBatch);
                const alignedBatchStrides = batchOut.map((_, outPos) => {
                    const inPos = outPos - (batchOut.length - targetBatch.length);
                    if (inPos < 0 || targetBatch[inPos] === 1) return 0;
                    return rawStrides[inPos];
                });
                // Stride 0 on M and N so batch index only increments when an M x N tile wraps
                return buildAguPointer(
                    builder,
                    loopCtx,
                    valueMap,
                    iterDims,
                    [...alignedBatchStrides, 0, 0],
                    `${tag}_${op.id}`,
                );
            };

            bOffsetA = buildBatchAgu(batchA, "bA");
            bOffsetB = buildBatchAgu(batchB, "bB");
        } else {
            let MNConst: ValueNode, NConst: ValueNode;
            if (typeof M === "number" && typeof N === "number") {
                MNConst = builder.createConstant(`MN`, scalarInt64(M * N));
                NConst = builder.createConstant(`N`, scalarInt64(N));
            } else {
                const mNode =
                    typeof M === "number"
                        ? builder.createConstant(`M_const`, scalarInt64(M))
                        : (M as ValueNode);
                NConst =
                    typeof N === "number"
                        ? builder.createConstant(`N_const`, scalarInt64(N))
                        : (N as ValueNode);
                [MNConst] = builder.createOp("Mul", [mNode, NConst]);
            }

            const [batchIter] = builder.createOp("Div", [iter, MNConst]);
            const [remMN] = builder.createOp("Mod", [iter, MNConst]);
            [iIdx] = builder.createOp("Div", [remMN, NConst]);
            [jIdx] = builder.createOp("Mod", [remMN, NConst]);

            const getBatchOffset = (targetBatch: number[], tag: string) => {
                if (targetBatch.length === 0)
                    return builder.createConstant(`${tag}_zero`, scalarInt64(0));
                const batchIndices = decodeMixedRadix(
                    builder,
                    batchIter,
                    batchOut,
                    `${tag}_decode`,
                );
                const actualIndices = targetBatch.map((dim, i) => {
                    const outPos = batchOut.length - targetBatch.length + i;
                    return dim === 1
                        ? builder.createConstant(`${tag}_dim_${i}_zero`, scalarInt64(0))
                        : batchIndices[outPos];
                });
                return buildLinearIndex(
                    builder,
                    actualIndices,
                    computeStrides(targetBatch),
                    `${tag}_offset`,
                );
            };

            bOffsetA = getBatchOffset(batchA, "bA");
            bOffsetB = getBatchOffset(batchB, "bB");
        }

        // 4. Reshape inputs to 3D [-1, M, K] and [-1, K, N]
        const build3DShape = (dim1: number | ValueNode, dim2: number | ValueNode, tag: string) => {
            const minusOne = builder.createConstant(`${tag}_m1`, int64Vec([-1]));
            const d1 =
                typeof dim1 === "number"
                    ? builder.createConstant(`${tag}_d1`, int64Vec([dim1]))
                    : builder.createOp("Unsqueeze", [dim1, axes])[0];
            const d2 =
                typeof dim2 === "number"
                    ? builder.createConstant(`${tag}_d2`, int64Vec([dim2]))
                    : builder.createOp("Unsqueeze", [dim2, axes])[0];
            return builder.createOp("Concat", [minusOne, d1, d2], { axis: 0 })[0];
        };

        const [A3D] = builder.createOp("Reshape", [tInnerA, build3DShape(M, K, "A")]);
        const [B3D] = builder.createOp("Reshape", [tInnerB, build3DShape(K, N, "B")]);

        // 5. Gather the Matrix, Row, and Column for this iteration
        const [bA_unsq] = builder.createOp("Unsqueeze", [bOffsetA, axes]);
        const [AMatrix] = builder.createOp("Gather", [A3D, bA_unsq], { axis: 0 });
        const [bB_unsq] = builder.createOp("Unsqueeze", [bOffsetB, axes]);
        const [BMatrix] = builder.createOp("Gather", [B3D, bB_unsq], { axis: 0 });

        const [iIdx_unsq] = builder.createOp("Unsqueeze", [iIdx, axes]);
        const [rowA] = builder.createOp("Gather", [AMatrix, iIdx_unsq], { axis: 1 });

        const [jIdx_unsq] = builder.createOp("Unsqueeze", [jIdx, axes]);
        const [colB] = builder.createOp("Gather", [BMatrix, jIdx_unsq], { axis: 2 });

        const [rowA_flat] = builder.createOp("Reshape", [
            rowA,
            builder.createConstant("K_flat_A", int64Vec([-1])),
        ]);
        const [colB_flat] = builder.createOp("Reshape", [
            colB,
            builder.createConstant("K_flat_B", int64Vec([-1])),
        ]);

        // 6. Compute dot product
        const [mulOp] = builder.createOp("Mul", [rowA_flat, colB_flat]);
        const [sumOut] = builder.createOp("ReduceSum", [mulOp], { keepdims: 0 });

        return sumOut;
    }
}
