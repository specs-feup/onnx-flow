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
    scalarInt64,
    broadcastShapes,
    int64Vec,
    UNKNOWN_SHAPE,
} from "../../../Utils.js";
import type { LoopLoweringRecipe, RecipeApplyResult } from "../LoopLoweringRecipe.js";
import {
    resolveRecipeInput,
    squeezeIfLen1,
    ensureFlatInput,
    buildLinearIndex,
    decodeMixedRadix,
} from "../RecipeUtils.js";
import { GraphBuilder } from "../../../GraphBuilder.js";

export class LowerCoalescedMatMulRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerCoalescedMatMul";
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
        const [A, B] = inputs;

        // 1. Static Case
        if (
            !A.shape.includes(-1) &&
            !B.shape.includes(-1) &&
            A.shape.length > 0 &&
            B.shape.length > 0
        ) {
            let sA = asStaticDims(A.shape),
                sB = asStaticDims(B.shape);
            if (sA.length === 1) sA = [1, sA[0]];
            if (sB.length === 1) sB = [sB[0], 1];

            const M = sA[sA.length - 2],
                K = sA[sA.length - 1],
                N = sB[sB.length - 1];
            const batchOut = broadcastShapes(...[sA.slice(0, -2), sB.slice(0, -2)]);
            const prodBatch = batchOut.reduce((a, b) => a * b, 1);

            return { totalIters: prodBatch * M * N * K, carryShape: [prodBatch * M * N] };
        }

        // 2. Dynamic Case (Dummy tensor approach)
        const builder = new GraphBuilder(op.graph as OnnxGraph.Class, `coal_bounds_${op.id}`);
        const axes0 = builder.createConstant(`axes0_${op.id}`, int64Vec([0]));

        const [shapeA] = builder.createOp("Shape", [A]);
        const [shapeB] = builder.createOp("Shape", [B]);

        const expectedCoS = [{ type: DataType.FLOAT, shape: UNKNOWN_SHAPE }];
        const [dummyA] = builder.createOp("ConstantOfShape", [shapeA], {}, expectedCoS);
        const [dummyB] = builder.createOp("ConstantOfShape", [shapeB], {}, expectedCoS);

        const expectedMM = [{ type: DataType.FLOAT, shape: UNKNOWN_SHAPE }];
        const [dummyOut] = builder.createOp("MatMul", [dummyA, dummyB], {}, expectedMM);

        // Get K dynamically from A
        const [rankA] = builder.createOp("Size", [shapeA]);
        const kOffset = builder.createConstant(`k_off_${op.id}`, scalarInt64(-1));
        const [kAxis] = builder.createOp("Add", [rankA, kOffset]);
        const [kValRaw] = builder.createOp(
            "Gather",
            [shapeA, builder.createOp("Unsqueeze", [kAxis, axes0])[0]],
            { axis: 0 },
        );
        const [kVal] = builder.createOp("Squeeze", [kValRaw, axes0]);

        const [targetShapeNode] = builder.createOp("Shape", [dummyOut]);

        // carryShape is the output shape size (batch * M * N)
        const [carrySize] = builder.createOp("ReduceProd", [targetShapeNode, axes0], {
            keepdims: 0,
        });
        const [carryShape] = builder.createOp("Unsqueeze", [carrySize, axes0]);

        // totalIters = carrySize * K
        const [totalIters] = builder.createOp("Mul", [carrySize, kVal]);

        return { totalIters, carryShape, targetShape: targetShapeNode };
    }

    apply(
        op: OperationNode.Class,
        body: OnnxGraph.Class,
        valueMap: Map<string, ValueNode>,
        iter: ConcreteValueNode,
        axes: ConcreteValueNode,
        outShape: KnownShape,
        carryNode: ConcreteValueNode,
        targetShapeNode: ValueNode,
    ): RecipeApplyResult {
        const builder = new GraphBuilder(body, `coal_mm_${op.id}`);
        const inputs = op.getInputs()!;

        let sA = asStaticDims(inputs[0].shape),
            sB = asStaticDims(inputs[1].shape);
        if (sA.length === 1) sA = [1, sA[0]];
        if (sB.length === 1) sB = [sB[0], 1];

        // --- DYNAMIC SUPPORT: Extract M, K, N ---
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

            const [rankA] = builder.createOp("Size", [builder.createOp("Shape", [inputs[0]])[0]]);
            const [rankB] = builder.createOp("Size", [builder.createOp("Shape", [inputs[1]])[0]]);

            M = isDynamicA ? getDimNode(inputs[0], rankA, -2, "M") : sA[sA.length - 2];
            K = isDynamicA ? getDimNode(inputs[0], rankA, -1, "K") : sA[sA.length - 1];
            N = isDynamicB ? getDimNode(inputs[1], rankB, -1, "N") : sB[sB.length - 1];
        } else {
            M = sA[sA.length - 2];
            K = sA[sA.length - 1];
            N = sB[sB.length - 1];
        }

        const bA = sA.slice(0, -2);
        const bB = sB.slice(0, -2);
        const batchOut = broadcastShapes(...[bA as number[], bB as number[]]);

        // 1. Decode multi-index: Batch, I, J, K
        const mNode =
            typeof M === "number"
                ? builder.createConstant(`M_const`, scalarInt64(M))
                : (M as ValueNode);
        const kNode =
            typeof K === "number"
                ? builder.createConstant(`K_const`, scalarInt64(K))
                : (K as ValueNode);
        const nNode =
            typeof N === "number"
                ? builder.createConstant(`N_const`, scalarInt64(N))
                : (N as ValueNode);

        const [nkNode] = builder.createOp("Mul", [nNode, kNode]);
        const [mnkNode] = builder.createOp("Mul", [mNode, nkNode]);

        const [batchIter] = builder.createOp("Div", [iter, mnkNode]);
        const [remMNK] = builder.createOp("Mod", [iter, mnkNode]);
        const [iIdx] = builder.createOp("Div", [remMNK, nkNode]);
        const [remNK] = builder.createOp("Mod", [remMNK, nkNode]);
        const [jIdx] = builder.createOp("Div", [remNK, kNode]);
        const [kIdx] = builder.createOp("Mod", [remNK, kNode]);

        // 2. Map coordinates to flat input indices for A and B
        const getFlatIdx = (
            targetBatch: (number | ValueNode)[],
            rowIdx: ValueNode,
            colIdx: ValueNode,
            rows: number | ValueNode,
            cols: number | ValueNode,
            tag: string,
        ) => {
            let batchOffset: ValueNode = builder.createConstant(
                `${tag}_batch_zero`,
                scalarInt64(0),
            );

            if (targetBatch.length > 0) {
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

                // --- DYNAMIC STRIDE COMPUTATION ---
                const strides: (number | ValueNode)[] = new Array(targetBatch.length).fill(1);
                let currentStride: ValueNode | number = 1;
                for (let i = targetBatch.length - 1; i >= 0; i--) {
                    strides[i] = currentStride;
                    if (i > 0) {
                        const dim = targetBatch[i];
                        if (typeof currentStride === "number" && typeof dim === "number") {
                            currentStride = currentStride * dim;
                        } else {
                            const cNode =
                                typeof currentStride === "number"
                                    ? builder.createConstant(
                                          `${tag}_cs_${i}`,
                                          scalarInt64(currentStride),
                                      )
                                    : currentStride;
                            const dNode =
                                typeof dim === "number"
                                    ? builder.createConstant(`${tag}_cd_${i}`, scalarInt64(dim))
                                    : (dim as ValueNode);
                            [currentStride] = builder.createOp("Mul", [cNode, dNode]);
                        }
                    }
                }
                // --------------------------------------

                batchOffset = buildLinearIndex(
                    builder,
                    actualIndices,
                    strides,
                    `${tag}_batch_offset`,
                );
            }

            const rowsNode =
                typeof rows === "number"
                    ? builder.createConstant(`${tag}_rows`, scalarInt64(rows))
                    : (rows as ValueNode);
            const colsNode =
                typeof cols === "number"
                    ? builder.createConstant(`${tag}_cols`, scalarInt64(cols))
                    : (cols as ValueNode);

            const [rcMul] = builder.createOp("Mul", [rowsNode, colsNode]);
            const [offset] = builder.createOp("Mul", [batchOffset, rcMul]);
            const [rowOff] = builder.createOp("Mul", [rowIdx, colsNode]);

            const [base] = builder.createOp("Add", [offset, rowOff]);
            return builder.createOp("Add", [base, colIdx])[0];
        };

        const idxA = getFlatIdx(bA, iIdx, kIdx, M, K, "A");
        const idxB = getFlatIdx(bB, kIdx, jIdx, K, N, "B");

        // 3. Gather elements and compute partial product
        const flatA = ensureFlatInput(
            builder,
            resolveRecipeInput(
                builder,
                inputs[0],
                valueMap,
                iter,
                axes,
                outShape,
                false,
                false,
                targetShapeNode,
            ),
        );
        const flatB = ensureFlatInput(
            builder,
            resolveRecipeInput(
                builder,
                inputs[1],
                valueMap,
                iter,
                axes,
                outShape,
                false,
                false,
                targetShapeNode,
            ),
        );

        const [gAOut] = builder.createOp(
            "Gather",
            [flatA, builder.createOp("Unsqueeze", [idxA, axes])[0]],
            { axis: 0 },
        );
        const [gBOut] = builder.createOp(
            "Gather",
            [flatB, builder.createOp("Unsqueeze", [idxB, axes])[0]],
            { axis: 0 },
        );

        const valA = squeezeIfLen1(builder, gAOut, axes, `sqA`);
        const valB = squeezeIfLen1(builder, gBOut, axes, `sqB`);
        const [prod] = builder.createOp("Mul", [valA, valB]);

        // 4. Update the accumulator (Carry state)
        const [mnNode] = builder.createOp("Mul", [mNode, nNode]);
        const flatOutIdx = buildLinearIndex(
            builder,
            [batchIter, iIdx, jIdx],
            [mnNode, nNode, 1], // Dynamically built strides
            `out_idx`,
        );
        const [flatOutIdxUnsq] = builder.createOp("Unsqueeze", [flatOutIdx, axes]);

        const [gAccOut] = builder.createOp("Gather", [carryNode, flatOutIdxUnsq], { axis: 0 });
        const currentAcc = squeezeIfLen1(builder, gAccOut, axes, `sqAcc`);

        const [newSum] = builder.createOp("Add", [currentAcc, prod]);
        const [scatterOut] = builder.createOp(
            "ScatterElements",
            [carryNode, flatOutIdxUnsq, builder.createOp("Unsqueeze", [newSum, axes])[0]],
            { axis: 0 },
        );

        return { resultNode: newSum, nextCarry: scatterOut };
    }
}
