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
    makeTensorProto,
    computeStrides,
    toStaticShape,
} from "../../../utils/index.js";
import type {
    LoopCarryContext,
    LoopLoweringRecipe,
    RecipeApplyResult,
} from "../LoopLoweringRecipe.js";
import {
    resolveRecipeInput,
    ensureFlatInput,
    gatherScalar,
    buildLinearIndex,
    decodeMixedRadix,
    buildAguPointer,
    getOrCreateIterationSpaceCounters,
    computeMatMulOutputDimsMixed,
    mulMixed,
    packShapeVector,
} from "../RecipeUtils.js";
import { GraphBuilder } from "../../../GraphBuilder.js";
import { IndexingMode } from "../../../../DecompositionOptions.js";

export class LowerCoalescedMatMulRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerCoalescedMatMul";
    public readonly targetOp = "MatMul";
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 60;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["linear-algebra", "coalesced", "self-carry"] as const;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "Size",
        "Gather",
        "ScatterElements",
        "Unsqueeze",
        "Squeeze",
        "Reshape",
        "Add",
        "Mul",
        "Div",
        "Mod",
        "Equal",
        "And",
        "Where",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "MatMul") return false;
        const inputs = op.getInputs();
        if (!inputs || inputs.length < 2) return false;
        const sA = toStaticShape(inputs[0].shape);
        const sB = toStaticShape(inputs[1].shape);
        if (sA.length === 0 || sB.length === 0) return false;
        if (sA.slice(0, -2).includes(-1) || sB.slice(0, -2).includes(-1)) return false;
        return true;
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

        // 2. Dynamic Case: O(rank) INT64 MatMul output & contraction bounds (no dummy tensors)
        const builder = new GraphBuilder(op.graph as OnnxGraph.Class, `coal_bounds_${op.id}`);
        const axes0 = builder.createConstant(`axes0_${op.id}`, int64Vec([0]));

        const { outDims, kDim } = computeMatMulOutputDimsMixed(builder, A, B, `coalb_${op.id}`);
        const targetShapeNode =
            outDims.length === 0
                ? builder.createConstant(`coalb_empty_${op.id}`, int64Vec([]))
                : packShapeVector(builder, outDims, `coalb_tsh_${op.id}`);

        const carrySize: number | ConcreteValueNode =
            outDims.length === 0
                ? 1
                : builder.createOp("ReduceProd", [targetShapeNode, axes0], { keepdims: 0 })[0];
        const carryShape =
            typeof carrySize === "number"
                ? builder.createConstant(`coalb_csh_${op.id}`, int64Vec([carrySize]))
                : builder.createOp("Unsqueeze", [carrySize, axes0])[0];

        const totalItersMixed = mulMixed(builder, carrySize, kDim, `coalb_it_${op.id}`);
        const totalIters =
            typeof totalItersMixed === "number"
                ? builder.createConstant(`coalb_tot_${op.id}`, scalarInt64(totalItersMixed))
                : totalItersMixed;

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
        targetShapeNode?: ValueNode,
        loopCtx?: LoopCarryContext,
    ): RecipeApplyResult {
        const builder = new GraphBuilder(body, `coal_mm_${op.id}`);
        const inputs = op.getInputs()!;

        // 1. Resolve and capture inputs inside the loop body first
        const rawInnerA = resolveRecipeInput(
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
        const rawInnerB = resolveRecipeInput(
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

        let sA = asStaticDims(inputs[0].shape),
            sB = asStaticDims(inputs[1].shape);
        if (sA.length === 1) sA = [1, sA[0]];
        if (sB.length === 1) sB = [sB[0], 1];

        // 2. Extract M, K, N (statically or dynamically from captured inner nodes)
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
                return gatherScalar(builder, shapeNode, targetAxis);
            };

            const [rankA] = builder.createOp("Size", [builder.createOp("Shape", [rawInnerA])[0]]);
            const [rankB] = builder.createOp("Size", [builder.createOp("Shape", [rawInnerB])[0]]);

            M = isDynamicA ? getDimNode(rawInnerA, rankA, -2, "M") : sA[sA.length - 2];
            K = isDynamicA ? getDimNode(rawInnerA, rankA, -1, "K") : sA[sA.length - 1];
            N = isDynamicB ? getDimNode(rawInnerB, rankB, -1, "N") : sB[sB.length - 1];
        } else {
            M = sA[sA.length - 2];
            K = sA[sA.length - 1];
            N = sB[sB.length - 1];
        }

        const bA = sA.slice(0, -2);
        const bB = sB.slice(0, -2);
        const batchOut = broadcastShapes(...[bA as number[], bB as number[]]);

        let idxA: ConcreteValueNode;
        let idxB: ConcreteValueNode;
        let flatOutIdx: ConcreteValueNode;
        let kIdxScalar: ConcreteValueNode;

        // 3. Address Generation: Pointer Mode (AGU) vs Radix Mode (Div/Mod)
        if (loopCtx && loopCtx.indexingMode === IndexingMode.Pointer) {
            const iterDims: (number | ValueNode)[] = [...batchOut, M, N, K];
            const { counters } = getOrCreateIterationSpaceCounters(
                builder,
                loopCtx,
                valueMap,
                iterDims,
                `mm_space_${op.id}`,
            );
            kIdxScalar = counters[counters.length - 1];

            const mulVal = (
                a: number | ValueNode,
                b: number | ValueNode,
                tag: string,
            ): number | ValueNode => {
                if (typeof a === "number" && typeof b === "number") return a * b;
                const aNode =
                    typeof a === "number" ? builder.createConstant(`${tag}_a`, scalarInt64(a)) : a;
                const bNode =
                    typeof b === "number" ? builder.createConstant(`${tag}_b`, scalarInt64(b)) : b;
                return builder.createOp("Mul", [aNode, bNode])[0];
            };

            const buildBatchStrides = (
                targetBatch: number[],
                matrixSize: number | ValueNode,
                tag: string,
            ): (number | ValueNode)[] => {
                if (batchOut.length === 0) return [];
                const rawStrides = targetBatch.length > 0 ? computeStrides(targetBatch) : [];
                return batchOut.map((_, outPos) => {
                    const inPos = outPos - (batchOut.length - targetBatch.length);
                    if (inPos < 0 || targetBatch[inPos] === 1) return 0;
                    return mulVal(rawStrides[inPos], matrixSize, `${tag}_bs_${outPos}`);
                });
            };

            const mkSize = mulVal(M, K, "MK");
            const knSize = mulVal(K, N, "KN");
            const mnSize = mulVal(M, N, "MN");

            // Strides across [...batchOut, M, N, K]
            const stridesA = [...buildBatchStrides(bA, mkSize, "A"), K, 0, 1];
            const stridesB = [...buildBatchStrides(bB, knSize, "B"), 0, 1, N];
            const stridesOut = [...buildBatchStrides(batchOut, mnSize, "C"), N, 1, 0];

            idxA = buildAguPointer(builder, loopCtx, valueMap, iterDims, stridesA, `mm_A_${op.id}`);
            idxB = buildAguPointer(builder, loopCtx, valueMap, iterDims, stridesB, `mm_B_${op.id}`);
            flatOutIdx = buildAguPointer(
                builder,
                loopCtx,
                valueMap,
                iterDims,
                stridesOut,
                `mm_C_${op.id}`,
            );
        } else {
            // Radix Mode: Decode multi-index Batch, I, J, K via Div and Mod
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
            kIdxScalar = kIdx;

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

            idxA = getFlatIdx(bA, iIdx, kIdx, M, K, "A");
            idxB = getFlatIdx(bB, kIdx, jIdx, K, N, "B");

            const [mnNode] = builder.createOp("Mul", [mNode, nNode]);
            flatOutIdx = buildLinearIndex(
                builder,
                [batchIter, iIdx, jIdx],
                [mnNode, nNode, 1],
                `out_idx`,
            );
        }

        // 4. Gather elements and compute partial product
        const flatA = ensureFlatInput(builder, rawInnerA);
        const flatB = ensureFlatInput(builder, rawInnerB);

        const valA = gatherScalar(builder, flatA, idxA);
        const valB = gatherScalar(builder, flatB, idxB);
        const [prod] = builder.createOp("Mul", [valA, valB]);

        // 5. Update the accumulator (0D scalar register when loopCtx is active to avoid carry Gather)
        const dtype = (inputs[0].literalType || DataType.FLOAT) as DataType;
        const zeroI64 = builder.createConstant(`mm_z64_${op.id}`, scalarInt64(0));
        const zeroVal = builder.createConstant(`mm_zval_${op.id}`, makeTensorProto(dtype, [], [0]));

        let baseAcc: ConcreteValueNode;
        if (loopCtx !== undefined) {
            const initAcc = loopCtx.outerBuilder.createConstant(
                `mm_init_acc_${op.id}`,
                makeTensorProto(dtype, [], [0]),
            );
            const accIn = loopCtx.addCarry(`mm_acc_${op.id}`, initAcc, dtype, []);
            const [isFirstK] = builder.createOp("Equal", [kIdxScalar, zeroI64]);
            [baseAcc] = builder.createOp("Where", [isFirstK, zeroVal, accIn]);
            const [newSum] = builder.createOp("Add", [baseAcc, prod]);
            loopCtx.setNextState(accIn, newSum);

            const [flatOutIdxUnsq] = builder.createOp("Unsqueeze", [flatOutIdx, axes]);
            const [scatterOut] = builder.createOp(
                "ScatterElements",
                [carryNode, flatOutIdxUnsq, builder.createOp("Unsqueeze", [newSum, axes])[0]],
                { axis: 0 },
            );

            return { resultNode: newSum, nextCarry: scatterOut };
        }

        const [flatOutIdxUnsq] = builder.createOp("Unsqueeze", [flatOutIdx, axes]);
        const currentAcc = gatherScalar(builder, carryNode, flatOutIdx);

        const [newSum] = builder.createOp("Add", [currentAcc, prod]);
        const [scatterOut] = builder.createOp(
            "ScatterElements",
            [carryNode, flatOutIdxUnsq, builder.createOp("Unsqueeze", [newSum, axes])[0]],
            { axis: 0 },
        );

        return { resultNode: newSum, nextCarry: scatterOut };
    }
}
