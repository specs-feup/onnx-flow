import type OperationNode from "../../../OperationNode.js";
import { type ConcreteValueNode, type KnownShape, DataType } from "../../../OnnxTypes.js";
import {
    bool,
    int64Vec,
    makeTensorProto,
    scalarInt64,
    toStaticShape,
} from "../../../utils/index.js";
import type { LoopLoweringRecipe } from "../LoopLoweringRecipe.js";
import { buildForLoopRegion } from "../LoopBuilder.js";
import {
    ensureInnerValueNode,
    extractMixedDims,
    gatherScalar,
    mulMixed,
    packShapeVector,
} from "../RecipeUtils.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { IndexingMode } from "../../../../DecompositionOptions.js";

/**
 * Loop-lowered Gaussian elimination with partial pivoting for Det([...batch, N, N]).
 * Keeps O(1) graph size (5 nested Loops) for arbitrary static or dynamic matrix size N and batch dims.
 */
export class LowerDetLoopRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerDetLoop";
    public readonly targetOp = "Det";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["linear-algebra", "nested-loop"] as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "Reshape",
        "Concat",
        "Gather",
        "ScatterElements",
        "Unsqueeze",
        "Squeeze",
        "ReduceProd",
        "Identity",
        "Abs",
        "Greater",
        "Equal",
        "Not",
        "Where",
        "Neg",
        "Add",
        "Sub",
        "Mul",
        "Div",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Det") return false;
        const X = op.getInputs()?.[0];
        if (!X || !Array.isArray(X.shape) || X.shape.length < 2) return false;
        const dims = toStaticShape(X.shape);
        const n = dims[dims.length - 1];
        const m = dims[dims.length - 2];
        if (n !== -1 && n < 1) return false;
        if (m !== -1 && m < 1) return false;
        if (n !== -1 && m !== -1 && n !== m) return false;
        return true;
    }

    apply(): never {
        throw new Error("LowerDetLoop uses lowerStandalone");
    }

    lowerStandalone(
        op: OperationNode.Class,
        outerBuilder: GraphBuilder,
        indexingMode: IndexingMode,
    ): void {
        const X = op.getInputs()![0] as ConcreteValueNode;
        const Y = op.getOutputs()[0];
        const dtype = (X.literalType || Y.literalType || DataType.FLOAT) as DataType;
        const rawDims = toStaticShape(X.shape);
        const rank = rawDims.length;
        const batchRawDims = rawDims.slice(0, rank - 2);

        const xMixed = extractMixedDims(outerBuilder, X, rawDims, `detl_d_${op.id}`);
        const nMixed = xMixed[rank - 1] !== -1 ? xMixed[rank - 1] : xMixed[rank - 2];
        const nnMixed = mulMixed(outerBuilder, nMixed, nMixed, `detl_nn_${op.id}`);

        let numBatchesMixed: number | ConcreteValueNode = 1;
        for (let i = 0; i < rank - 2; i++) {
            numBatchesMixed = mulMixed(
                outerBuilder,
                numBatchesMixed,
                xMixed[i],
                `detl_nb_${i}_${op.id}`,
            );
        }

        const shape2d = packShapeVector(
            outerBuilder,
            [numBatchesMixed, nnMixed],
            `detl_2d_${op.id}`,
        );
        const nnStaticOrDyn = typeof nnMixed === "number" ? nnMixed : -1;
        const [x2d] = outerBuilder.createOp("Reshape", [X, shape2d], {}, [
            {
                type: dtype,
                shape: [typeof numBatchesMixed === "number" ? numBatchesMixed : -1, nnStaticOrDyn],
            },
        ]);

        const nOuter: ConcreteValueNode =
            typeof nMixed === "number"
                ? outerBuilder.createConstant(`detl_n_${op.id}`, scalarInt64(nMixed))
                : nMixed;
        const zeroOuter = outerBuilder.createConstant(
            `detl_zf_${op.id}`,
            makeTensorProto(dtype, [], [0]),
        );
        const oneOuter = outerBuilder.createConstant(
            `detl_of_${op.id}`,
            makeTensorProto(dtype, [], [1]),
        );
        const negOneAbsOuter = outerBuilder.createConstant(
            `detl_n1f_${op.id}`,
            makeTensorProto(dtype, [], [-1]),
        );
        const zeroI64Outer = outerBuilder.createConstant(`detl_z64_${op.id}`, scalarInt64(0));
        const oneI64Outer = outerBuilder.createConstant(`detl_o64_${op.id}`, scalarInt64(1));
        const oneVecShapeOuter = outerBuilder.createConstant(`detl_1v_${op.id}`, int64Vec([1]));

        const batchLoop = buildForLoopRegion(
            outerBuilder,
            numBatchesMixed,
            dtype,
            [1],
            `detl_batch_${op.id}`,
            `det_${op.id}`,
            indexingMode,
            false,
        );
        const bBuilder = batchLoop.innerBuilder;
        const innerX2d = ensureInnerValueNode(bBuilder, x2d);
        const [mat0] = bBuilder.createOp("Gather", [innerX2d, batchLoop.trip], { axis: 0 }, [
            { type: dtype, shape: [nnStaticOrDyn] },
        ]);
        const nInB = ensureInnerValueNode(bBuilder, nOuter);
        const oneInB = ensureInnerValueNode(bBuilder, oneOuter);
        const zeroInB = ensureInnerValueNode(bBuilder, zeroOuter);
        const neg1InB = ensureInnerValueNode(bBuilder, negOneAbsOuter);
        const z64InB = ensureInnerValueNode(bBuilder, zeroI64Outer);
        const o64InB = ensureInnerValueNode(bBuilder, oneI64Outer);
        const oneShInB = ensureInnerValueNode(bBuilder, oneVecShapeOuter);

        // Outer Gaussian elimination loop over column k = 0..N-1
        const kLoop = buildForLoopRegion(
            bBuilder,
            nInB,
            dtype,
            [1],
            `detl_k_${op.id}`,
            `detl_k_${op.id}`,
            indexingMode,
            false,
        );
        const kBuilder = kLoop.innerBuilder;
        const matCarry = kLoop.loopCtx.addCarry(`detl_mat_${op.id}`, mat0, dtype, [nnStaticOrDyn]);
        const detCarry = kLoop.loopCtx.addCarry(`detl_acc_${op.id}`, oneInB, dtype, []);

        // Alias kLoop.trip away from "iter" so nested inner loops never shadow k via ensureInnerValueNode
        const [k] = kBuilder.createOp("Identity", [kLoop.trip]);
        const nInK = ensureInnerValueNode(kBuilder, nInB);
        const zeroInK = ensureInnerValueNode(kBuilder, zeroInB);
        const oneInK = ensureInnerValueNode(kBuilder, oneInB);
        const neg1InK = ensureInnerValueNode(kBuilder, neg1InB);
        const z64InK = ensureInnerValueNode(kBuilder, z64InB);
        const o64InK = ensureInnerValueNode(kBuilder, o64InB);
        const oneShInK = ensureInnerValueNode(kBuilder, oneShInB);

        const [rem] = kBuilder.createOp("Sub", [nInK, k]);
        const [kPlus1] = kBuilder.createOp("Add", [k, o64InK]);

        // 1. Pivot search over rows r = k..N-1
        const pivLoop = buildForLoopRegion(
            kBuilder,
            rem,
            dtype,
            [1],
            `detl_piv_${op.id}`,
            `detl_piv_${op.id}`,
            indexingMode,
            false,
        );
        const pBuilder = pivLoop.innerBuilder;
        const maxAbsIn = pivLoop.loopCtx.addCarry(`detl_mabs_${op.id}`, neg1InK, dtype, []);
        const pivRowIn = pivLoop.loopCtx.addCarry(`detl_prow_${op.id}`, k, DataType.INT64, []);
        const pMat = ensureInnerValueNode(pBuilder, matCarry);
        const pN = ensureInnerValueNode(pBuilder, nInK);
        const pK = ensureInnerValueNode(pBuilder, k);

        const [rCur] = pBuilder.createOp("Add", [pK, pivLoop.trip]);
        const [rStride] = pBuilder.createOp("Mul", [rCur, pN]);
        const [colKIdx] = pBuilder.createOp("Add", [rStride, pK]);
        const colVal = gatherScalar(pBuilder, pMat, colKIdx);
        const [absVal] = pBuilder.createOp("Abs", [colVal]);
        const [isBetter] = pBuilder.createOp("Greater", [absVal, maxAbsIn]);
        const [maxAbsNext] = pBuilder.createOp("Where", [isBetter, absVal, maxAbsIn]);
        const [pivRowNext] = pBuilder.createOp("Where", [isBetter, rCur, pivRowIn]);
        pivLoop.loopCtx.setNextState(maxAbsIn, maxAbsNext);
        pivLoop.loopCtx.setNextState(pivRowIn, pivRowNext);
        pivLoop.finalize([]);

        const pivRow = pivLoop.loopOutputs[1];

        // 2. Flip det sign when pivRow != k and multiply det by chosen pivot element
        const [noSwap] = kBuilder.createOp("Equal", [pivRow, k]);
        const [didSwap] = kBuilder.createOp("Not", [noSwap]);
        const [negDet] = kBuilder.createOp("Neg", [detCarry]);
        const [detSigned] = kBuilder.createOp("Where", [didSwap, negDet, detCarry]);

        const [pivStride] = kBuilder.createOp("Mul", [pivRow, nInK]);
        const [pivIdx] = kBuilder.createOp("Add", [pivStride, k]);
        const pivVal = gatherScalar(kBuilder, matCarry, pivIdx);
        const [detNext] = kBuilder.createOp("Mul", [detSigned, pivVal]);

        const [isZeroPiv] = kBuilder.createOp("Equal", [pivVal, zeroInK]);
        const [safePiv] = kBuilder.createOp("Where", [isZeroPiv, oneInK, pivVal]);

        // 3. Swap row k and pivRow across columns c = k..N-1
        const swapLoop = buildForLoopRegion(
            kBuilder,
            rem,
            dtype,
            [1],
            `detl_swp_${op.id}`,
            `detl_swp_${op.id}`,
            indexingMode,
            false,
        );
        const sBuilder = swapLoop.innerBuilder;
        const matSwapIn = swapLoop.loopCtx.addCarry(`detl_mswp_${op.id}`, matCarry, dtype, [
            nnStaticOrDyn,
        ]);
        const sN = ensureInnerValueNode(sBuilder, nInK);
        const sK = ensureInnerValueNode(sBuilder, k);
        const sPivRow = ensureInnerValueNode(sBuilder, pivRow);
        const sOneSh = ensureInnerValueNode(sBuilder, oneShInK);

        const [cCur] = sBuilder.createOp("Add", [sK, swapLoop.trip]);
        const [kRowBase] = sBuilder.createOp("Mul", [sK, sN]);
        const [pRowBase] = sBuilder.createOp("Mul", [sPivRow, sN]);
        const [idxK] = sBuilder.createOp("Add", [kRowBase, cCur]);
        const [idxP] = sBuilder.createOp("Add", [pRowBase, cCur]);
        const vK = gatherScalar(sBuilder, matSwapIn, idxK);
        const vP = gatherScalar(sBuilder, matSwapIn, idxP);

        const [idxK1] = sBuilder.createOp("Reshape", [idxK, sOneSh]);
        const [idxP1] = sBuilder.createOp("Reshape", [idxP, sOneSh]);
        const [vK1] = sBuilder.createOp("Reshape", [vK, sOneSh]);
        const [vP1] = sBuilder.createOp("Reshape", [vP, sOneSh]);
        const [matMid] = sBuilder.createOp("ScatterElements", [matSwapIn, idxK1, vP1], { axis: 0 });
        const [matSwapNext] = sBuilder.createOp("ScatterElements", [matMid, idxP1, vK1], {
            axis: 0,
        });
        swapLoop.loopCtx.setNextState(matSwapIn, matSwapNext);
        swapLoop.finalize([]);

        const matSwapped = swapLoop.loopOutputs[0];

        // 4. Rank-1 elimination over (rem - 1) x (rem - 1) trailing submatrix without Div/Mod
        const [remMinus1] = kBuilder.createOp("Sub", [rem, o64InK]);
        const [elimIters] = kBuilder.createOp("Mul", [remMinus1, remMinus1]);
        const elimLoop = buildForLoopRegion(
            kBuilder,
            elimIters,
            dtype,
            [1],
            `detl_el_${op.id}`,
            `detl_el_${op.id}`,
            indexingMode,
            false,
        );
        const eBuilder = elimLoop.innerBuilder;
        const matElimIn = elimLoop.loopCtx.addCarry(`detl_mel_${op.id}`, matSwapped, dtype, [
            nnStaticOrDyn,
        ]);
        const rOffIn = elimLoop.loopCtx.addCarry(`detl_roff_${op.id}`, z64InK, DataType.INT64, []);
        const cOffIn = elimLoop.loopCtx.addCarry(`detl_coff_${op.id}`, z64InK, DataType.INT64, []);

        const eN = ensureInnerValueNode(eBuilder, nInK);
        const eK = ensureInnerValueNode(eBuilder, k);
        const eKp1 = ensureInnerValueNode(eBuilder, kPlus1);
        const eSafePiv = ensureInnerValueNode(eBuilder, safePiv);
        const eRemM1 = ensureInnerValueNode(eBuilder, remMinus1);
        const eZero = ensureInnerValueNode(eBuilder, z64InK);
        const eOne = ensureInnerValueNode(eBuilder, o64InK);
        const eOneSh = ensureInnerValueNode(eBuilder, oneShInK);

        const [r] = eBuilder.createOp("Add", [eKp1, rOffIn]);
        const [c] = eBuilder.createOp("Add", [eKp1, cOffIn]);
        const [rBase] = eBuilder.createOp("Mul", [r, eN]);
        const [kBase] = eBuilder.createOp("Mul", [eK, eN]);

        const [aRkIdx] = eBuilder.createOp("Add", [rBase, eK]);
        const aRk = gatherScalar(eBuilder, matElimIn, aRkIdx);
        const [factor] = eBuilder.createOp("Div", [aRk, eSafePiv]);

        const [aKcIdx] = eBuilder.createOp("Add", [kBase, c]);
        const aKc = gatherScalar(eBuilder, matElimIn, aKcIdx);

        const [aRcIdx] = eBuilder.createOp("Add", [rBase, c]);
        const aRc = gatherScalar(eBuilder, matElimIn, aRcIdx);

        const [upd] = eBuilder.createOp("Mul", [factor, aKc]);
        const [newRc] = eBuilder.createOp("Sub", [aRc, upd]);
        const [aRcIdx1] = eBuilder.createOp("Reshape", [aRcIdx, eOneSh]);
        const [newRc1] = eBuilder.createOp("Reshape", [newRc, eOneSh]);
        const [matElimNext] = eBuilder.createOp("ScatterElements", [matElimIn, aRcIdx1, newRc1], {
            axis: 0,
        });

        const [cInc] = eBuilder.createOp("Add", [cOffIn, eOne]);
        const [wrap] = eBuilder.createOp("Equal", [cInc, eRemM1]);
        const [cNext] = eBuilder.createOp("Where", [wrap, eZero, cInc]);
        const [rInc] = eBuilder.createOp("Add", [rOffIn, eOne]);
        const [rNext] = eBuilder.createOp("Where", [wrap, rInc, rOffIn]);

        elimLoop.loopCtx.setNextState(matElimIn, matElimNext);
        elimLoop.loopCtx.setNextState(rOffIn, rNext);
        elimLoop.loopCtx.setNextState(cOffIn, cNext);
        elimLoop.finalize([]);

        kLoop.loopCtx.setNextState(matCarry, elimLoop.loopOutputs[0]);
        kLoop.loopCtx.setNextState(detCarry, detNext);
        kLoop.finalize([]);

        const detFinal = kLoop.loopOutputs[1];
        batchLoop.finalize([], [{ tag: "det", node: detFinal, elemTy: dtype, shape: [] }]);
        const rawLoopOut = batchLoop.loopOutputs[0];

        let finalOut: ConcreteValueNode;
        if (rank === 2) {
            const sqAx = outerBuilder.createConstant(`detl_sq0_${op.id}`, int64Vec([0]));
            [finalOut] = outerBuilder.createOp("Squeeze", [rawLoopOut, sqAx], {}, [
                { type: dtype, shape: [] },
            ]);
        } else {
            const batchShapeVec = packShapeVector(
                outerBuilder,
                xMixed.slice(0, rank - 2),
                `detl_bsh_${op.id}`,
            );
            [finalOut] = outerBuilder.createOp("Reshape", [rawLoopOut, batchShapeVec], {}, [
                { type: dtype, shape: batchRawDims as KnownShape },
            ]);
        }

        // Guard unused outer constant if rank===2 never touched bool
        void bool;
        outerBuilder.replaceAllUsesWith(Y, finalOut);
        outerBuilder.removeNode(op);
    }
}
