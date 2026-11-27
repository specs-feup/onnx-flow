import type OperationNode from "../../../ir/OperationNode.js";
import { type ConcreteValueNode, type KnownShape, DataType } from "../../../ir/OnnxTypes.js";
import {
    bool,
    getIntAttr,
    int64Vec,
    makeTensorProto,
    normalizeAxis,
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
import type { GraphBuilder } from "../../../ir/GraphBuilder.js";
import type { IndexingMode } from "../../../../DecompositionOptions.js";

/**
 * Standalone nested scalar-loop lowering for TopK(X, K).
 * Executes in 3 constant nested Loops (rows -> k -> axis scan) regardless of K,
 * eliminating repeated ArgMax/ReduceMax/Concat barriers and sentinel collisions.
 */
export class LowerTopKScalarLoopRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerTopKScalarLoop";
    public readonly targetOp = "TopK";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["topk", "dynamic-k", "nested-loop"] as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "Transpose",
        "Reshape",
        "Concat",
        "Gather",
        "ScatterElements",
        "Unsqueeze",
        "ReduceProd",
        "Expand",
        "Identity",
        "Cast",
        "Greater",
        "Less",
        "And",
        "Or",
        "Not",
        "Where",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "TopK") return false;
        const ins = op.getInputs();
        return (
            !!ins &&
            ins.length >= 2 &&
            Array.isArray(ins[0].shape) &&
            ins[0].shape.length >= 1 &&
            op.getOutputs().length >= 2
        );
    }

    apply(): never {
        throw new Error("LowerTopKScalarLoop uses lowerStandalone");
    }

    lowerStandalone(
        op: OperationNode.Class,
        outerBuilder: GraphBuilder,
        indexingMode: IndexingMode,
    ): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const K = ins[1];
        const outputs = op.getOutputs();
        const YVals = outputs[0];
        const YIdxs = outputs[1];

        const dtype = (X.literalType || YVals.literalType || DataType.FLOAT) as DataType;
        const rawDims = toStaticShape(X.shape);
        const rank = rawDims.length;
        const axis = normalizeAxis(getIntAttr(op, "axis", -1), rank);
        const largest = getIntAttr(op, "largest", 1) !== 0;

        // Move target axis to trailing position so each slice [D] is contiguous
        const needsPermute = axis !== rank - 1;
        const perm = Array.from({ length: rank }, (_, i) => i).filter((i) => i !== axis);
        perm.push(axis);

        const workX: ConcreteValueNode = needsPermute
            ? outerBuilder.createOp("Transpose", [X], { perm })[0]
            : X;

        const permRawDims = perm.map((p) => rawDims[p]);
        const permDims = extractMixedDims(outerBuilder, workX, permRawDims, `topkl_d_${op.id}`);
        const dMixed = permDims[rank - 1];
        let numRowsMixed: number | ConcreteValueNode = 1;
        for (let i = 0; i < rank - 1; i++) {
            numRowsMixed = mulMixed(
                outerBuilder,
                numRowsMixed,
                permDims[i],
                `topkl_nr_${i}_${op.id}`,
            );
        }

        const shape2d = packShapeVector(outerBuilder, [numRowsMixed, dMixed], `topkl_2d_${op.id}`);
        const dStaticOrDyn = typeof dMixed === "number" ? dMixed : -1;
        const [workX2d] = outerBuilder.createOp("Reshape", [workX, shape2d], {}, [
            {
                type: dtype,
                shape: [typeof numRowsMixed === "number" ? numRowsMixed : -1, dStaticOrDyn],
            },
        ]);

        // Trip count K -> 0D INT64 scalar + 1D [1] vector for output reshaping
        const k64: ConcreteValueNode =
            K.literalType === DataType.INT64
                ? K
                : outerBuilder.createOp("Cast", [K], { to: DataType.INT64 }, [
                      { type: DataType.INT64, shape: toStaticShape(K.shape) as KnownShape },
                  ])[0];
        const scalarShapeConst = outerBuilder.createConstant(`topkl_scsh_${op.id}`, int64Vec([]));
        const [kTripOuter] = outerBuilder.createOp("Reshape", [k64, scalarShapeConst], {}, [
            { type: DataType.INT64, shape: [] },
        ]);

        const dTripOuter: ConcreteValueNode =
            typeof dMixed === "number"
                ? outerBuilder.createConstant(`topkl_dtrip_${op.id}`, scalarInt64(dMixed))
                : dMixed;
        const oneVecShapeOuter = outerBuilder.createConstant(`topkl_1v_${op.id}`, int64Vec([1]));
        const [dShape1D] = outerBuilder.createOp("Reshape", [dTripOuter, oneVecShapeOuter], {}, [
            { type: DataType.INT64, shape: [1] },
        ]);

        const falseOuter = outerBuilder.createConstant(`topkl_f_${op.id}`, bool(false));
        const trueVecOuter = outerBuilder.createConstant(
            `topkl_tv_${op.id}`,
            makeTensorProto(DataType.BOOL, [1], [1]),
        );
        const [initUsedOuter] = outerBuilder.createOp("Expand", [falseOuter, dShape1D], {}, [
            { type: DataType.BOOL, shape: [dStaticOrDyn] },
        ]);
        const zeroValOuter = outerBuilder.createConstant(
            `topkl_zv_${op.id}`,
            makeTensorProto(dtype, [], [0]),
        );
        const zeroI64Outer = outerBuilder.createConstant(`topkl_zi64_${op.id}`, scalarInt64(0));

        // 1. Outer loop over independent slices [0..numRows-1]
        const rowLoop = buildForLoopRegion(
            outerBuilder,
            numRowsMixed,
            dtype,
            [1],
            `topkl_rows_${op.id}`,
            `topk_${op.id}`,
            indexingMode,
            false,
        );
        const rBuilder = rowLoop.innerBuilder;
        const innerX2d = ensureInnerValueNode(rBuilder, workX2d);
        const [rowVec] = rBuilder.createOp("Gather", [innerX2d, rowLoop.trip], { axis: 0 }, [
            { type: dtype, shape: [dStaticOrDyn] },
        ]);
        const kRow = ensureInnerValueNode(rBuilder, kTripOuter);
        const dRow = ensureInnerValueNode(rBuilder, dTripOuter);
        const initUsedRow = ensureInnerValueNode(rBuilder, initUsedOuter);
        const falseRow = ensureInnerValueNode(rBuilder, falseOuter);
        const trueVecRow = ensureInnerValueNode(rBuilder, trueVecOuter);
        const zeroValRow = ensureInnerValueNode(rBuilder, zeroValOuter);
        const zeroI64Row = ensureInnerValueNode(rBuilder, zeroI64Outer);
        const oneVecRow = ensureInnerValueNode(rBuilder, oneVecShapeOuter);

        // 2. Step loop extracting one extremum per k = 0..K-1 via scan_outputs
        const stepLoop = buildForLoopRegion(
            rBuilder,
            kRow,
            dtype,
            [1],
            `topkl_k_${op.id}`,
            `topkl_k_${op.id}`,
            indexingMode,
            false,
        );
        const stBuilder = stepLoop.innerBuilder;
        const usedCarry = stepLoop.loopCtx.addCarry(
            `topkl_used_${op.id}`,
            initUsedRow,
            DataType.BOOL,
            [dStaticOrDyn],
        );
        const dStep = ensureInnerValueNode(stBuilder, dRow);
        const rowVecStep = ensureInnerValueNode(stBuilder, rowVec);
        const falseStep = ensureInnerValueNode(stBuilder, falseRow);
        const trueVecStep = ensureInnerValueNode(stBuilder, trueVecRow);
        const zeroValStep = ensureInnerValueNode(stBuilder, zeroValRow);
        const zeroI64Step = ensureInnerValueNode(stBuilder, zeroI64Row);
        const oneVecStep = ensureInnerValueNode(stBuilder, oneVecRow);

        // 3. Inner scalar scan over j = 0..D-1 finding next best unmasked element
        const scanLoop = buildForLoopRegion(
            stBuilder,
            dStep,
            dtype,
            [1],
            `topkl_j_${op.id}`,
            `topkl_j_${op.id}`,
            indexingMode,
            false,
        );
        const jBuilder = scanLoop.innerBuilder;
        const hasBestIn = scanLoop.loopCtx.addCarry(
            `topkl_hb_${op.id}`,
            falseStep,
            DataType.BOOL,
            [],
        );
        const bestValIn = scanLoop.loopCtx.addCarry(`topkl_bv_${op.id}`, zeroValStep, dtype, []);
        const bestIdxIn = scanLoop.loopCtx.addCarry(
            `topkl_bi_${op.id}`,
            zeroI64Step,
            DataType.INT64,
            [],
        );

        const jRowVec = ensureInnerValueNode(jBuilder, rowVecStep);
        const jUsed = ensureInnerValueNode(jBuilder, usedCarry);
        const isUsed = gatherScalar(jBuilder, jUsed, scanLoop.trip);
        const [canPick] = jBuilder.createOp("Not", [isUsed]);
        const curVal = gatherScalar(jBuilder, jRowVec, scanLoop.trip);
        const [isBetter] = jBuilder.createOp(largest ? "Greater" : "Less", [curVal, bestValIn]);
        const [noBestYet] = jBuilder.createOp("Not", [hasBestIn]);
        const [acceptCmp] = jBuilder.createOp("Or", [noBestYet, isBetter]);
        const [take] = jBuilder.createOp("And", [canPick, acceptCmp]);
        const [nextHasBest] = jBuilder.createOp("Or", [hasBestIn, canPick]);
        const [nextBestVal] = jBuilder.createOp("Where", [take, curVal, bestValIn]);
        const [nextBestIdx] = jBuilder.createOp("Where", [take, scanLoop.trip, bestIdxIn]);

        scanLoop.loopCtx.setNextState(hasBestIn, nextHasBest);
        scanLoop.loopCtx.setNextState(bestValIn, nextBestVal);
        scanLoop.loopCtx.setNextState(bestIdxIn, nextBestIdx);
        scanLoop.finalize([]);

        const pickedVal = scanLoop.loopOutputs[1];
        const pickedIdx = scanLoop.loopOutputs[2];
        const [pickedIdx1] = stBuilder.createOp("Reshape", [pickedIdx, oneVecStep]);
        const [nextUsed] = stBuilder.createOp(
            "ScatterElements",
            [usedCarry, pickedIdx1, trueVecStep],
            { axis: 0 },
        );
        stepLoop.loopCtx.setNextState(usedCarry, nextUsed);
        stepLoop.finalize(
            [],
            [
                { tag: "kval", node: pickedVal, elemTy: dtype, shape: [] },
                { tag: "kidx", node: pickedIdx, elemTy: DataType.INT64, shape: [] },
            ],
        );

        const kOutShape = toStaticShape(YVals.shape);
        const kDimKnown = kOutShape.length > axis && kOutShape[axis] > 0 ? kOutShape[axis] : -1;
        rowLoop.finalize(
            [],
            [
                { tag: "rval", node: stepLoop.loopOutputs[1], elemTy: dtype, shape: [kDimKnown] },
                {
                    tag: "ridx",
                    node: stepLoop.loopOutputs[2],
                    elemTy: DataType.INT64,
                    shape: [kDimKnown],
                },
            ],
        );

        const rawValsOut = rowLoop.loopOutputs[0];
        const rawIdxsOut = rowLoop.loopOutputs[1];
        const permOutDims = [...permDims.slice(0, rank - 1), kTripOuter];
        const permOutStatic = [...permRawDims.slice(0, rank - 1), kDimKnown] as KnownShape;
        const permOutShapeVec = packShapeVector(outerBuilder, permOutDims, `topkl_psh_${op.id}`);

        let finalVals: ConcreteValueNode = outerBuilder.createOp(
            "Reshape",
            [rawValsOut, permOutShapeVec],
            {},
            [{ type: dtype, shape: permOutStatic }],
        )[0];
        let finalIdxs: ConcreteValueNode = outerBuilder.createOp(
            "Reshape",
            [rawIdxsOut, permOutShapeVec],
            {},
            [{ type: DataType.INT64, shape: permOutStatic }],
        )[0];

        if (needsPermute) {
            const invPerm = new Array<number>(rank);
            perm.forEach((p, i) => {
                invPerm[p] = i;
            });
            const finalShape = [
                ...rawDims.slice(0, axis),
                kDimKnown,
                ...rawDims.slice(axis + 1),
            ] as KnownShape;
            [finalVals] = outerBuilder.createOp("Transpose", [finalVals], { perm: invPerm }, [
                { type: dtype, shape: finalShape },
            ]);
            [finalIdxs] = outerBuilder.createOp("Transpose", [finalIdxs], { perm: invPerm }, [
                { type: DataType.INT64, shape: finalShape },
            ]);
        }

        outerBuilder.replaceAllUsesWith(YVals, finalVals);
        outerBuilder.replaceAllUsesWith(YIdxs, finalIdxs);
        outerBuilder.removeNode(op);
    }
}
