import type OperationNode from "../../../ir/OperationNode.js";
import TensorNode from "../../../ir/TensorNode.js";
import { type ConcreteValueNode, type KnownShape, DataType } from "../../../ir/OnnxTypes.js";
import {
    getIntAttr,
    makeTensorProto,
    normalizeAxis,
    scalarInt64,
    toStaticShape,
} from "../../../utils/index.js";
import type { LoopLoweringRecipe } from "../LoopLoweringRecipe.js";
import { buildForLoopRegion } from "../LoopBuilder.js";
import {
    ensureInnerValueNode,
    gatherScalar,
    mulMixed,
    packShapeVector,
    extractMixedDims,
} from "../RecipeUtils.js";
import type { GraphBuilder } from "../../../ir/GraphBuilder.js";
import type { IndexingMode } from "../../../../DecompositionOptions.js";

export class LowerSoftmaxLoopRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerSoftmaxLoop";
    public readonly targetOp = "Softmax";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["softmax", "online-fused", "nested-loop"] as const;
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
        "Identity",
        "Max",
        "Sub",
        "Exp",
        "Mul",
        "Add",
        "Div",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Softmax") return false;
        const X = op.getInputs()?.[0];
        return !!X && Array.isArray(X.shape) && X.shape.length >= 1;
    }

    apply(): never {
        throw new Error("LowerSoftmaxLoop uses lowerStandalone");
    }

    lowerStandalone(
        op: OperationNode.Class,
        outerBuilder: GraphBuilder,
        indexingMode: IndexingMode,
    ): void {
        lowerOnlineSoftmaxFamily(op, outerBuilder, indexingMode, false);
    }
}

export function lowerOnlineSoftmaxFamily(
    op: OperationNode.Class,
    outerBuilder: GraphBuilder,
    indexingMode: IndexingMode,
    isLogSoftmax: boolean,
): void {
    const X = op.getInputs()![0] as ConcreteValueNode;
    const Y = op.getOutputs()[0];
    const dtype = (X.literalType || Y.literalType || DataType.FLOAT) as DataType;
    const rawDims = toStaticShape(X.shape);
    const rank = rawDims.length;
    const axis = normalizeAxis(getIntAttr(op, "axis", -1), rank);

    // Permute if axis is not trailing so rows are contiguous slices [numRows, D]
    const needsPermute = axis !== rank - 1;
    const perm = Array.from({ length: rank }, (_, i) => i).filter((i) => i !== axis);
    perm.push(axis);

    const workX: ConcreteValueNode = needsPermute
        ? outerBuilder.createOp("Transpose", [X], { perm })[0]
        : X;

    const permRawDims = perm.map((p) => rawDims[p]);
    const permDims = extractMixedDims(outerBuilder, workX, permRawDims, `sm_d_${op.id}`);
    const dMixed = permDims[rank - 1];
    let numRowsMixed: number | ConcreteValueNode = 1;
    for (let i = 0; i < rank - 1; i++) {
        numRowsMixed = mulMixed(outerBuilder, numRowsMixed, permDims[i], `sm_nr_${i}_${op.id}`);
    }

    const shape2d = packShapeVector(outerBuilder, [numRowsMixed, dMixed], `sm_2d_${op.id}`);
    const [workX2d] = outerBuilder.createOp("Reshape", [workX, shape2d], {}, [
        {
            type: dtype,
            shape: [
                typeof numRowsMixed === "number" ? numRowsMixed : -1,
                typeof dMixed === "number" ? dMixed : -1,
            ],
        },
    ]);

    const origPermShape = packShapeVector(outerBuilder, permDims, `sm_osh_${op.id}`);

    const negInfVal = dtype === DataType.DOUBLE ? -1.7976931348623157e308 : -3.4028234663852886e38;
    const negInfOuter = outerBuilder.createConstant(
        `sm_ninf_${op.id}`,
        makeTensorProto(dtype, [], [negInfVal]),
    );
    const zeroOuter = outerBuilder.createConstant(`sm_z_${op.id}`, makeTensorProto(dtype, [], [0]));
    const oneOuter = isLogSoftmax
        ? undefined
        : outerBuilder.createConstant(`sm_one_${op.id}`, makeTensorProto(dtype, [], [1]));
    const dTripOuter: ConcreteValueNode =
        typeof dMixed === "number"
            ? outerBuilder.createConstant(`sm_dtrip_${op.id}`, scalarInt64(dMixed))
            : dMixed;

    // Outer loop over rows [0..numRows-1] with 0 tensor carries and scan_output [D]
    const outerLoop = buildForLoopRegion(
        outerBuilder,
        numRowsMixed,
        dtype,
        [1],
        `sm_rows_${op.id}`,
        `softmax_${op.id}`,
        indexingMode,
        false,
    );
    const rowBuilder = outerLoop.innerBuilder;
    const innerX2d = ensureInnerValueNode(rowBuilder, workX2d);
    const [rowVec] = rowBuilder.createOp("Gather", [innerX2d, outerLoop.trip], { axis: 0 }, [
        { type: dtype, shape: [typeof dMixed === "number" ? dMixed : -1] },
    ]);

    const dTripRow = ensureInnerValueNode(rowBuilder, dTripOuter);
    const mInitRow = ensureInnerValueNode(rowBuilder, negInfOuter);
    const dInitRow = ensureInnerValueNode(rowBuilder, zeroOuter);
    const oneRow = oneOuter ? ensureInnerValueNode(rowBuilder, oneOuter) : undefined;

    // Pass 1: Fused online loop computing (m_final, d_final)
    const pass1 = buildForLoopRegion(
        rowBuilder,
        dTripRow,
        dtype,
        [1],
        `sm_p1_${op.id}`,
        `sm_p1_${op.id}`,
        indexingMode,
        false,
    );
    const p1Builder = pass1.innerBuilder;
    const mIn = pass1.loopCtx.addCarry(`sm_m_${op.id}`, mInitRow, dtype, []);
    const dIn = pass1.loopCtx.addCarry(`sm_d_${op.id}`, dInitRow, dtype, []);
    const p1RowVec = ensureInnerValueNode(p1Builder, rowVec);
    const xVal1 = gatherScalar(p1Builder, p1RowVec, pass1.trip);
    const [mNew] = p1Builder.createOp("Max", [mIn, xVal1]);
    const [mDiff] = p1Builder.createOp("Sub", [mIn, mNew]);
    const [xDiff] = p1Builder.createOp("Sub", [xVal1, mNew]);
    const [expScale] = p1Builder.createOp("Exp", [mDiff]);
    const [expVal] = p1Builder.createOp("Exp", [xDiff]);
    const [scaledD] = p1Builder.createOp("Mul", [dIn, expScale]);
    const [dNew] = p1Builder.createOp("Add", [scaledD, expVal]);
    pass1.loopCtx.setNextState(mIn, mNew);
    pass1.loopCtx.setNextState(dIn, dNew);
    pass1.finalize([]);

    const mFinal = pass1.loopOutputs[0];
    const dFinal = pass1.loopOutputs[1];
    const rowNormScalar = isLogSoftmax
        ? rowBuilder.createOp("Log", [dFinal], {}, [{ type: dtype, shape: [] }])[0]
        : rowBuilder.createOp("Div", [oneRow!, dFinal], {}, [{ type: dtype, shape: [] }])[0];

    // Pass 2: Normalize row scalars via scan_output -> [D]
    const pass2 = buildForLoopRegion(
        rowBuilder,
        dTripRow,
        dtype,
        [1],
        `sm_p2_${op.id}`,
        `sm_p2_${op.id}`,
        indexingMode,
        false,
    );
    const p2Builder = pass2.innerBuilder;
    const p2RowVec = ensureInnerValueNode(p2Builder, rowVec);
    const p2M = ensureInnerValueNode(p2Builder, mFinal);
    const p2NormScalar = ensureInnerValueNode(p2Builder, rowNormScalar);
    const xVal2 = gatherScalar(p2Builder, p2RowVec, pass2.trip);
    const [shifted] = p2Builder.createOp("Sub", [xVal2, p2M]);
    const [normVal] = isLogSoftmax
        ? p2Builder.createOp("Sub", [shifted, p2NormScalar], {}, [{ type: dtype, shape: [] }])
        : (() => {
              const [exp2] = p2Builder.createOp("Exp", [shifted]);
              return p2Builder.createOp("Mul", [exp2, p2NormScalar], {}, [
                  { type: dtype, shape: [] },
              ]);
          })();
    pass2.finalize([], [{ tag: "row", node: normVal, elemTy: dtype, shape: [] }]);

    const rowOutShape: KnownShape = [typeof dMixed === "number" ? dMixed : -1];
    const normRow = pass2.loopOutputs[0];
    outerLoop.finalize([], [{ tag: "rows", node: normRow, elemTy: dtype, shape: rowOutShape }]);

    const rawLoopOut = outerLoop.loopOutputs[0];
    let finalOut: ConcreteValueNode = outerBuilder.createOp(
        "Reshape",
        [rawLoopOut, origPermShape],
        {},
        [{ type: dtype, shape: permRawDims as KnownShape }],
    )[0];

    if (needsPermute) {
        const invPerm = new Array<number>(rank);
        perm.forEach((p, i) => {
            invPerm[p] = i;
        });
        [finalOut] = outerBuilder.createOp("Transpose", [finalOut], { perm: invPerm }, [
            { type: dtype, shape: rawDims as KnownShape },
        ]);
    }

    if (Y.is(TensorNode) && finalOut.is(TensorNode)) {
        outerBuilder.replaceAllUsesWith(Y, finalOut);
    } else {
        outerBuilder.replaceAllUsesWith(Y, finalOut);
    }
    outerBuilder.removeNode(op);
}
