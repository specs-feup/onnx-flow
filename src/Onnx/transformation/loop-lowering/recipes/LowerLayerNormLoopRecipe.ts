import type OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import { type ConcreteValueNode, type KnownShape, DataType } from "../../../OnnxTypes.js";
import {
    getFloatAttr,
    getIntAttr,
    int64Vec,
    makeTensorProto,
    normalizeAxis,
    scalarInt64,
    toStaticShape,
} from "../../../Utils.js";
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

export class LowerLayerNormLoopRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerLayerNormLoop";
    public readonly targetOp = "LayerNormalization";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["normalization", "welford-shifted", "nested-loop"] as const;
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
        "ReduceProd",
        "Identity",
        "Cast",
        "Max",
        "Sub",
        "Add",
        "Mul",
        "Sqrt",
        "Div",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "LayerNormalization") return false;
        const usedExtraOuts = op
            .getOutputs()
            .slice(1)
            .filter(
                (t) =>
                    t.is(TensorNode) &&
                    (t.as(TensorNode).type === "output" || t.as(TensorNode).outgoers.length > 0),
            );
        if (usedExtraOuts.length > 0) return false; // optional Mean / InvStdDev not emitted here
        const ins = op.getInputs();
        return !!ins && ins.length >= 2 && Array.isArray(ins[0].shape) && ins[0].shape.length >= 1;
    }

    apply(): never {
        throw new Error("LowerLayerNormLoop uses lowerStandalone");
    }

    lowerStandalone(
        op: OperationNode.Class,
        outerBuilder: GraphBuilder,
        indexingMode: IndexingMode,
    ): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const Scale = ins[1];
        const B = ins.length > 2 ? ins[2] : undefined;
        const Y = op.getOutputs()[0];

        const dtype = (X.literalType || Y.literalType || DataType.FLOAT) as DataType;
        const rawDims = toStaticShape(X.shape);
        const rank = rawDims.length;
        const axis = normalizeAxis(getIntAttr(op, "axis", -1), rank);
        const epsilon = getFloatAttr(op, "epsilon", 1e-5);

        const xMixed = extractMixedDims(outerBuilder, X, rawDims, `ln_d_${op.id}`);
        let numRowsMixed: number | ConcreteValueNode = 1;
        for (let i = 0; i < axis; i++) {
            numRowsMixed = mulMixed(outerBuilder, numRowsMixed, xMixed[i], `ln_nr_${i}_${op.id}`);
        }
        let dMixed: number | ConcreteValueNode = 1;
        for (let i = axis; i < rank; i++) {
            dMixed = mulMixed(outerBuilder, dMixed, xMixed[i], `ln_nd_${i}_${op.id}`);
        }

        const shape2d = packShapeVector(outerBuilder, [numRowsMixed, dMixed], `ln_2d_${op.id}`);
        const dStaticOrDyn = typeof dMixed === "number" ? dMixed : -1;
        const [x2d] = outerBuilder.createOp("Reshape", [X, shape2d], {}, [
            {
                type: dtype,
                shape: [typeof numRowsMixed === "number" ? numRowsMixed : -1, dStaticOrDyn],
            },
        ]);
        const flatShapeConst = outerBuilder.createConstant(`ln_flat_${op.id}`, int64Vec([-1]));
        const [scaleFlat] = outerBuilder.createOp("Reshape", [Scale, flatShapeConst], {}, [
            { type: dtype, shape: [dStaticOrDyn] },
        ]);
        const biasFlat = B
            ? outerBuilder.createOp("Reshape", [B, flatShapeConst], {}, [
                  { type: dtype, shape: [dStaticOrDyn] },
              ])[0]
            : undefined;

        const origShapeVec = packShapeVector(outerBuilder, xMixed, `ln_osh_${op.id}`);
        const dTripOuter: ConcreteValueNode =
            typeof dMixed === "number"
                ? outerBuilder.createConstant(`ln_dtrip_${op.id}`, scalarInt64(dMixed))
                : dMixed;

        const zeroScalarOuter = outerBuilder.createConstant(
            `ln_z_${op.id}`,
            makeTensorProto(dtype, [], [0]),
        );
        const oneScalarOuter = outerBuilder.createConstant(
            `ln_one_${op.id}`,
            makeTensorProto(dtype, [], [1]),
        );
        const epsOuter = outerBuilder.createConstant(
            `ln_eps_${op.id}`,
            makeTensorProto(dtype, [], [epsilon]),
        );
        const [dFloatOuter] = outerBuilder.createOp("Cast", [dTripOuter], { to: dtype }, [
            { type: dtype, shape: [] },
        ]);
        const [invDOuter] = outerBuilder.createOp("Div", [oneScalarOuter, dFloatOuter], {}, [
            { type: dtype, shape: [] },
        ]);

        // Outer loop over normalized slices [0..numRows-1]
        const outerLoop = buildForLoopRegion(
            outerBuilder,
            numRowsMixed,
            dtype,
            [1],
            `ln_rows_${op.id}`,
            `layernorm_${op.id}`,
            indexingMode,
            false,
        );
        const rowBuilder = outerLoop.innerBuilder;
        const innerX2d = ensureInnerValueNode(rowBuilder, x2d);
        const [rowVec] = rowBuilder.createOp("Gather", [innerX2d, outerLoop.trip], { axis: 0 }, [
            { type: dtype, shape: [dStaticOrDyn] },
        ]);

        const zeroI64Row = rowBuilder.createConstant(`ln_zi64_${op.id}`, scalarInt64(0));
        const x0Row = gatherScalar(rowBuilder, rowVec, zeroI64Row);
        const dTripRow = ensureInnerValueNode(rowBuilder, dTripOuter);
        const zeroRow = ensureInnerValueNode(rowBuilder, zeroScalarOuter);
        const oneRow = ensureInnerValueNode(rowBuilder, oneScalarOuter);
        const invDRow = ensureInnerValueNode(rowBuilder, invDOuter);
        const epsRow = ensureInnerValueNode(rowBuilder, epsOuter);

        // Pass 1: Shifted single-pass (sum(x - x0), sum((x - x0)^2)) in 0D scalar carries
        const pass1 = buildForLoopRegion(
            rowBuilder,
            dTripRow,
            dtype,
            [1],
            `ln_p1_${op.id}`,
            `ln_p1_${op.id}`,
            indexingMode,
            false,
        );
        const p1Builder = pass1.innerBuilder;
        const s1In = pass1.loopCtx.addCarry(`ln_s1_${op.id}`, zeroRow, dtype, []);
        const s2In = pass1.loopCtx.addCarry(`ln_s2_${op.id}`, zeroRow, dtype, []);
        const p1RowVec = ensureInnerValueNode(p1Builder, rowVec);
        const p1X0 = ensureInnerValueNode(p1Builder, x0Row);
        const x1 = gatherScalar(p1Builder, p1RowVec, pass1.trip);
        const [dx] = p1Builder.createOp("Sub", [x1, p1X0]);
        const [dxSq] = p1Builder.createOp("Mul", [dx, dx]);
        const [s1New] = p1Builder.createOp("Add", [s1In, dx]);
        const [s2New] = p1Builder.createOp("Add", [s2In, dxSq]);
        pass1.loopCtx.setNextState(s1In, s1New);
        pass1.loopCtx.setNextState(s2In, s2New);
        pass1.finalize([]);

        const s1Final = pass1.loopOutputs[0];
        const s2Final = pass1.loopOutputs[1];
        const [meanOff] = rowBuilder.createOp("Mul", [s1Final, invDRow]);
        const [meanRow] = rowBuilder.createOp("Add", [x0Row, meanOff]);
        const [meanOffSq] = rowBuilder.createOp("Mul", [meanOff, meanOff]);
        const [rawEx2] = rowBuilder.createOp("Mul", [s2Final, invDRow]);
        const [rawVar] = rowBuilder.createOp("Sub", [rawEx2, meanOffSq]);
        const [varPos] = rowBuilder.createOp("Max", [rawVar, zeroRow]);
        const [varEps] = rowBuilder.createOp("Add", [varPos, epsRow]);
        const [stdRow] = rowBuilder.createOp("Sqrt", [varEps]);
        const [invStdRow] = rowBuilder.createOp("Div", [oneRow, stdRow], {}, [
            { type: dtype, shape: [] },
        ]);

        // Pass 2: Normalize & affine transform via scan_output -> [D]
        const pass2 = buildForLoopRegion(
            rowBuilder,
            dTripRow,
            dtype,
            [1],
            `ln_p2_${op.id}`,
            `ln_p2_${op.id}`,
            indexingMode,
            false,
        );
        const p2Builder = pass2.innerBuilder;
        const p2RowVec = ensureInnerValueNode(p2Builder, rowVec);
        const p2Scale = ensureInnerValueNode(p2Builder, scaleFlat);
        const p2Mean = ensureInnerValueNode(p2Builder, meanRow);
        const p2InvStd = ensureInnerValueNode(p2Builder, invStdRow);

        const x2 = gatherScalar(p2Builder, p2RowVec, pass2.trip);
        const s2 = gatherScalar(p2Builder, p2Scale, pass2.trip);
        const [centered] = p2Builder.createOp("Sub", [x2, p2Mean]);
        const [normed] = p2Builder.createOp("Mul", [centered, p2InvStd]);
        let [outScalar] = p2Builder.createOp("Mul", [normed, s2]);
        if (biasFlat) {
            const p2Bias = ensureInnerValueNode(p2Builder, biasFlat);
            const b2 = gatherScalar(p2Builder, p2Bias, pass2.trip);
            [outScalar] = p2Builder.createOp("Add", [outScalar, b2]);
        }
        pass2.finalize([], [{ tag: "row", node: outScalar, elemTy: dtype, shape: [] }]);

        const normRow = pass2.loopOutputs[0];
        outerLoop.finalize(
            [],
            [{ tag: "rows", node: normRow, elemTy: dtype, shape: [dStaticOrDyn] }],
        );

        const rawLoopOut = outerLoop.loopOutputs[0];
        const [finalOut] = outerBuilder.createOp("Reshape", [rawLoopOut, origShapeVec], {}, [
            { type: Y.literalType || dtype, shape: rawDims as KnownShape },
        ]);

        outerBuilder.replaceAllUsesWith(Y, finalOut);
        outerBuilder.removeNode(op);
    }
}
