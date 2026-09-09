import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import {
    getIntAttr,
    makeTensorProto,
    toStaticShape,
    tryAsConcreteValueNode,
} from "../../../Utils.js";

export class LowerDequantizeLinearRecipe implements DecompositionRecipe {
    public readonly name = "LowerDequantizeLinear";
    public readonly targetOp = "DequantizeLinear";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Cast", "Shape", "Unsqueeze", "Expand", "Sub", "Mul"];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "DequantizeLinear") return false;
        return true;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const S = ins[1];
        const Z = ins.length > 2 ? ins[2] : undefined;
        const Y = op.getOutputs()[0];

        const floatT = [
            DataType.FLOAT,
            DataType.FLOAT16,
            DataType.BFLOAT16,
            DataType.DOUBLE,
        ].includes(Y.literalType as DataType)
            ? (Y.literalType as DataType)
            : DataType.FLOAT;

        // Force output to float
        tryAsConcreteValueNode(Y)!.setLiteralType(floatT);

        // ONNX Default for DequantizeLinear axis is 1
        let axisAttr = getIntAttr(op, "axis", 1);
        const rank = X.shape.length;
        if (axisAttr < 0 && rank > 0) axisAttr += rank;

        // Strictly define expected outputs to bypass inference errors
        const expectedXf = [{ type: floatT, shape: X.shape as KnownShape }];
        const expectedSf = [{ type: floatT, shape: S.shape as KnownShape }];
        const expectedZf = [{ type: floatT, shape: Z ? (Z.shape as KnownShape) : [] }];
        const expectedBroadcast = [{ type: floatT, shape: X.shape as KnownShape }];

        const Xf = builder.createOp("Cast", [X], { to: floatT }, expectedXf)[0];
        const Sf = builder.createOp("Cast", [S], { to: floatT }, expectedSf)[0];
        const Zf = Z
            ? builder.createOp("Cast", [Z], { to: floatT }, expectedZf)[0]
            : builder.createConstant(`DQL_Zzero_${op.id}`, makeTensorProto(floatT, [], [0]));

        const xShapeStatic = toStaticShape(X.shape);
        let shapeX: ConcreteValueNode;
        if (xShapeStatic.length > 0 && xShapeStatic.every((d) => d > 0)) {
            shapeX = builder.createConstant(
                `DQL_shapeX_${op.id}`,
                makeTensorProto(DataType.INT64, [xShapeStatic.length], xShapeStatic),
            );
        } else {
            shapeX = builder.createOp("Shape", [Xf], {}, [
                { type: DataType.INT64, shape: [rank] },
            ])[0];
        }

        let Sx = Sf;
        let Zx = Zf;
        const sRank = S.shape.length;
        const isPerAxis = sRank === 1 && rank > 1 && !(S.shape[0] === 1);

        if (isPerAxis) {
            const axesVals = Array.from({ length: rank }, (_, i) => i).filter(
                (i) => i !== axisAttr,
            );
            const axesConst = builder.createConstant(
                `DQL_axes_${op.id}`,
                makeTensorProto(DataType.INT64, [axesVals.length], axesVals),
            );

            const expectedSUnsq = [
                {
                    type: floatT,
                    shape: Array(rank)
                        .fill(1)
                        .map((_, i) => (i === axisAttr ? (S.shape[0] ?? 1) : 1)),
                },
            ];
            const sRanked = builder.createOp("Unsqueeze", [Sf, axesConst], {}, expectedSUnsq)[0];

            // Only Unsqueeze Zf if Z was provided. Otherwise, it is a scalar and shouldn't be unsqueezed.
            const zRanked = Z
                ? builder.createOp("Unsqueeze", [Zf, axesConst], {}, expectedSUnsq)[0]
                : Zf;

            Sx = builder.createOp("Expand", [sRanked, shapeX], {}, expectedBroadcast)[0];
            Zx = builder.createOp("Expand", [zRanked, shapeX], {}, expectedBroadcast)[0];
        } else {
            Sx = builder.createOp("Expand", [Sf, shapeX], {}, expectedBroadcast)[0];
            Zx = builder.createOp("Expand", [Zf, shapeX], {}, expectedBroadcast)[0];
        }

        const sub = builder.createOp("Sub", [Xf, Zx], {}, expectedBroadcast)[0];
        const mul = builder.createOp("Mul", [sub, Sx], {}, expectedBroadcast)[0];

        builder.replaceAllUsesWith(Y, mul);
        builder.removeNode(op);
    }
}
