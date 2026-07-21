import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { getIntAttr, makeTensorProto, toStaticShape } from "../../../Utils.js";

export class LowerQuantizeLinearRecipe implements DecompositionRecipe {
    public readonly name = "LowerQuantizeLinear";
    public readonly targetOp = "QuantizeLinear";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = [
        "Cast",
        "Shape",
        "Unsqueeze",
        "Expand",
        "Div",
        "Round",
        "Add",
        "Clip",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "QuantizeLinear") return false;
        return true;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const S = ins[1];
        const Z = ins.length > 2 ? ins[2] : undefined;
        const Y = op.getOutputs()[0];

        const floatT = (X.literalType as DataType | undefined) ?? DataType.FLOAT;
        const targetType = Z ? Z.literalType : Y.literalType;

        const rank = X.shape.length;
        let axisAttr = getIntAttr(op, "axis", 1);
        if (axisAttr < 0 && rank > 0) axisAttr += rank;

        const expectedZf = [{ type: floatT, shape: Z ? (Z.shape as KnownShape) : [] }];
        const Zf = Z
            ? builder.createOp("Cast", [Z], { to: floatT }, expectedZf)[0]
            : builder.createConstant(`QL_Zzero_${op.id}`, makeTensorProto(floatT, [], [0]));

        const xShapeStatic = toStaticShape(X.shape);
        let shapeX: ConcreteValueNode;
        if (xShapeStatic.length > 0 && xShapeStatic.every((d) => d > 0)) {
            shapeX = builder.createConstant(
                `QL_shapeX_${op.id}`,
                makeTensorProto(DataType.INT64, [xShapeStatic.length], xShapeStatic),
            );
        } else {
            shapeX = builder.createOp("Shape", [X], {}, [
                { type: DataType.INT64, shape: [rank] },
            ])[0];
        }

        let Sx = S;
        let Zx = Zf;
        const sRank = S.shape.length;
        const isPerAxis = sRank === 1 && rank > 1 && !(S.shape[0] === 1);
        const expectedBroadcast = [{ type: floatT, shape: X.shape as KnownShape }];

        if (isPerAxis) {
            const axesVals = Array.from({ length: rank }, (_, i) => i).filter(
                (i) => i !== axisAttr,
            );
            const axesConst = builder.createConstant(
                `QL_axes_${op.id}`,
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
            const sRanked = builder.createOp("Unsqueeze", [S, axesConst], {}, expectedSUnsq)[0];
            const zRanked = builder.createOp("Unsqueeze", [Zf, axesConst], {}, expectedSUnsq)[0];

            Sx = builder.createOp("Expand", [sRanked, shapeX], {}, expectedBroadcast)[0];
            Zx = builder.createOp("Expand", [zRanked, shapeX], {}, expectedBroadcast)[0];
        } else {
            Sx = builder.createOp("Expand", [S, shapeX], {}, expectedBroadcast)[0];
            Zx = builder.createOp("Expand", [Zf, shapeX], {}, expectedBroadcast)[0];
        }

        const divOp = builder.createOp("Div", [X, Sx], {}, expectedBroadcast)[0];
        const roundOp = builder.createOp("Round", [divOp], {}, expectedBroadcast)[0];
        const addOp = builder.createOp("Add", [roundOp, Zx], {}, expectedBroadcast)[0];

        const minVal = targetType === DataType.INT8 ? -128 : 0;
        const maxVal = targetType === DataType.INT8 ? 127 : 255;
        const minConst = builder.createConstant(
            `QL_min_${op.id}`,
            makeTensorProto(floatT, [], [minVal]),
        );
        const maxConst = builder.createConstant(
            `QL_max_${op.id}`,
            makeTensorProto(floatT, [], [maxVal]),
        );

        const clipOp = builder.createOp(
            "Clip",
            [addOp, minConst, maxConst],
            {},
            expectedBroadcast,
        )[0];

        const expectedY = [{ type: targetType, shape: X.shape as KnownShape }];
        const finalY = builder.createOp("Cast", [clipOp], { to: targetType }, expectedY)[0];

        builder.replaceAllUsesWith(Y, finalY);
        builder.removeNode(op);
    }
}
