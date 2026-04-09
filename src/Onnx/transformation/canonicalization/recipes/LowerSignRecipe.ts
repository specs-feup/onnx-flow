import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";

export class LowerSignRecipe implements DecompositionRecipe {
    public readonly name = "LowerSign";
    public readonly targetOp = "Sign";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Greater", "Where", "Less"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Sign";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const A = ins[0];
        const Y = op.getOutputs()[0];

        //Expected output type is same as input type and shape is same as input shape
        const OutType = A.literalType as DataType | undefined ?? DataType.FLOAT;
        const OutShape = (Y.shape as KnownShape) ?? (A.shape as KnownShape);
        const Output = [{ type: OutType, shape: OutShape }];
        const BoolOutput = [{ type: DataType.BOOL, shape: OutShape }];

        //Create a scalar '0' constant of the same type
        const zeroConst = builder.createConstant(
            `sign_zero_${op.id}`,
            makeTensorProto(OutType, [], [0]),
        );

        const oneConst = builder.createConstant(
            `sign_one_${op.id}`,
            makeTensorProto(OutType, [], [1]),
        );

        const minusOneConst = builder.createConstant(
            `sign_minus_one_${op.id}`,
            makeTensorProto(OutType, [], [-1]),
        );


        //Greater (A,0)
        const GreaterOut = builder.createOp("Greater", [A, zeroConst], {}, BoolOutput)[0];

        //Less (A,0)
        const LessOut = builder.createOp("Less", [A, zeroConst], {}, BoolOutput)[0];

        //Where(LessOut, -1, 0)
        const WhereLessOut = builder.createOp("Where", [LessOut, minusOneConst, zeroConst], {}, Output)[0];

        //Where(GreaterOut, 1, Where(LessOut, -1, 0), -1, 0)
        const WhereOut = builder.createOp("Where", [GreaterOut, oneConst, WhereLessOut], {}, Output)[0];

        builder.replaceAllUsesWith(Y, WhereOut);
        op.remove();
    }
}