import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";

export class LowerRoundAddRecipe implements DecompositionRecipe {
    public readonly name = "LowerRoundAdd";
    public readonly targetOp = "RoundAdd";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false
    public readonly producedOps = ["Floor", "Sub", "Equal", "Greater", "Mod", "Where", "Add"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Round";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const Y = op.getOutputs()[0];

        const dtype =
            (Y.literalType as DataType | undefined) ??
            (X.literalType as DataType | undefined) ??
            DataType.FLOAT;

        const outShape =
            (Y.shape as KnownShape) ??
            (X.shape as KnownShape);

        const output = [{ type: dtype, shape: outShape }];
        const boolOutput = [{ type: DataType.BOOL, shape: outShape }];

        const halfConst = builder.createConstant(
            `round_half_${op.id}`,
            makeTensorProto(dtype, [], [0.5]),
        );

        const oneConst = builder.createConstant(
            `round_one_${op.id}`,
            makeTensorProto(dtype, [], [1]),
        );

        const zeroConst = builder.createConstant(
            `round_zero_${op.id}`,
            makeTensorProto(dtype, [], [0]),
        );

        const twoConst = builder.createConstant(
            `round_two_${op.id}`,
            makeTensorProto(dtype, [], [2]),
        );

        // Sub(X, Floor(X))
        const floorOut = builder.createOp("Floor", [X], {}, output)[0];
        const subOut = builder.createOp("Sub", [X, floorOut], {}, output)[0];

        // Equal (subOut,0.5)
        const isHalf = builder.createOp("Equal", [subOut, halfConst], {}, boolOutput)[0];

        // Equal(Mod (Floor(X), 2), 1)
        const modOut = builder.createOp("Mod", [floorOut, twoConst], {}, output)[0];
        const isOdd = builder.createOp("Equal", [modOut, oneConst], {}, boolOutput)[0];

        // Where (isOdd, 1, 0)
        const tieAdd = builder.createOp("Where", [isOdd, oneConst, zeroConst], {}, output)[0];
    
        // (Greater (subOut, 0.5))
        const gtHalf = builder.createOp("Greater", [subOut, halfConst], {}, boolOutput)[0];

        // Where (gtHalf, 1, 0)
        const gtHalfout = builder.createOp("Where", [gtHalf, oneConst, zeroConst], {}, output)[0];

        // Add (floorOut, Where(isHalf, tieAdd, gtHalfout))
        const addTerm = builder.createOp("Where", [isHalf, tieAdd, gtHalfout], {}, output)[0];
        const roundOut = builder.createOp("Add", [floorOut, addTerm], {}, output)[0];

        builder.replaceAllUsesWith(Y, roundOut);
        op.remove();
    }
}