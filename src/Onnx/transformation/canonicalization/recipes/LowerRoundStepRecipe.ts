import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";

export class LowerRoundStepRecipe implements DecompositionRecipe {
    public readonly name = "LowerRoundStep";
    public readonly targetOp = "Round";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
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

        // floor(X)
        const floorOut = builder.createOp("Floor", [X], {}, output)[0];

        // sub(X, floor(X)
        const subOut = builder.createOp("Sub", [X, floorOut], {}, output)[0];

        // equal(subOut, 0.5)
        const IsHalf = builder.createOp("Equal", [subOut, halfConst], {}, boolOutput)[0];

        // Equal(Mod(floor(X), 2), 0)
        const modOut = builder.createOp("Mod", [floorOut, twoConst], {}, output)[0];
        const IsOdd = builder.createOp("Equal", [modOut, oneConst], {}, boolOutput)[0];

        // Where(IsOdd, 1, 0)
        const whereOut = builder.createOp("Where", [IsOdd, oneConst, zeroConst], {}, output)[0];

        // Greater ( subout, 0.5)
        const greaterOut = builder.createOp("Greater", [subOut, halfConst], {}, boolOutput)[0];

        // Where (greaterOut, 1, 0)
        const whereGreaterOut = builder.createOp("Where", [greaterOut, oneConst, zeroConst], {}, output)[0];

        // Where(IsHalf, whereOut, whereGreaterOut)
        const roundOut = builder.createOp("Where", [IsHalf, whereOut, whereGreaterOut], {}, output)[0];

        // Add (floor(X), roundOut)
        const finalRoundOut = builder.createOp("Add", [floorOut, roundOut], {}, output)[0];

        builder.replaceAllUsesWith(Y, roundOut);
        op.remove();
    }
}