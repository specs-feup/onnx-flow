import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";

export class LowerFloorRecipe implements DecompositionRecipe {
    public readonly name = "LowerFloor";
    public readonly targetOp = "Floor";
    public readonly precision = "exact" as const;
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Cast", "Less", "Sub", "Where"];

    match(op: OperationNode.Class): boolean {
        return op.type === "Floor";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = op.getInputs()![0] as ConcreteValueNode;
        const Y = op.getOutputs()[0];

        const dtype =
            (X.literalType as DataType | undefined) ??
            (Y.literalType as DataType | undefined) ??
            DataType.FLOAT;
        const outShape = Y.shape as KnownShape;
        const outMeta = [{ type: dtype, shape: outShape }];
        const intMeta = [{ type: DataType.INT64, shape: outShape }];
        const boolMeta = [{ type: DataType.BOOL, shape: outShape }];

        // 1. Truncate toward zero: trunc = float(int64(X))
        const [truncInt] = builder.createOp("Cast", [X], { to: DataType.INT64 }, intMeta);
        const [trunc] = builder.createOp("Cast", [truncInt], { to: dtype }, outMeta);

        // 2. If X < trunc (negative with fractional part), subtract 1
        const one = builder.createConstant(`floor_one_${op.id}`, makeTensorProto(dtype, [], [1]));
        const [isLess] = builder.createOp("Less", [X, trunc], {}, boolMeta);
        const [truncMinusOne] = builder.createOp("Sub", [trunc, one], {}, outMeta);
        const [floorOut] = builder.createOp("Where", [isLess, truncMinusOne, trunc], {}, outMeta);

        builder.replaceAllUsesWith(Y, floorOut);
        builder.removeNode(op);
    }
}
