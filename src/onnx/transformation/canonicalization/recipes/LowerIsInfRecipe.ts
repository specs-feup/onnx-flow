import type OperationNode from "../../../ir/OperationNode.js";
import type { GraphBuilder } from "../../../ir/GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../ir/OnnxTypes.js";
import { DataType } from "../../../ir/OnnxTypes.js";
import { makeTensorProto } from "../../../utils/index.js";

export class LowerIsInfRecipe implements DecompositionRecipe {
    public readonly name = "LowerIsInf";
    public readonly targetOp = "IsInf";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly producedOps = ["Equal", "Abs"];

    match(op: OperationNode.Class): boolean {
        return op.type === "IsInf";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const A = ins[0];
        const Y = op.getOutputs()[0];

        const dtype = (A.literalType as DataType | undefined) ?? DataType.FLOAT;

        const Yshape: KnownShape | undefined = Y.shape as KnownShape | undefined;
        const outShape = (Yshape ?? A.shape) as KnownShape;

        const output = [{ type: dtype, shape: outShape }];
        const boolOutput = [{ type: DataType.BOOL, shape: outShape }];

        const INF = builder.createConstant(
            `isinf_inf_${op.id}_inf`,
            makeTensorProto(DataType.FLOAT, [], [Infinity]),
        );

        //Equal(Abs(A), Inf)
        const AbsOut = builder.createOp("Abs", [A], {}, output)[0];
        const EqualOut = builder.createOp("Equal", [AbsOut, INF], {}, boolOutput)[0];

        builder.replaceAllUsesWith(Y, EqualOut);
        op.remove();
    }
}
