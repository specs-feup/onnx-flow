import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { int64Vec } from "../../../utils/index.js";

export class LowerMatMulIntegerRecipe implements DecompositionRecipe {
    public readonly name = "LowerMatMulInteger";
    public readonly targetOp = "MatMulInteger";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly producedOps = ["Cast", "Unsqueeze", "Sub", "MatMul"];

    match(op: OperationNode.Class): boolean {
        return op.type === "MatMulInteger" && (op.getInputs()?.length ?? 0) >= 2;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const A = ins[0];
        const B = ins[1];
        const aZeroPoint = ins.length >= 3 ? ins[2] : undefined;
        const bZeroPoint = ins.length >= 4 ? ins[3] : undefined;
        const Y = op.getOutputs()[0];

        let [aEff] = builder.createOp("Cast", [A], { to: DataType.INT32 }, [
            { type: DataType.INT32, shape: A.shape as KnownShape },
        ]);
        if (aZeroPoint) {
            let [azp] = builder.createOp("Cast", [aZeroPoint], { to: DataType.INT32 }, [
                { type: DataType.INT32, shape: aZeroPoint.shape as KnownShape },
            ]);
            // Per ONNX spec, 1D a_zero_point [M] broadcasts along A's rows [..., M, K] -> [M, 1]
            if (aZeroPoint.shape.length === 1 && A.shape.length >= 2) {
                const lastAxis = builder.createConstant(`mmi_azp_ax_${op.id}`, int64Vec([1]));
                [azp] = builder.createOp("Unsqueeze", [azp, lastAxis], {}, [
                    { type: DataType.INT32, shape: [aZeroPoint.shape[0], 1] as KnownShape },
                ]);
            }
            [aEff] = builder.createOp("Sub", [aEff, azp], {}, [
                { type: DataType.INT32, shape: A.shape as KnownShape },
            ]);
        }

        let [bEff] = builder.createOp("Cast", [B], { to: DataType.INT32 }, [
            { type: DataType.INT32, shape: B.shape as KnownShape },
        ]);
        if (bZeroPoint) {
            const [bzp] = builder.createOp("Cast", [bZeroPoint], { to: DataType.INT32 }, [
                { type: DataType.INT32, shape: bZeroPoint.shape as KnownShape },
            ]);
            [bEff] = builder.createOp("Sub", [bEff, bzp], {}, [
                { type: DataType.INT32, shape: B.shape as KnownShape },
            ]);
        }

        const [out] = builder.createOp("MatMul", [aEff, bEff], {}, [
            { type: DataType.INT32, shape: Y.shape as KnownShape },
        ]);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
