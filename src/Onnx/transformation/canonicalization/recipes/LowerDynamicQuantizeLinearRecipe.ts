import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";

export class LowerDynamicQuantizeLinearRecipe implements DecompositionRecipe {
    public readonly name = "LowerDynamicQuantizeLinear";
    public readonly targetOp = "DynamicQuantizeLinear";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly producedOps = [
        "ReduceMin",
        "ReduceMax",
        "Min",
        "Max",
        "Sub",
        "Mul",
        "Div",
        "Equal",
        "Where",
        "Round",
        "Clip",
        "Cast",
        "QuantizeLinear",
        "Identity",
    ];

    match(op: OperationNode.Class): boolean {
        return op.type === "DynamicQuantizeLinear";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = op.getInputs()![0] as ConcreteValueNode;
        const outputs = op.getOutputs();
        const Y = outputs[0];
        const YScale = outputs[1];
        const YZeroPoint = outputs[2];

        const dtype = X.literalType as DataType;
        const outShape = Y.shape as KnownShape;
        const scalarFloat = [{ type: dtype, shape: [] as KnownShape }];
        const scalarBool = [{ type: DataType.BOOL, shape: [] as KnownShape }];
        const scalarU8 = [{ type: DataType.UINT8, shape: [] as KnownShape }];
        const outU8 = [{ type: DataType.UINT8, shape: outShape }];

        const c = (tag: string, val: number) =>
            builder.createConstant(`dql_${tag}_${op.id}`, makeTensorProto(dtype, [], [val]));

        const zero = c("zero", 0.0);
        const one = c("one", 1.0);
        const maxQ = c("max_q", 255.0);
        const invMaxQ = c("inv_max_q", 1.0 / 255.0);

        // 1. min = Min(0, ReduceMin(X)), max = Max(0, ReduceMax(X))
        const [rawMin] = builder.createOp("ReduceMin", [X], { keepdims: 0 }, scalarFloat);
        const [rawMax] = builder.createOp("ReduceMax", [X], { keepdims: 0 }, scalarFloat);
        const [xMin] = builder.createOp("Min", [zero, rawMin], {}, scalarFloat);
        const [xMax] = builder.createOp("Max", [zero, rawMax], {}, scalarFloat);

        // 2. y_scale = (x_max - x_min) / 255
        const [range] = builder.createOp("Sub", [xMax, xMin], {}, scalarFloat);
        const [scale] = builder.createOp("Mul", [range, invMaxQ], {}, scalarFloat);

        // 3. y_zero_point = uint8(Clip(Round(-x_min / safe_scale), 0, 255))
        const [negMin] = builder.createOp("Sub", [zero, xMin], {}, scalarFloat);
        const [isZeroScale] = builder.createOp("Equal", [scale, zero], {}, scalarBool);
        const [safeScale] = builder.createOp("Where", [isZeroScale, one, scale], {}, scalarFloat);
        const [rawZp] = builder.createOp("Div", [negMin, safeScale], {}, scalarFloat);
        const [guardedZp] = builder.createOp("Where", [isZeroScale, zero, rawZp], {}, scalarFloat);
        const [roundedZp] = builder.createOp("Round", [guardedZp], {}, scalarFloat);
        const [clippedZp] = builder.createOp("Clip", [roundedZp, zero, maxQ], {}, scalarFloat);
        const [zp] = builder.createOp("Cast", [clippedZp], { to: DataType.UINT8 }, scalarU8);

        // 4. y = QuantizeLinear(X, scale, zp)
        const [quantY] = builder.createOp("QuantizeLinear", [X, scale, zp], {}, outU8);

        builder.replaceAllUsesWith(Y, quantY);
        builder.replaceAllUsesWith(YScale, scale);
        builder.replaceAllUsesWith(YZeroPoint, zp);
        builder.removeNode(op);
    }
}
