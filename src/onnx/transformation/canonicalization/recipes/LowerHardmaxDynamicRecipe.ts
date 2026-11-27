import type OperationNode from "../../../ir/OperationNode.js";
import type { GraphBuilder } from "../../../ir/GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import { DataType, type ConcreteValueNode, type KnownShape } from "../../../ir/OnnxTypes.js";
import {
    getIntAttr,
    int64Vec,
    makeTensorProto,
    normalizeAxis,
    scalarInt64,
} from "../../../utils/index.js";

export class LowerHardmaxDynamicRecipe implements DecompositionRecipe {
    public readonly name = "LowerHardmaxDynamic";
    public readonly targetOp = "Hardmax";
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 10;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["argmax-based", "dynamic-axis"] as const;
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = [
        "Shape",
        "Gather",
        "ArgMax",
        "Range",
        "Unsqueeze",
        "Equal",
        "Where",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Hardmax") return false;
        const X = op.getInputs()?.[0];
        return X !== undefined && Array.isArray(X.shape) && X.shape.length > 0;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = op.getInputs()![0] as ConcreteValueNode;
        const Y = op.getOutputs()[0];
        const dtype = (X.literalType || DataType.FLOAT) as DataType;
        const outShape = Y.shape as KnownShape;
        const rank = X.shape.length;
        const axis = normalizeAxis(getIntAttr(op, "axis", -1), rank);

        const reducedShape = [...(X.shape as KnownShape)];
        reducedShape[axis] = 1;

        const [shapeX] = builder.createOp("Shape", [X], {}, [
            { type: DataType.INT64, shape: [rank] },
        ]);
        const axSc = builder.createConstant(`hmaxd_ax_${op.id}`, scalarInt64(axis));
        const [dimAxis] = builder.createOp("Gather", [shapeX, axSc], { axis: 0 }, [
            { type: DataType.INT64, shape: [] },
        ]);

        const [argMax] = builder.createOp(
            "ArgMax",
            [X],
            { axis, keepdims: 1, select_last_index: 0 },
            [{ type: DataType.INT64, shape: reducedShape }],
        );

        const startC = builder.createConstant(`hmaxd_s_${op.id}`, scalarInt64(0));
        const deltaC = builder.createConstant(`hmaxd_d_${op.id}`, scalarInt64(1));
        const [rangeVec] = builder.createOp("Range", [startC, dimAxis, deltaC], {}, [
            { type: DataType.INT64, shape: [X.shape[axis]] as KnownShape },
        ]);

        let rangeBcast: ConcreteValueNode = rangeVec;
        if (rank > 1) {
            const unsqAxes: number[] = [];
            for (let d = 0; d < rank; d++) {
                if (d !== axis) unsqAxes.push(d);
            }
            const unsqConst = builder.createConstant(`hmaxd_ua_${op.id}`, int64Vec(unsqAxes));
            [rangeBcast] = builder.createOp("Unsqueeze", [rangeVec, unsqConst]);
        }

        const [mask] = builder.createOp("Equal", [rangeBcast, argMax], {}, [
            { type: DataType.BOOL, shape: outShape },
        ]);
        const oneC = builder.createConstant(`hmaxd_1_${op.id}`, makeTensorProto(dtype, [], [1]));
        const zeroC = builder.createConstant(`hmaxd_0_${op.id}`, makeTensorProto(dtype, [], [0]));
        const [out] = builder.createOp("Where", [mask, oneC, zeroC], {}, [
            { type: dtype, shape: outShape },
        ]);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
