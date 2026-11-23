import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import {
    getIntAttr,
    int64Vec,
    makeTensorProto,
    normalizeAxis,
    scalarInt64,
    toStaticShape,
} from "../../../Utils.js";

export class LowerHardmaxRecipe implements DecompositionRecipe {
    public readonly name = "LowerHardmax";
    public readonly targetOp = "Hardmax";
    public readonly precision = "exact" as const;
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["ArgMax", "Range", "Reshape", "Equal", "Where"];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Hardmax") return false;
        const X = op.getInputs()?.[0];
        if (!X || !Array.isArray(X.shape) || X.shape.length === 0) return false;
        const shape = toStaticShape(X.shape);
        const axis = normalizeAxis(getIntAttr(op, "axis", -1), shape.length);
        return shape[axis] > 0;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = op.getInputs()![0] as ConcreteValueNode;
        const Y = op.getOutputs()[0];

        const dtype = X.literalType as DataType;
        const inShape = toStaticShape(X.shape);
        const rank = inShape.length;
        const axis = normalizeAxis(getIntAttr(op, "axis", -1), rank);
        const axisDim = inShape[axis];

        const argmaxShape = [...inShape];
        argmaxShape[axis] = 1;

        const [argMax] = builder.createOp("ArgMax", [X], { axis, keepdims: 1 }, [
            { type: DataType.INT64, shape: argmaxShape as KnownShape },
        ]);

        const startConst = builder.createConstant(`hmax_start_${op.id}`, scalarInt64(0));
        const limitConst = builder.createConstant(`hmax_limit_${op.id}`, scalarInt64(axisDim));
        const deltaConst = builder.createConstant(`hmax_delta_${op.id}`, scalarInt64(1));

        const [range1D] = builder.createOp("Range", [startConst, limitConst, deltaConst], {}, [
            { type: DataType.INT64, shape: [axisDim] },
        ]);

        const bcastShape = new Array<number>(rank).fill(1);
        bcastShape[axis] = axisDim;
        const bcastShapeConst = builder.createConstant(`hmax_bcast_${op.id}`, int64Vec(bcastShape));

        const [rangeND] = builder.createOp("Reshape", [range1D, bcastShapeConst], {}, [
            { type: DataType.INT64, shape: bcastShape as KnownShape },
        ]);

        const [mask] = builder.createOp("Equal", [rangeND, argMax], {}, [
            { type: DataType.BOOL, shape: inShape as KnownShape },
        ]);
        const oneVal = builder.createConstant(`hmax_one_${op.id}`, makeTensorProto(dtype, [], [1]));
        const zeroVal = builder.createConstant(
            `hmax_zero_${op.id}`,
            makeTensorProto(dtype, [], [0]),
        );
        const [out] = builder.createOp("Where", [mask, oneVal, zeroVal], {}, [
            { type: dtype, shape: inShape as KnownShape },
        ]);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
