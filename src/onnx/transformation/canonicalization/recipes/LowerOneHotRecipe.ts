import type OperationNode from "../../../ir/OperationNode.js";
import type { GraphBuilder } from "../../../ir/GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../ir/OnnxTypes.js";
import { DataType } from "../../../ir/OnnxTypes.js";
import {
    getIntAttr,
    int64Vec,
    normalizeAxis,
    readScalarFromTensorNode,
    scalarInt64,
    toStaticShape,
} from "../../../utils/index.js";

export class LowerOneHotRecipe implements DecompositionRecipe {
    public readonly name = "LowerOneHot";
    public readonly targetOp = "OneHot";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 100;
    public readonly shapeSupport = "dynamic-batch" as const;
    public readonly tags = ["constant-depth"];
    public readonly producedOps = [
        "Cast",
        "Less",
        "Add",
        "Where",
        "Unsqueeze",
        "Range",
        "Reshape",
        "Equal",
        "Gather",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "OneHot") return false;
        const ins = op.getInputs();
        if (!ins || ins.length < 3) return false;
        const depthVal = readScalarFromTensorNode(ins[1]);
        return depthVal !== undefined && Math.floor(depthVal) > 0 && Array.isArray(ins[0].shape);
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const [indices, depthNode, values] = op.getInputs() as ConcreteValueNode[];
        const Y = op.getOutputs()[0];

        const depth = Math.floor(readScalarFromTensorNode(depthNode)!);
        const valDtype = values.literalType as DataType;
        const idxShape = toStaticShape(indices.shape);
        const outRank = idxShape.length + 1;
        const axis = normalizeAxis(getIntAttr(op, "axis", -1), outRank);

        const outShape = [...idxShape];
        outShape.splice(axis, 0, depth);
        const unsqShape = [...idxShape];
        unsqShape.splice(axis, 0, 1);

        // 1. Cast indices to INT64 and wrap negative indices: idx < 0 ? idx + depth : idx
        let idx64: ConcreteValueNode = indices;
        if (indices.literalType !== DataType.INT64) {
            [idx64] = builder.createOp("Cast", [indices], { to: DataType.INT64 }, [
                { type: DataType.INT64, shape: idxShape as KnownShape },
            ]);
        }

        const zero = builder.createConstant(`oh_zero_${op.id}`, scalarInt64(0));
        const one = builder.createConstant(`oh_one_${op.id}`, scalarInt64(1));
        const depthConst = builder.createConstant(`oh_depth_${op.id}`, scalarInt64(depth));

        const [isNeg] = builder.createOp("Less", [idx64, zero], {}, [
            { type: DataType.BOOL, shape: idxShape as KnownShape },
        ]);
        const [idxPlusDepth] = builder.createOp("Add", [idx64, depthConst], {}, [
            { type: DataType.INT64, shape: idxShape as KnownShape },
        ]);
        const [normIdx] = builder.createOp("Where", [isNeg, idxPlusDepth, idx64], {}, [
            { type: DataType.INT64, shape: idxShape as KnownShape },
        ]);

        // 2. Unsqueeze indices along `axis`
        const axisConst = builder.createConstant(`oh_axis_${op.id}`, int64Vec([axis]));
        const [idxUnsq] = builder.createOp("Unsqueeze", [normIdx, axisConst], {}, [
            { type: DataType.INT64, shape: unsqShape as KnownShape },
        ]);

        // 3. Build depth range [0..depth-1] reshaped to broadcast along `axis`
        const [range1D] = builder.createOp("Range", [zero, depthConst, one], {}, [
            { type: DataType.INT64, shape: [depth] },
        ]);
        const bcastShape = new Array<number>(outRank).fill(1);
        bcastShape[axis] = depth;
        const bcastShapeConst = builder.createConstant(`oh_bcast_${op.id}`, int64Vec(bcastShape));
        const [rangeND] = builder.createOp("Reshape", [range1D, bcastShapeConst], {}, [
            { type: DataType.INT64, shape: bcastShape as KnownShape },
        ]);

        // 4. Compare and select [off_value, on_value]
        const [mask] = builder.createOp("Equal", [idxUnsq, rangeND], {}, [
            { type: DataType.BOOL, shape: outShape as KnownShape },
        ]);
        const [offVal] = builder.createOp("Gather", [values, zero], { axis: 0 }, [
            { type: valDtype, shape: [] },
        ]);
        const [onVal] = builder.createOp("Gather", [values, one], { axis: 0 }, [
            { type: valDtype, shape: [] },
        ]);
        const [out] = builder.createOp("Where", [mask, onVal, offVal], {}, [
            { type: valDtype, shape: outShape as KnownShape },
        ]);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
