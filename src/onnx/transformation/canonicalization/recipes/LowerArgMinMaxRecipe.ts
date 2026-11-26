import type OperationNode from "../../../ir/OperationNode.js";
import type { GraphBuilder } from "../../../ir/GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../ir/OnnxTypes.js";
import { DataType } from "../../../ir/OnnxTypes.js";
import {
    getIntAttr,
    int64Vec,
    normalizeAxis,
    scalarInt64,
    toStaticShape,
} from "../../../utils/index.js";

abstract class BaseLowerArgMinMaxRecipe implements DecompositionRecipe {
    public abstract readonly name: string;
    public abstract readonly targetOp: "ArgMax" | "ArgMin";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 100;
    public readonly shapeSupport = "dynamic-batch" as const;
    public readonly tags = ["mask-based"];
    public readonly producedOps = ["ReduceMax", "ReduceMin", "Equal", "Range", "Reshape", "Where"];

    match(op: OperationNode.Class): boolean {
        if (op.type !== this.targetOp) return false;
        const X = op.getInputs()?.[0];
        if (!X || !Array.isArray(X.shape)) return false;
        const dims = toStaticShape(X.shape);
        if (dims.length === 0) return false;
        const axis = normalizeAxis(getIntAttr(op, "axis", 0), dims.length);
        return dims[axis] > 0;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = op.getInputs()![0] as ConcreteValueNode;
        const Y = op.getOutputs()[0];
        const inDims = toStaticShape(X.shape);
        const rank = inDims.length;
        const inDtype = X.literalType as DataType;

        const axis = normalizeAxis(getIntAttr(op, "axis", 0), rank);
        const keepdims = getIntAttr(op, "keepdims", 1);
        const selectLast = getIntAttr(op, "select_last_index", 0) !== 0;
        const axisDim = inDims[axis];

        const keepdimsShape = [...inDims];
        keepdimsShape[axis] = 1;

        const outShape = keepdims === 1 ? keepdimsShape : inDims.filter((_, i) => i !== axis);

        const axesConst = builder.createConstant(`arg_axes_${op.id}`, int64Vec([axis]));

        // 1. Find extremum value along `axis` with keepdims=1
        const extOp = this.targetOp === "ArgMax" ? "ReduceMax" : "ReduceMin";
        const [extVal] = builder.createOp(extOp, [X, axesConst], { keepdims: 1 }, [
            { type: inDtype, shape: keepdimsShape as KnownShape },
        ]);

        // 2. Mask all positions matching the extremum
        const [isExt] = builder.createOp("Equal", [X, extVal], {}, [
            { type: DataType.BOOL, shape: inDims as KnownShape },
        ]);

        // 3. Build 0..axisDim-1 index vector and reshape to broadcast along `axis`
        const startC = builder.createConstant(`arg_start_${op.id}`, scalarInt64(0));
        const limitC = builder.createConstant(`arg_limit_${op.id}`, scalarInt64(axisDim));
        const deltaC = builder.createConstant(`arg_delta_${op.id}`, scalarInt64(1));
        const [idxVec] = builder.createOp("Range", [startC, limitC, deltaC], {}, [
            { type: DataType.INT64, shape: [axisDim] as KnownShape },
        ]);

        const bcastShape = new Array<number>(rank).fill(1);
        bcastShape[axis] = axisDim;
        const bcastShapeC = builder.createConstant(`arg_bshape_${op.id}`, int64Vec(bcastShape));
        const [idxBcast] = builder.createOp("Reshape", [idxVec, bcastShapeC], {}, [
            { type: DataType.INT64, shape: bcastShape as KnownShape },
        ]);

        // 4. Replace non-extremum positions with sentinel and reduce to first/last matching index
        const sentinelVal = selectLast ? -1 : axisDim;
        const sentinelC = builder.createConstant(`arg_sent_${op.id}`, scalarInt64(sentinelVal));
        const [maskedIdx] = builder.createOp("Where", [isExt, idxBcast, sentinelC], {}, [
            { type: DataType.INT64, shape: inDims as KnownShape },
        ]);

        const tieOp = selectLast ? "ReduceMax" : "ReduceMin";
        const [out] = builder.createOp(tieOp, [maskedIdx, axesConst], { keepdims }, [
            { type: DataType.INT64, shape: outShape as KnownShape },
        ]);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}

export class LowerArgMaxRecipe extends BaseLowerArgMinMaxRecipe {
    public readonly name = "LowerArgMax";
    public readonly targetOp = "ArgMax" as const;
}

export class LowerArgMinRecipe extends BaseLowerArgMinMaxRecipe {
    public readonly name = "LowerArgMin";
    public readonly targetOp = "ArgMin" as const;
}
