import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import {
    getIntAttr,
    getStringAttr,
    int64Vec,
    normalizeAxis,
    scalarInt64,
} from "../../../utils/index.js";

/**
 * Canonicalizes TensorScatter (Opset 24) KV-cache update to ScatterElements:
 *   present_cache = ScatterElements(past_cache, indices, update, axis=axis)
 */
export class LowerTensorScatterRecipe implements DecompositionRecipe {
    public readonly name = "LowerTensorScatter";
    public readonly targetOp = "TensorScatter";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly producedOps = [
        "Shape",
        "Gather",
        "Range",
        "Reshape",
        "Add",
        "Mod",
        "Expand",
        "ScatterElements",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "TensorScatter") return false;
        const ins = op.getInputs();
        return !!ins && ins.length >= 2 && ins[0].shape.length >= 1 && ins[1].shape.length >= 1;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const pastCache = ins[0];
        const update = ins[1];
        const writeIndices = ins.length >= 3 && ins[2].id !== "" ? ins[2] : undefined;
        const Y = op.getOutputs()[0];

        const dtype =
            (pastCache.literalType as DataType | undefined) ??
            (Y.literalType as DataType | undefined) ??
            DataType.FLOAT;
        const rank = pastCache.shape.length;
        const axis = normalizeAxis(getIntAttr(op, "axis", -2), rank);
        const mode = getStringAttr(op, "mode", "linear");

        const [shapeCache] = builder.createOp("Shape", [pastCache], {}, [
            { type: DataType.INT64, shape: [rank] },
        ]);
        const [shapeUpdate] = builder.createOp("Shape", [update], {}, [
            { type: DataType.INT64, shape: [rank] },
        ]);

        const axIdx = builder.createConstant(`ts_ax_${op.id}`, scalarInt64(axis));
        const [seqLen] = builder.createOp("Gather", [shapeUpdate, axIdx], { axis: 0 }, [
            { type: DataType.INT64, shape: [] },
        ]);

        const zeroI64 = builder.createConstant(`ts_z_${op.id}`, scalarInt64(0));
        const oneI64 = builder.createConstant(`ts_o_${op.id}`, scalarInt64(1));
        const [seqRange] = builder.createOp("Range", [zeroI64, seqLen, oneI64], {}, [
            { type: DataType.INT64, shape: [update.shape[axis] ?? -1] },
        ]);

        const rangeShapeArr = new Array<number>(rank).fill(1);
        rangeShapeArr[axis] = -1;
        const rangeShapeConst = builder.createConstant(
            `ts_rshape_${op.id}`,
            int64Vec(rangeShapeArr),
        );
        const rangeOutShape: KnownShape = Array.from({ length: rank }, (_, i) =>
            i === axis ? (update.shape[axis] ?? -1) : 1,
        );
        let [idxND] = builder.createOp("Reshape", [seqRange, rangeShapeConst], {}, [
            { type: DataType.INT64, shape: rangeOutShape },
        ]);

        if (writeIndices) {
            const wiCast: ConcreteValueNode =
                writeIndices.literalType === DataType.INT64
                    ? writeIndices
                    : builder.createOp("Cast", [writeIndices], { to: DataType.INT64 }, [
                          { type: DataType.INT64, shape: writeIndices.shape as KnownShape },
                      ])[0];

            const wiShapeArr = new Array<number>(rank).fill(1);
            wiShapeArr[0] = -1;
            const wiShapeConst = builder.createConstant(`ts_wshape_${op.id}`, int64Vec(wiShapeArr));
            const wiOutShape: KnownShape = Array.from({ length: rank }, (_, i) =>
                i === 0 ? (pastCache.shape[0] ?? -1) : 1,
            );
            const [wiND] = builder.createOp("Reshape", [wiCast, wiShapeConst], {}, [
                { type: DataType.INT64, shape: wiOutShape },
            ]);

            const addedShape: KnownShape = Array.from({ length: rank }, (_, i) =>
                i === 0 ? (pastCache.shape[0] ?? -1) : i === axis ? (update.shape[axis] ?? -1) : 1,
            );
            [idxND] = builder.createOp("Add", [wiND, idxND], {}, [
                { type: DataType.INT64, shape: addedShape },
            ]);
        }

        if (mode === "circular") {
            const [maxSeqLen] = builder.createOp("Gather", [shapeCache, axIdx], { axis: 0 }, [
                { type: DataType.INT64, shape: [] },
            ]);
            [idxND] = builder.createOp("Mod", [idxND, maxSeqLen], { fmod: 0 }, [
                { type: DataType.INT64, shape: idxND.shape as KnownShape },
            ]);
        }

        const [expandedIdx] = builder.createOp("Expand", [idxND, shapeUpdate], {}, [
            { type: DataType.INT64, shape: update.shape as KnownShape },
        ]);

        const [out] = builder.createOp(
            "ScatterElements",
            [pastCache, expandedIdx, update],
            { axis },
            [{ type: dtype, shape: Y.shape as KnownShape }],
        );

        builder.replaceAllUsesWith(Y, out);
        op.remove();
    }
}
