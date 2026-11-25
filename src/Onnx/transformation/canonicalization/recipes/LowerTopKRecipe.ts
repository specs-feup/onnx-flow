import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import ConstantNode from "../../../ConstantNode.js";
import {
    asConcreteValueNode,
    getIntAttr,
    int64Vec,
    makeTensorProto,
    normalizeAxis,
    scalarInt64,
    toStaticShape,
} from "../../../Utils.js";

function readSingleIntConstant(node: ConcreteValueNode): number | undefined {
    if (!node.is(ConstantNode)) return undefined;
    const t = node.as(ConstantNode).constantValue;
    if (Array.isArray(t.int64Data) && t.int64Data.length === 1) return Number(t.int64Data[0]);
    if (Array.isArray(t.int32Data) && t.int32Data.length === 1) return Number(t.int32Data[0]);
    if (t.rawData) {
        const buf = Buffer.isBuffer(t.rawData)
            ? t.rawData
            : Buffer.from((t.rawData as { data: number[] }).data);
        if (t.dataType === DataType.INT64 && buf.length >= 8) return Number(buf.readBigInt64LE(0));
        if (t.dataType === DataType.INT32 && buf.length >= 4) return buf.readInt32LE(0);
    }
    return undefined;
}

export class LowerTopKRecipe implements DecompositionRecipe {
    public readonly name = "LowerTopK";
    public readonly targetOp = "TopK";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 100;
    public readonly shapeSupport = "static" as const;
    public readonly tags = ["unrolled", "constant-k"];
    public readonly producedOps = [
        "ArgMax",
        "ArgMin",
        "ReduceMax",
        "ReduceMin",
        "Range",
        "Reshape",
        "Equal",
        "Where",
        "Concat",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "TopK") return false;
        const inputs = op.getInputs();
        if (!inputs || inputs.length < 2) return false;
        const X = asConcreteValueNode(inputs[0]);
        const K = asConcreteValueNode(inputs[1]);
        if (!Array.isArray(X.shape)) return false;
        const dims = toStaticShape(X.shape);
        if (dims.length === 0 || dims.includes(-1)) return false;
        const k = readSingleIntConstant(K);
        if (k === undefined || k < 1) return false;
        const axis = normalizeAxis(getIntAttr(op, "axis", -1), dims.length);
        return k <= dims[axis];
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const inputs = op.getInputs()!;
        const X = asConcreteValueNode(inputs[0]);
        const K = asConcreteValueNode(inputs[1]);
        const outputs = op.getOutputs();
        const YVals = outputs[0];
        const YIdxs = outputs[1];

        const inDims = toStaticShape(X.shape);
        const rank = inDims.length;
        const dtype = X.literalType as DataType;
        const axis = normalizeAxis(getIntAttr(op, "axis", -1), rank);
        const largest = getIntAttr(op, "largest", 1) !== 0;
        const k = readSingleIntConstant(K)!;
        const axisDim = inDims[axis];

        const stepShape = [...inDims];
        stepShape[axis] = 1;
        const outShape = [...inDims];
        outShape[axis] = k;

        const axesConst = builder.createConstant(`topk_axes_${op.id}`, int64Vec([axis]));

        // Broadcastable index vector 0..axisDim-1 along `axis`
        const startC = builder.createConstant(`topk_start_${op.id}`, scalarInt64(0));
        const limitC = builder.createConstant(`topk_limit_${op.id}`, scalarInt64(axisDim));
        const deltaC = builder.createConstant(`topk_delta_${op.id}`, scalarInt64(1));
        const [idxVec] = builder.createOp("Range", [startC, limitC, deltaC], {}, [
            { type: DataType.INT64, shape: [axisDim] as KnownShape },
        ]);
        const bcastShape = new Array<number>(rank).fill(1);
        bcastShape[axis] = axisDim;
        const bcastShapeC = builder.createConstant(`topk_bshape_${op.id}`, int64Vec(bcastShape));
        const [idxBcast] = builder.createOp("Reshape", [idxVec, bcastShapeC], {}, [
            { type: DataType.INT64, shape: bcastShape as KnownShape },
        ]);

        // Finite sentinel so selected elements are never picked again
        const isInt = dtype === DataType.INT32 || dtype === DataType.INT64;
        const sentinelVal = isInt
            ? largest
                ? -2147483648
                : 2147483647
            : largest
              ? -3.4028234e38
              : 3.4028234e38;
        const sentinelC = builder.createConstant(
            `topk_sent_${op.id}`,
            makeTensorProto(dtype, [], [sentinelVal]),
        );

        const argOp = largest ? "ArgMax" : "ArgMin";
        const redOp = largest ? "ReduceMax" : "ReduceMin";

        let curX: ConcreteValueNode = X;
        const valSlices: ConcreteValueNode[] = [];
        const idxSlices: ConcreteValueNode[] = [];

        for (let t = 0; t < k; t++) {
            const [idxT] = builder.createOp(
                argOp,
                [curX],
                { axis, keepdims: 1, select_last_index: 0 },
                [{ type: DataType.INT64, shape: stepShape as KnownShape }],
            );
            const [valT] = builder.createOp(redOp, [curX, axesConst], { keepdims: 1 }, [
                { type: dtype, shape: stepShape as KnownShape },
            ]);
            valSlices.push(valT);
            idxSlices.push(idxT);

            if (t < k - 1) {
                const [maskT] = builder.createOp("Equal", [idxBcast, idxT], {}, [
                    { type: DataType.BOOL, shape: inDims as KnownShape },
                ]);
                const [nextX] = builder.createOp("Where", [maskT, sentinelC, curX], {}, [
                    { type: dtype, shape: inDims as KnownShape },
                ]);
                curX = nextX;
            }
        }

        let finalVals: ConcreteValueNode;
        let finalIdxs: ConcreteValueNode;
        if (k === 1) {
            finalVals = valSlices[0];
            finalIdxs = idxSlices[0];
        } else {
            [finalVals] = builder.createOp("Concat", valSlices, { axis }, [
                { type: dtype, shape: outShape as KnownShape },
            ]);
            [finalIdxs] = builder.createOp("Concat", idxSlices, { axis }, [
                { type: DataType.INT64, shape: outShape as KnownShape },
            ]);
        }

        builder.replaceAllUsesWith(YVals, finalVals);
        builder.replaceAllUsesWith(YIdxs, finalIdxs);
        builder.removeNode(op);
    }
}
