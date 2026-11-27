import type OnnxGraph from "../../ir/OnnxGraph.js";
import type OperationNode from "../../ir/OperationNode.js";
import ConstantNode from "../../ir/ConstantNode.js";
import { GraphBuilder } from "../../ir/GraphBuilder.js";
import { DataType, type TensorProto } from "../../ir/OnnxTypes.js";
import {
    getIntAttr,
    getIntsAttr,
    makeTensorProto,
    normalizeAxis,
    readTensorData,
    topologicalSortOperationNodes,
    toStaticShape,
} from "../../utils/index.js";
import { BaseGraphPass } from "../GraphPass.js";
import {
    asRewritableIntermediate,
    cleanupOrphanConstant,
    collectCapturedIds,
} from "./OptimizationUtils.js";

const FOLDABLE_DTYPES = new Set<DataType>([
    DataType.FLOAT,
    DataType.DOUBLE,
    DataType.INT32,
    DataType.INT64,
    DataType.BOOL,
]);

const INT_DTYPES = new Set<DataType>([DataType.INT32, DataType.INT64]);

interface FoldedTensor {
    dtype: DataType;
    shape: number[];
    values: number[];
}

function shapeSize(shape: readonly number[]): number {
    return shape.reduce((a, b) => a * b, 1);
}

function computeStrides(shape: readonly number[]): number[] {
    const strides = new Array<number>(shape.length).fill(1);
    for (let i = shape.length - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

function broadcastShapes(a: readonly number[], b: readonly number[]): number[] | undefined {
    const rank = Math.max(a.length, b.length);
    const out = new Array<number>(rank);
    for (let i = 0; i < rank; i++) {
        const da = i < rank - a.length ? 1 : a[i - (rank - a.length)];
        const db = i < rank - b.length ? 1 : b[i - (rank - b.length)];
        if (da !== db && da !== 1 && db !== 1) return undefined;
        out[i] = Math.max(da, db);
    }
    return out;
}

function broadcastIndex(
    flatIdx: number,
    outShape: readonly number[],
    outStrides: readonly number[],
    inShape: readonly number[],
    inStrides: readonly number[],
): number {
    const pad = outShape.length - inShape.length;
    let rem = flatIdx;
    let idx = 0;
    for (let i = 0; i < outShape.length; i++) {
        const coord = Math.floor(rem / outStrides[i]);
        rem -= coord * outStrides[i];
        const j = i - pad;
        if (j >= 0 && inShape[j] > 1) {
            idx += coord * inStrides[j];
        }
    }
    return idx;
}

/**
 * Evaluates pure operations whose inputs are all `ConstantNode`s (up to `maxElements`)
 * at compile time, replacing them with folded `ConstantNode`s across all nested regions.
 *
 * Particularly effective at collapsing dynamic `Shape`/`Gather`/`Concat`/`Slice`/`Add`/`Mul`/`ReduceProd`
 * indexing and bound subgraphs once shapes become static.
 */
export class ConstantFoldingPass extends BaseGraphPass {
    public override readonly name = "ConstantFolding";

    constructor(private readonly maxElements: number = 4096) {
        super();
    }

    public override run(graph: OnnxGraph.Class): boolean {
        let changed = this.runOnRegions(graph);
        const capturedIds = collectCapturedIds(graph);

        // Single topological pass suffices: folding replaces ops with ConstantNodes
        // in producer-before-consumer order, so downstream ops immediately see folded inputs.
        const ops = topologicalSortOperationNodes(graph);
        for (const op of ops) {
            if (!graph.hasNode(op.id)) continue;
            const outputs = op.getOutputs();
            if (outputs.length !== 1) continue;

            const outTensor = asRewritableIntermediate(outputs[0], capturedIds);
            if (outTensor === undefined || outTensor.outgoers.length === 0) continue;

            const folded = this.tryFold(op);
            if (folded === undefined) continue;
            if (shapeSize(folded.shape) > this.maxElements) continue;

            const inputs = op.getInputs() ?? [];
            const proto: TensorProto = makeTensorProto(folded.dtype, folded.shape, folded.values);
            const builder = new GraphBuilder(graph, `cf_${op.id}`);
            const foldedConst = builder.createConstant(`folded_${op.id}`, proto);
            builder.replaceAllUsesWith(outTensor, foldedConst);
            op.remove();
            if (graph.hasNode(outTensor.id) && outTensor.outgoers.length === 0) {
                outTensor.remove();
            }
            for (const inp of inputs) {
                cleanupOrphanConstant(inp, graph, capturedIds);
            }
            changed = true;
        }

        return changed;
    }

    private readConstTensor(node: ConstantNode.Class): FoldedTensor | undefined {
        const dtype = node.literalType;
        if (!FOLDABLE_DTYPES.has(dtype)) return undefined;
        const shape = toStaticShape(node.shape);
        if (shape.includes(-1)) return undefined;
        const size = shapeSize(shape);
        if (size > this.maxElements) return undefined;
        if (size === 0) return { dtype, shape, values: [] };
        const values = readTensorData(node);
        if (values === undefined || values.length !== size) return undefined;
        return { dtype, shape, values };
    }

    private readAllConstantInputs(op: OperationNode.Class): FoldedTensor[] | undefined {
        const inputs = op.getInputs();
        if (!inputs || inputs.length === 0) return undefined;
        const result: FoldedTensor[] = [];
        for (const inp of inputs) {
            if (!inp.is(ConstantNode)) return undefined;
            const t = this.readConstTensor(inp.as(ConstantNode));
            if (t === undefined) return undefined;
            result.push(t);
        }
        return result;
    }

    private tryFold(op: OperationNode.Class): FoldedTensor | undefined {
        // Shape(X) can fold even when X is not a ConstantNode as long as X has a static shape
        if (op.type === "Shape") {
            const inp = op.getInputs()?.[0];
            if (inp === undefined) return undefined;
            const sh = toStaticShape(inp.shape);
            if (sh.length === 0 || sh.includes(-1)) return undefined;
            return { dtype: DataType.INT64, shape: [sh.length], values: sh };
        }

        const constInputs = this.readAllConstantInputs(op);
        if (constInputs === undefined) return undefined;

        switch (op.type) {
            case "Add":
            case "Sub":
            case "Mul":
            case "Div":
            case "Mod":
            case "Max":
            case "Min":
            case "Equal":
            case "Less":
            case "Greater":
            case "And":
            case "Or":
                return constInputs.length === 2
                    ? this.foldBinary(op, constInputs[0], constInputs[1])
                    : undefined;
            case "Neg":
            case "Abs":
            case "Not":
            case "Floor":
            case "Ceil":
            case "Sqrt":
            case "Cast":
                return constInputs.length === 1 ? this.foldUnary(op, constInputs[0]) : undefined;
            case "Unsqueeze":
                return this.foldUnsqueeze(op, constInputs);
            case "Squeeze":
                return this.foldSqueeze(op, constInputs);
            case "Reshape":
                return constInputs.length >= 2
                    ? this.foldReshape(op, constInputs[0], constInputs[1])
                    : undefined;
            case "Concat":
                return this.foldConcat(op, constInputs);
            case "Gather":
                return constInputs.length === 2
                    ? this.foldGather(op, constInputs[0], constInputs[1])
                    : undefined;
            case "Slice":
                return this.foldSlice(constInputs);
            case "Transpose":
                return constInputs.length === 1
                    ? this.foldTranspose(op, constInputs[0])
                    : undefined;
            case "ReduceProd":
            case "ReduceSum":
                return this.foldReduce(op, constInputs);
            case "Where":
                return constInputs.length === 3
                    ? this.foldWhere(constInputs[0], constInputs[1], constInputs[2])
                    : undefined;
            default:
                return undefined;
        }
    }

    private foldBinary(
        op: OperationNode.Class,
        a: FoldedTensor,
        b: FoldedTensor,
    ): FoldedTensor | undefined {
        const outShape = broadcastShapes(a.shape, b.shape);
        if (outShape === undefined) return undefined;
        const outSize = shapeSize(outShape);
        if (outSize > this.maxElements) return undefined;

        const isComp = op.type === "Equal" || op.type === "Less" || op.type === "Greater";
        const isLogic = op.type === "And" || op.type === "Or";
        const outDtype = isComp || isLogic ? DataType.BOOL : a.dtype;
        const isInt = INT_DTYPES.has(a.dtype);
        const fmod = getIntAttr(op, "fmod", 0);

        const outStrides = computeStrides(outShape);
        const aStrides = computeStrides(a.shape);
        const bStrides = computeStrides(b.shape);
        const values = new Array<number>(outSize);

        for (let i = 0; i < outSize; i++) {
            const va = a.values[broadcastIndex(i, outShape, outStrides, a.shape, aStrides)];
            const vb = b.values[broadcastIndex(i, outShape, outStrides, b.shape, bStrides)];
            let res: number;

            switch (op.type) {
                case "Add":
                    res = va + vb;
                    break;
                case "Sub":
                    res = va - vb;
                    break;
                case "Mul":
                    res = va * vb;
                    break;
                case "Div":
                    if (vb === 0) return undefined;
                    res = isInt ? Math.trunc(va / vb) : va / vb;
                    break;
                case "Mod":
                    if (vb === 0) return undefined;
                    if (fmod === 1) {
                        res = va % vb;
                    } else if (isInt) {
                        res = ((va % vb) + vb) % vb;
                    } else {
                        return undefined;
                    }
                    break;
                case "Max":
                    res = Math.max(va, vb);
                    break;
                case "Min":
                    res = Math.min(va, vb);
                    break;
                case "Equal":
                    res = va === vb ? 1 : 0;
                    break;
                case "Less":
                    res = va < vb ? 1 : 0;
                    break;
                case "Greater":
                    res = va > vb ? 1 : 0;
                    break;
                case "And":
                    res = va !== 0 && vb !== 0 ? 1 : 0;
                    break;
                case "Or":
                    res = va !== 0 || vb !== 0 ? 1 : 0;
                    break;
                default:
                    return undefined;
            }

            if (INT_DTYPES.has(outDtype) && !Number.isFinite(res)) return undefined;
            values[i] = INT_DTYPES.has(outDtype) ? Math.trunc(res) : res;
        }

        return { dtype: outDtype, shape: outShape, values };
    }

    private foldUnary(op: OperationNode.Class, a: FoldedTensor): FoldedTensor | undefined {
        if (op.type === "Cast") {
            const to = getIntAttr(op, "to", DataType.UNDEFINED) as DataType;
            if (!FOLDABLE_DTYPES.has(to)) return undefined;
            const values = a.values.map((v) => {
                if (to === DataType.BOOL) return v !== 0 ? 1 : 0;
                if (INT_DTYPES.has(to)) return Math.trunc(v);
                return v;
            });
            if (INT_DTYPES.has(to) && values.some((v) => !Number.isFinite(v))) return undefined;
            return { dtype: to, shape: [...a.shape], values };
        }

        const outDtype = op.type === "Not" ? DataType.BOOL : a.dtype;
        const values = new Array<number>(a.values.length);
        for (let i = 0; i < a.values.length; i++) {
            const v = a.values[i];
            switch (op.type) {
                case "Neg":
                    values[i] = -v;
                    break;
                case "Abs":
                    values[i] = Math.abs(v);
                    break;
                case "Not":
                    values[i] = v === 0 ? 1 : 0;
                    break;
                case "Floor":
                    values[i] = Math.floor(v);
                    break;
                case "Ceil":
                    values[i] = Math.ceil(v);
                    break;
                case "Sqrt":
                    if (v < 0) return undefined;
                    values[i] = Math.sqrt(v);
                    break;
                default:
                    return undefined;
            }
        }
        return { dtype: outDtype, shape: [...a.shape], values };
    }

    private foldUnsqueeze(
        op: OperationNode.Class,
        inputs: FoldedTensor[],
    ): FoldedTensor | undefined {
        const data = inputs[0];
        const rawAxes = inputs.length >= 2 ? inputs[1].values : getIntsAttr(op, "axes", []);
        if (rawAxes.length === 0) return undefined;
        const outRank = data.shape.length + rawAxes.length;
        const normAxes = rawAxes.map((ax) => (ax < 0 ? ax + outRank : ax));
        const axisSet = new Set(normAxes);
        if (axisSet.size !== normAxes.length) return undefined;
        if (normAxes.some((ax) => ax < 0 || ax >= outRank)) return undefined;

        const outShape: number[] = [];
        let srcIdx = 0;
        for (let i = 0; i < outRank; i++) {
            if (axisSet.has(i)) outShape.push(1);
            else outShape.push(data.shape[srcIdx++]);
        }
        return { dtype: data.dtype, shape: outShape, values: [...data.values] };
    }

    private foldSqueeze(op: OperationNode.Class, inputs: FoldedTensor[]): FoldedTensor | undefined {
        const data = inputs[0];
        const rawAxes = inputs.length >= 2 ? inputs[1].values : getIntsAttr(op, "axes", []);
        const axisSet = new Set<number>();
        if (rawAxes.length > 0) {
            for (const ax of rawAxes) {
                const n = normalizeAxis(ax, data.shape.length);
                if (n < 0 || n >= data.shape.length || data.shape[n] !== 1) return undefined;
                axisSet.add(n);
            }
        } else {
            data.shape.forEach((d, i) => {
                if (d === 1) axisSet.add(i);
            });
        }
        const outShape = data.shape.filter((_, i) => !axisSet.has(i));
        return { dtype: data.dtype, shape: outShape, values: [...data.values] };
    }

    private foldReshape(
        op: OperationNode.Class,
        data: FoldedTensor,
        targetSpec: FoldedTensor,
    ): FoldedTensor | undefined {
        const allowZero = getIntAttr(op, "allowzero", 0);
        const total = data.values.length;
        const resolved = [...targetSpec.values];
        let inferIdx = -1;
        let knownProd = 1;

        for (let i = 0; i < resolved.length; i++) {
            if (resolved[i] === 0 && allowZero === 0) {
                if (i >= data.shape.length) return undefined;
                resolved[i] = data.shape[i];
            }
            if (resolved[i] === -1) {
                if (inferIdx !== -1) return undefined;
                inferIdx = i;
            } else if (resolved[i] < 0) {
                return undefined;
            } else {
                knownProd *= resolved[i];
            }
        }

        if (inferIdx !== -1) {
            if (knownProd === 0 || total % knownProd !== 0) return undefined;
            resolved[inferIdx] = total / knownProd;
        }

        if (shapeSize(resolved) !== total) return undefined;
        return { dtype: data.dtype, shape: resolved, values: [...data.values] };
    }

    private foldConcat(op: OperationNode.Class, inputs: FoldedTensor[]): FoldedTensor | undefined {
        if (inputs.length === 0) return undefined;
        const rank = inputs[0].shape.length;
        if (rank === 0) return undefined;
        const axis = normalizeAxis(getIntAttr(op, "axis", 0), rank);
        if (axis < 0 || axis >= rank) return undefined;

        const dtype = inputs[0].dtype;
        let axisSum = 0;
        for (const t of inputs) {
            if (t.dtype !== dtype || t.shape.length !== rank) return undefined;
            for (let d = 0; d < rank; d++) {
                if (d !== axis && t.shape[d] !== inputs[0].shape[d]) return undefined;
            }
            axisSum += t.shape[axis];
        }

        const outShape = [...inputs[0].shape];
        outShape[axis] = axisSum;
        const outSize = shapeSize(outShape);
        if (outSize > this.maxElements) return undefined;

        const outerSize = outShape.slice(0, axis).reduce((a, b) => a * b, 1);
        const innerSize = outShape.slice(axis + 1).reduce((a, b) => a * b, 1);
        const values: number[] = [];

        for (let o = 0; o < outerSize; o++) {
            for (const t of inputs) {
                const chunk = t.shape[axis] * innerSize;
                const start = o * chunk;
                for (let k = 0; k < chunk; k++) {
                    values.push(t.values[start + k]);
                }
            }
        }
        return { dtype, shape: outShape, values };
    }

    private foldGather(
        op: OperationNode.Class,
        data: FoldedTensor,
        indices: FoldedTensor,
    ): FoldedTensor | undefined {
        if (data.shape.length === 0) return undefined;
        const axis = normalizeAxis(getIntAttr(op, "axis", 0), data.shape.length);
        if (axis < 0 || axis >= data.shape.length) return undefined;

        const axisDim = data.shape[axis];
        const normIndices = indices.values.map((idx) => (idx < 0 ? idx + axisDim : idx));
        if (normIndices.some((idx) => idx < 0 || idx >= axisDim || !Number.isInteger(idx))) {
            return undefined;
        }

        const outShape = [
            ...data.shape.slice(0, axis),
            ...indices.shape,
            ...data.shape.slice(axis + 1),
        ];
        if (shapeSize(outShape) > this.maxElements) return undefined;

        const outerSize = data.shape.slice(0, axis).reduce((a, b) => a * b, 1);
        const innerSize = data.shape.slice(axis + 1).reduce((a, b) => a * b, 1);
        const values: number[] = [];

        for (let o = 0; o < outerSize; o++) {
            for (const idx of normIndices) {
                const base = (o * axisDim + idx) * innerSize;
                for (let k = 0; k < innerSize; k++) {
                    values.push(data.values[base + k]);
                }
            }
        }
        return { dtype: data.dtype, shape: outShape, values };
    }

    private foldSlice(inputs: FoldedTensor[]): FoldedTensor | undefined {
        if (inputs.length < 3) return undefined;
        const data = inputs[0];
        const starts = inputs[1].values;
        const ends = inputs[2].values;
        const rank = data.shape.length;
        if (starts.length !== ends.length) return undefined;

        const axes =
            inputs.length >= 4
                ? inputs[3].values.map((ax) => normalizeAxis(ax, rank))
                : starts.map((_, i) => i);
        const steps = inputs.length >= 5 ? inputs[4].values : starts.map(() => 1);
        if (axes.length !== starts.length || steps.length !== starts.length) return undefined;

        const axisSlice = new Map<number, { start: number; step: number; len: number }>();
        for (let i = 0; i < axes.length; i++) {
            const ax = axes[i];
            const step = steps[i];
            if (ax < 0 || ax >= rank || step === 0) return undefined;
            const dim = data.shape[ax];
            let s = starts[i] < 0 ? starts[i] + dim : starts[i];
            let e = ends[i] < 0 ? ends[i] + dim : ends[i];
            if (step > 0) {
                s = Math.max(0, Math.min(dim, s));
                e = Math.max(0, Math.min(dim, e));
                const len = e > s ? Math.ceil((e - s) / step) : 0;
                axisSlice.set(ax, { start: s, step, len });
            } else {
                s = Math.max(-1, Math.min(dim - 1, s));
                e = Math.max(-1, Math.min(dim - 1, e));
                const len = s > e ? Math.ceil((s - e) / -step) : 0;
                axisSlice.set(ax, { start: s, step, len });
            }
        }

        const outShape = data.shape.map((d, ax) => axisSlice.get(ax)?.len ?? d);
        const outSize = shapeSize(outShape);
        if (outSize > this.maxElements) return undefined;

        const inStrides = computeStrides(data.shape);
        const outStrides = computeStrides(outShape);
        const values = new Array<number>(outSize);

        for (let i = 0; i < outSize; i++) {
            let rem = i;
            let srcIdx = 0;
            for (let d = 0; d < rank; d++) {
                const coord = outStrides[d] > 0 ? Math.floor(rem / outStrides[d]) : 0;
                rem -= coord * outStrides[d];
                const sl = axisSlice.get(d);
                const srcCoord = sl !== undefined ? sl.start + coord * sl.step : coord;
                srcIdx += srcCoord * inStrides[d];
            }
            values[i] = data.values[srcIdx];
        }
        return { dtype: data.dtype, shape: outShape, values };
    }

    private foldTranspose(op: OperationNode.Class, data: FoldedTensor): FoldedTensor | undefined {
        const rank = data.shape.length;
        const defaultPerm = Array.from({ length: rank }, (_, i) => rank - 1 - i);
        const perm = getIntsAttr(op, "perm", defaultPerm);
        if (perm.length !== rank) return undefined;

        const outShape = perm.map((p) => data.shape[p]);
        const outSize = shapeSize(outShape);
        if (outSize > this.maxElements) return undefined;

        const inStrides = computeStrides(data.shape);
        const outStrides = computeStrides(outShape);
        const values = new Array<number>(outSize);

        for (let i = 0; i < outSize; i++) {
            let rem = i;
            let srcIdx = 0;
            for (let d = 0; d < rank; d++) {
                const coord = Math.floor(rem / outStrides[d]);
                rem -= coord * outStrides[d];
                srcIdx += coord * inStrides[perm[d]];
            }
            values[i] = data.values[srcIdx];
        }
        return { dtype: data.dtype, shape: outShape, values };
    }

    private foldReduce(op: OperationNode.Class, inputs: FoldedTensor[]): FoldedTensor | undefined {
        const data = inputs[0];
        const rank = data.shape.length;
        const keepDims = getIntAttr(op, "keepdims", 1) !== 0;
        const noopWithEmpty = getIntAttr(op, "noop_with_empty_axes", 0) !== 0;

        const rawAxes = inputs.length >= 2 ? inputs[1].values : getIntsAttr(op, "axes", []);
        if (rawAxes.length === 0 && noopWithEmpty) {
            return { dtype: data.dtype, shape: [...data.shape], values: [...data.values] };
        }
        const axes =
            rawAxes.length > 0
                ? new Set(rawAxes.map((a) => normalizeAxis(a, rank)))
                : new Set(Array.from({ length: rank }, (_, i) => i));

        const outShape: number[] = [];
        for (let d = 0; d < rank; d++) {
            if (axes.has(d)) {
                if (keepDims) outShape.push(1);
            } else {
                outShape.push(data.shape[d]);
            }
        }
        const outSize = shapeSize(outShape);
        if (outSize > this.maxElements) return undefined;

        const isProd = op.type === "ReduceProd";
        const values = new Array<number>(outSize).fill(isProd ? 1 : 0);
        const inStrides = computeStrides(data.shape);
        const outStrides = computeStrides(outShape);

        for (let i = 0; i < data.values.length; i++) {
            let rem = i;
            let outIdx = 0;
            let outDim = 0;
            for (let d = 0; d < rank; d++) {
                const coord = Math.floor(rem / inStrides[d]);
                rem -= coord * inStrides[d];
                if (axes.has(d)) {
                    if (keepDims) outDim++;
                } else {
                    outIdx += coord * outStrides[outDim++];
                }
            }
            values[outIdx] = isProd
                ? values[outIdx] * data.values[i]
                : values[outIdx] + data.values[i];
        }
        return { dtype: data.dtype, shape: outShape, values };
    }

    private foldWhere(
        cond: FoldedTensor,
        x: FoldedTensor,
        y: FoldedTensor,
    ): FoldedTensor | undefined {
        const xyShape = broadcastShapes(x.shape, y.shape);
        if (xyShape === undefined) return undefined;
        const outShape = broadcastShapes(cond.shape, xyShape);
        if (outShape === undefined) return undefined;
        const outSize = shapeSize(outShape);
        if (outSize > this.maxElements) return undefined;

        const outStrides = computeStrides(outShape);
        const cStrides = computeStrides(cond.shape);
        const xStrides = computeStrides(x.shape);
        const yStrides = computeStrides(y.shape);
        const values = new Array<number>(outSize);

        for (let i = 0; i < outSize; i++) {
            const c = cond.values[broadcastIndex(i, outShape, outStrides, cond.shape, cStrides)];
            values[i] =
                c !== 0
                    ? x.values[broadcastIndex(i, outShape, outStrides, x.shape, xStrides)]
                    : y.values[broadcastIndex(i, outShape, outStrides, y.shape, yStrides)];
        }
        return { dtype: x.dtype, shape: outShape, values };
    }
}
