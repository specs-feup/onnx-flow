import type {
    RawOnnxGraph,
    RawOnnxModel,
    RawOnnxNode,
    RawOnnxTensorType,
    TensorProto,
} from "../OnnxTypes.js";
import { AttributeType, DataType } from "../OnnxTypes.js";
import { LATEST_ONNX_OPSET } from "../utils/Constants.js";

// --- Helpers ---
function createInt64Initializer(name: string, values: (number | string | bigint)[]): TensorProto {
    return {
        name,
        dataType: DataType.INT64,
        dims: [values.length],
        int64Data: values.map((v) => BigInt(v)),
    };
}

function createFloatInitializer(name: string, value: number): TensorProto {
    return {
        name,
        dataType: DataType.FLOAT,
        dims: [],
        floatData: [value],
    };
}

function getOpType(node: RawOnnxNode): string | undefined {
    return node.opType ?? node.op_type;
}

function createNameAllocator(graphProto: RawOnnxGraph): (prefix: string) => string {
    const used = new Set<string>();
    for (const init of graphProto.initializer ?? []) {
        if (init.name !== undefined && init.name !== "") used.add(init.name);
    }
    for (const vi of [
        ...(graphProto.input ?? []),
        ...(graphProto.output ?? []),
        ...(graphProto.valueInfo ?? graphProto.value_info ?? []),
    ]) {
        if (vi.name) used.add(vi.name);
    }
    for (const node of graphProto.node ?? []) {
        for (const name of [...(node.input ?? []), ...(node.output ?? [])]) {
            if (name) used.add(name);
        }
    }

    let counter = 0;
    return (prefix: string): string => {
        const sanitized = prefix.replace(/[^A-Za-z0-9_]/g, "_");
        let candidate = `${sanitized}_${counter++}`;
        while (used.has(candidate)) {
            candidate = `${sanitized}_${counter++}`;
        }
        used.add(candidate);
        return candidate;
    };
}

function resolveTensorRank(graphProto: RawOnnxGraph, tensorName: string): number | undefined {
    const init = graphProto.initializer?.find((i) => i.name === tensorName);
    if (init?.dims) return init.dims.length;

    const allValueInfos = [
        ...(graphProto.input ?? []),
        ...(graphProto.valueInfo ?? graphProto.value_info ?? []),
        ...(graphProto.output ?? []),
    ];
    for (const vi of allValueInfos) {
        if (vi.name !== tensorName) continue;
        const tensorType: RawOnnxTensorType | undefined =
            vi.type?.tensorType ?? vi.type?.tensor_type;
        if (tensorType?.shape?.dim) {
            return tensorType.shape.dim.length;
        }
    }
    return undefined;
}

function resolveSourceOpset(data: RawOnnxModel): number | undefined {
    const opsets = data.opsetImport ?? data.opset_import;
    if (!opsets || opsets.length === 0) return undefined;
    const defaultEntry = opsets.find((o) => (o.domain ?? "") === "" || o.domain === "ai.onnx");
    if (!defaultEntry || defaultEntry.version === undefined) return undefined;
    return Number(defaultEntry.version);
}

function moveAttributeToInput(
    node: RawOnnxNode,
    graphProto: RawOnnxGraph,
    attrName: string,
    inputIndex: number,
    type: "int" | "float" | "ints",
    nextName: (prefix: string) => string,
): void {
    const idx = node.attribute?.findIndex((a) => a.name === attrName);
    if (idx === undefined || idx === -1) return;

    const attr = node.attribute![idx];
    const opType = getOpType(node) ?? "UnknownOp";
    const name = nextName(`${node.name ?? opType}_${attrName}`);

    let init: TensorProto;
    if (type === "ints") {
        const val = attr.ints ?? [0];
        init = createInt64Initializer(name, val);
    } else if (type === "int") {
        const val = [Number(attr.i ?? 0)];
        init = createInt64Initializer(name, val);
    } else {
        const val = Number(attr.f ?? 0);
        init = createFloatInitializer(name, val);
    }

    graphProto.initializer ??= [];
    graphProto.initializer.push(init);

    node.input ??= [];
    while (node.input.length < inputIndex) node.input.push("");
    node.input[inputIndex] = name;

    node.attribute?.splice(idx, 1);
}

// --- Adapters ---
/**
 * Adapter: FreezeGraphOverridableInputs
 *
 * ONNX allows "Overridable Initializers" (names present in both `initializer` and `input`).
 * We strictly split Data (`ConstantNode`) vs Flow (`TensorNode`) so `initGraph` does not
 * overwrite constant weights with empty graph input placeholders.
 */
function freezeGraphOverridableInputs(graphProto: RawOnnxGraph): void {
    if (!graphProto.input || !graphProto.initializer) return;

    const initializerNames = new Set(graphProto.initializer.map((init) => init.name));
    graphProto.input = graphProto.input.filter((input) => !initializerNames.has(input.name));
}

export function freezeOverridableInputs(data: RawOnnxModel): void {
    if (!data.graph) return;
    freezeGraphOverridableInputs(data.graph);
}

function adaptPad(
    node: RawOnnxNode,
    graph: RawOnnxGraph,
    nextName: (prefix: string) => string,
): void {
    if (getOpType(node) !== "Pad") return;
    moveAttributeToInput(node, graph, "pads", 1, "ints", nextName);
    moveAttributeToInput(node, graph, "value", 2, "float", nextName);
}

function adaptClip(
    node: RawOnnxNode,
    graph: RawOnnxGraph,
    nextName: (prefix: string) => string,
): void {
    if (getOpType(node) !== "Clip") return;
    moveAttributeToInput(node, graph, "min", 1, "float", nextName);
    moveAttributeToInput(node, graph, "max", 2, "float", nextName);
}

function adaptSqueezeUnsqueeze(
    node: RawOnnxNode,
    graph: RawOnnxGraph,
    nextName: (prefix: string) => string,
): void {
    const opType = getOpType(node);
    if (opType !== "Squeeze" && opType !== "Unsqueeze") return;
    // Opset 13: axes moved from attribute to input[1]
    moveAttributeToInput(node, graph, "axes", 1, "ints", nextName);
}

function adaptSlice(
    node: RawOnnxNode,
    graph: RawOnnxGraph,
    nextName: (prefix: string) => string,
): void {
    if (getOpType(node) !== "Slice") return;
    // Opset 10: starts/ends/axes/steps moved to inputs
    moveAttributeToInput(node, graph, "starts", 1, "ints", nextName);
    moveAttributeToInput(node, graph, "ends", 2, "ints", nextName);
    moveAttributeToInput(node, graph, "axes", 3, "ints", nextName);
    moveAttributeToInput(node, graph, "steps", 4, "ints", nextName);
}

/**
 * Upgrades "Reshape" (Opset < 5 used 'shape' attribute)
 */
function adaptReshape(
    node: RawOnnxNode,
    graph: RawOnnxGraph,
    nextName: (prefix: string) => string,
): void {
    if (getOpType(node) !== "Reshape") return;
    moveAttributeToInput(node, graph, "shape", 1, "ints", nextName);
}

/**
 * Upgrades "Split" (Opset < 13 used 'split' attribute)
 */
function adaptSplit(
    node: RawOnnxNode,
    graph: RawOnnxGraph,
    nextName: (prefix: string) => string,
): void {
    if (getOpType(node) !== "Split") return;
    moveAttributeToInput(node, graph, "split", 1, "ints", nextName);
}

/**
 * Upgrades "Dropout" (Opset < 12 used 'ratio' attribute and Opset < 7 had 'is_test')
 */
function adaptDropout(
    node: RawOnnxNode,
    graph: RawOnnxGraph,
    nextName: (prefix: string) => string,
): void {
    if (getOpType(node) !== "Dropout") return;
    moveAttributeToInput(node, graph, "ratio", 1, "float", nextName);
    if (node.attribute) {
        const isTestIdx = node.attribute.findIndex((a) => a.name === "is_test");
        if (isTestIdx !== -1) {
            node.attribute.splice(isTestIdx, 1);
        }
    }
}

/**
 * Upgrades "BatchNormalization" (Opset < 9 used 'spatial' attribute)
 */
function adaptBatchNormalization(node: RawOnnxNode): void {
    if (getOpType(node) !== "BatchNormalization" || !node.attribute) return;
    const idx = node.attribute.findIndex((a) => a.name === "spatial");
    if (idx !== -1) {
        node.attribute.splice(idx, 1);
    }
}

/**
 * Upgrades "Upsample" (Opset 7/9) and pre-11 "Resize" to Opset 11+ "Resize" standards
 * where input layout is [X, roi, scales, sizes].
 */
function adaptResize(
    node: RawOnnxNode,
    graph: RawOnnxGraph,
    sourceOpset: number | undefined,
    nextName: (prefix: string) => string,
): void {
    const origOpType = getOpType(node);
    const isUpsample = origOpType === "Upsample";
    const isLegacyResize = origOpType === "Resize" && sourceOpset !== undefined && sourceOpset < 11;
    if (!isUpsample && origOpType !== "Resize") return;

    if (isUpsample) {
        node.opType = "Resize";
        if ("op_type" in node) delete node.op_type;
    }

    const scalesAttrIdx = node.attribute?.findIndex((a) => a.name === "scales") ?? -1;
    if (scalesAttrIdx !== -1 && node.attribute) {
        const attr = node.attribute[scalesAttrIdx];
        const vals =
            attr.floats && attr.floats.length > 0
                ? attr.floats
                : attr.f !== undefined
                  ? [Number(attr.f)]
                  : [];
        const name = nextName(`${node.name ?? "Resize"}_scales`);
        const init: TensorProto = {
            name,
            dataType: DataType.FLOAT,
            dims: [vals.length],
            floatData: vals,
        };

        graph.initializer ??= [];
        graph.initializer.push(init);

        node.input ??= [];
        while (node.input.length < 3) node.input.push("");
        node.input[2] = name;
        node.attribute.splice(scalesAttrIdx, 1);
    } else if ((isUpsample || isLegacyResize) && node.input && node.input.length === 2) {
        // Opset 9 Upsample and Opset 10 Resize passed scales at input[1] with no roi input.
        const scalesInput = node.input[1];
        if (scalesInput && scalesInput !== "") {
            node.input[1] = "";
            node.input[2] = scalesInput;
        }
    }

    if (isUpsample || isLegacyResize) {
        node.attribute ??= [];
        const hasCoordMode = node.attribute.some(
            (a) => a.name === "coordinate_transformation_mode",
        );
        if (!hasCoordMode) {
            node.attribute.push({
                name: "coordinate_transformation_mode",
                type: AttributeType.STRING,
                s: "asymmetric",
            });
        }
    }
}

const REDUCTION_OPS = new Set([
    "ReduceSum",
    "ReduceMean",
    "ReduceMax",
    "ReduceMin",
    "ReduceProd",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceSumSquare",
    "ReduceL1",
    "ReduceL2",
]);

function adaptReductions(
    node: RawOnnxNode,
    graph: RawOnnxGraph,
    sourceOpset: number | undefined,
    nextName: (prefix: string) => string,
): void {
    const opType = getOpType(node);
    if (opType === undefined || !REDUCTION_OPS.has(opType)) return;

    // ReduceSum moved axes to input[1] at Opset 13; all other Reduce* moved at Opset 18.
    const cutoff = opType === "ReduceSum" ? 13 : 18;
    if (sourceOpset === undefined || sourceOpset < cutoff) {
        moveAttributeToInput(node, graph, "axes", 1, "ints", nextName);
    }
}

function adaptTopK(
    node: RawOnnxNode,
    graph: RawOnnxGraph,
    nextName: (prefix: string) => string,
): void {
    if (getOpType(node) !== "TopK") return;
    // Opset 10: k moved from attribute to input[1]
    moveAttributeToInput(node, graph, "k", 1, "int", nextName);
}

const PRE7_BROADCAST_OPS = new Set([
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "And",
    "Or",
    "Xor",
    "Equal",
    "Greater",
    "Less",
    "Min",
    "Max",
    "PRelu",
]);

/**
 * Upgrades pre-Opset 7 binary broadcast (`broadcast=1`, optional `axis`) to multidirectional
 * numpy broadcasting by explicitly trailing-unsqueezing input[1] when `axis` aligns it to
 * a non-trailing prefix of input[0].
 */
function adaptLegacyBroadcast(
    node: RawOnnxNode,
    graph: RawOnnxGraph,
    sourceOpset: number | undefined,
    nextName: (prefix: string) => string,
): RawOnnxNode[] | undefined {
    const opType = getOpType(node);
    if (opType === undefined || !PRE7_BROADCAST_OPS.has(opType)) return undefined;
    if (sourceOpset !== undefined && sourceOpset >= 7) return undefined;
    if (!node.attribute || node.attribute.length === 0) return undefined;

    const bIdx = node.attribute.findIndex((a) => a.name === "broadcast");
    if (bIdx === -1) return undefined;

    const isLegacyBroadcast = Number(node.attribute[bIdx].i ?? 0) === 1;
    node.attribute.splice(bIdx, 1);

    const aIdx = node.attribute.findIndex((a) => a.name === "axis");
    let axisVal: number | undefined;
    if (aIdx !== -1) {
        axisVal = Number(node.attribute[aIdx].i ?? 0);
        node.attribute.splice(aIdx, 1);
    }

    if (!isLegacyBroadcast || axisVal === undefined || !node.input || node.input.length < 2) {
        return undefined;
    }

    const rankA = resolveTensorRank(graph, node.input[0]);
    const rankB = resolveTensorRank(graph, node.input[1]);
    if (rankA === undefined || rankB === undefined) {
        return undefined;
    }

    const normalizedAxis = axisVal < 0 ? rankA + axisVal : axisVal;
    const trailingOnes = rankA - (normalizedAxis + rankB);
    if (trailingOnes <= 0) {
        return undefined;
    }

    const unsqueezeAxes: number[] = [];
    for (let k = 0; k < trailingOnes; k++) {
        unsqueezeAxes.push(rankB + k);
    }

    const axesName = nextName(`${node.name ?? opType}_bcast_axes`);
    graph.initializer ??= [];
    graph.initializer.push(createInt64Initializer(axesName, unsqueezeAxes));

    const unsqueezedName = nextName(`${node.input[1]}_bcast`);
    const unsqueezeNode: RawOnnxNode = {
        name: nextName(`${node.name ?? opType}_Unsqueeze`),
        opType: "Unsqueeze",
        input: [node.input[1], axesName],
        output: [unsqueezedName],
    };

    node.input[1] = unsqueezedName;
    return [unsqueezeNode, node];
}

const SOFTMAX_FAMILY = new Set(["Softmax", "LogSoftmax", "Hardmax"]);

/**
 * Upgrades Opset < 13 Softmax / LogSoftmax / Hardmax.
 *
 * Pre-13 normalized across flattened `[0..axis-1] x [axis..rank-1]` with default `axis=1`,
 * whereas Opset 13+ normalizes strictly along single dimension `axis` with default `axis=-1`.
 */
function adaptLegacySoftmax(
    node: RawOnnxNode,
    graph: RawOnnxGraph,
    sourceOpset: number | undefined,
    nextName: (prefix: string) => string,
): RawOnnxNode[] | undefined {
    const opType = getOpType(node);
    if (opType === undefined || !SOFTMAX_FAMILY.has(opType)) return undefined;
    if (sourceOpset === undefined || sourceOpset >= 13) return undefined;
    if (
        !node.input ||
        node.input.length === 0 ||
        node.input[0] === "" ||
        !node.output ||
        node.output.length === 0 ||
        node.output[0] === ""
    ) {
        return undefined;
    }

    node.attribute ??= [];
    const axisAttr = node.attribute.find((a) => a.name === "axis");
    const legacyAxis = axisAttr !== undefined && axisAttr.i !== undefined ? Number(axisAttr.i) : 1;

    const inputName = node.input[0];
    const outputName = node.output[0];
    const rank = resolveTensorRank(graph, inputName);

    if (rank !== undefined && rank >= 1) {
        const normalizedAxis = legacyAxis < 0 ? rank + legacyAxis : legacyAxis;
        if (normalizedAxis === rank - 1) {
            if (axisAttr) {
                axisAttr.i = -1;
            } else {
                node.attribute.push({ name: "axis", type: AttributeType.INT, i: -1 });
            }
            return undefined;
        }
    }

    const base = node.name ?? opType;
    const shapeOut = nextName(`${base}_orig_shape`);
    const flatOut = nextName(`${base}_flat`);
    const smOut = nextName(`${base}_sm2d`);

    const shapeNode: RawOnnxNode = {
        name: nextName(`${base}_Shape`),
        opType: "Shape",
        input: [inputName],
        output: [shapeOut],
    };
    const flattenNode: RawOnnxNode = {
        name: nextName(`${base}_Flatten`),
        opType: "Flatten",
        input: [inputName],
        output: [flatOut],
        attribute: [{ name: "axis", type: AttributeType.INT, i: legacyAxis }],
    };
    const sm2dNode: RawOnnxNode = {
        name: node.name ?? nextName(`${base}_sm2d`),
        opType,
        input: [flatOut],
        output: [smOut],
        attribute: [{ name: "axis", type: AttributeType.INT, i: 1 }],
    };
    const reshapeNode: RawOnnxNode = {
        name: nextName(`${base}_Reshape`),
        opType: "Reshape",
        input: [smOut, shapeOut],
        output: [outputName],
    };

    return [shapeNode, flattenNode, sm2dNode, reshapeNode];
}

function adaptGraph(
    graph: RawOnnxGraph,
    sourceOpset: number | undefined,
    _targetOpset: number,
): void {
    freezeGraphOverridableInputs(graph);
    if (!graph.node || graph.node.length === 0) return;

    const nextName = createNameAllocator(graph);
    const updatedNodes: RawOnnxNode[] = [];

    for (const node of graph.node) {
        if (node.attribute) {
            for (const attr of node.attribute) {
                if (attr.g) {
                    adaptGraph(attr.g, sourceOpset, _targetOpset);
                }
                if (attr.graphs) {
                    for (const subGraph of attr.graphs) {
                        adaptGraph(subGraph, sourceOpset, _targetOpset);
                    }
                }
            }
        }

        adaptPad(node, graph, nextName);
        adaptClip(node, graph, nextName);
        adaptSlice(node, graph, nextName);
        adaptSqueezeUnsqueeze(node, graph, nextName);
        adaptReshape(node, graph, nextName);
        adaptSplit(node, graph, nextName);
        adaptDropout(node, graph, nextName);
        adaptBatchNormalization(node);
        adaptResize(node, graph, sourceOpset, nextName);
        adaptReductions(node, graph, sourceOpset, nextName);
        adaptTopK(node, graph, nextName);

        const bcastExpanded = adaptLegacyBroadcast(node, graph, sourceOpset, nextName);
        if (bcastExpanded) {
            updatedNodes.push(...bcastExpanded);
            continue;
        }

        const softmaxExpanded = adaptLegacySoftmax(node, graph, sourceOpset, nextName);
        if (softmaxExpanded) {
            updatedNodes.push(...softmaxExpanded);
            continue;
        }

        updatedNodes.push(node);
    }

    graph.node = updatedNodes;
}

// --- Main Entry Point ---
export function applyAdapters(data: RawOnnxModel, targetOpset: number = LATEST_ONNX_OPSET): void {
    if (!data.graph) return;
    const sourceOpset = resolveSourceOpset(data);
    adaptGraph(data.graph, sourceOpset, targetOpset);
}
