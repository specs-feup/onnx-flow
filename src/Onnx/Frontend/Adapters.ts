import type { TensorProto, RawOnnxModel, RawOnnxGraph, RawOnnxNode } from "../OnnxTypes.js";
import { DataType } from "../OnnxTypes.js";

// --- Helpers ---
function createInt64Initializer(name: string, values: (number | string | bigint)[]): TensorProto {
    return {
        name: name,
        dataType: DataType.INT64,
        dims: [values.length],
        int64Data: values.map((v) => BigInt(v)),
    };
}

function createFloatInitializer(name: string, value: number): TensorProto {
    return {
        name: name,
        dataType: DataType.FLOAT,
        dims: [],
        floatData: [value],
    };
}

function moveAttributeToInput(
    node: RawOnnxNode,
    graphProto: RawOnnxGraph,
    attrName: string,
    inputIndex: number,
    type: "int" | "float" | "ints" = "ints",
) {
    if (!node.attribute) return;

    const idx = node.attribute.findIndex((a) => a.name === attrName);
    if (idx !== -1) {
        const attr = node.attribute[idx];
        let init: TensorProto | undefined;

        // Safely get opType (handling snake_case fallback)
        const opType = node.opType ?? node.op_type ?? "UnknownOp";
        const name = `${node.name ?? opType}_${attrName}_${Math.random().toString(36).substr(2, 5)}`;

        if (type === "ints") {
            const val = attr.ints || [];
            init = createInt64Initializer(name, val);
        } else if (type === "int") {
            const val = [Number(attr.i ?? 0)];
            init = createInt64Initializer(name, val);
        } else if (type === "float") {
            const val = Number(attr.f ?? 0);
            init = createFloatInitializer(name, val);
        }

        if (init) {
            graphProto.initializer = graphProto.initializer || [];
            graphProto.initializer.push(init);

            // Ensure inputs array is initialized and large enough
            node.input = node.input || [];
            while (node.input.length < inputIndex) node.input.push("");
            node.input[inputIndex] = name;
        }

        // Remove the attribute
        node.attribute.splice(idx, 1);
    }
}

// --- Adapters ---
/**
 * Adapter: FreezeOverridableInputs
 * * Problem: ONNX allows "Overridable Initializers" (nodes present in both 'initializer' and 'input').
 * We strictly split Data (ConstantNode) vs Flow (TensorNode).
 * A node cannot be both.
 * * Fix: If a node is in 'initializer', we remove it from 'input'.
 * This treats the value as a compile-time ConstantNode, preventing initGraph from
 * overwriting it with an empty TensorNode.
 */
export function freezeOverridableInputs(data: RawOnnxModel): void {
    if (!data?.graph?.input || !data?.graph?.initializer) return;

    const initializerNames = new Set(data.graph.initializer.map((init) => init.name));

    // Filter out inputs that are actually initializers
    data.graph.input = data.graph.input.filter((input) => !initializerNames.has(input.name));
}

function adaptPad(node: RawOnnxNode, graph: RawOnnxGraph) {
    const opType = node.opType ?? node.op_type;
    if (opType !== "Pad") return;
    moveAttributeToInput(node, graph, "pads", 1, "ints");
    moveAttributeToInput(node, graph, "value", 2, "float");
}

function adaptClip(node: RawOnnxNode, graph: RawOnnxGraph) {
    const opType = node.opType ?? node.op_type;
    if (opType !== "Clip") return;
    moveAttributeToInput(node, graph, "min", 1, "float");
    moveAttributeToInput(node, graph, "max", 2, "float");
}

function adaptSqueezeUnsqueeze(node: RawOnnxNode, graph: RawOnnxGraph) {
    const opType = node.opType ?? node.op_type;
    if (opType !== "Squeeze" && opType !== "Unsqueeze") return;
    // Opset 13: axes moved from attribute to input[1]
    moveAttributeToInput(node, graph, "axes", 1, "ints");
}

function adaptSlice(node: RawOnnxNode, graph: RawOnnxGraph) {
    const opType = node.opType ?? node.op_type;
    if (opType !== "Slice") return;
    // Opset 10: starts/ends/axes/steps moved to inputs
    moveAttributeToInput(node, graph, "starts", 1, "ints");
    moveAttributeToInput(node, graph, "ends", 2, "ints");
    moveAttributeToInput(node, graph, "axes", 3, "ints");
    moveAttributeToInput(node, graph, "steps", 4, "ints");
}

/**
 * Upgrades "Reshape" (Opset < 5 used 'shape' attribute)
 */
function adaptReshape(node: RawOnnxNode, graph: RawOnnxGraph) {
    const opType = node.opType ?? node.op_type;
    if (opType !== "Reshape") return;
    // Opset 5: shape moved from attribute to input[1]
    moveAttributeToInput(node, graph, "shape", 1, "ints");
}

/**
 * Upgrades "Split" (Opset < 13 used 'split' attribute)
 */
function adaptSplit(node: RawOnnxNode, graph: RawOnnxGraph) {
    const opType = node.opType ?? node.op_type;
    if (opType !== "Split") return;
    // Opset 13: split lengths moved from attribute to input[1]
    moveAttributeToInput(node, graph, "split", 1, "ints");
}

/**
 * Upgrades "BatchNormalization" (Opset < 9 used 'spatial' attribute)
 */
function adaptBatchNormalization(node: RawOnnxNode, _graph: RawOnnxGraph) {
    const opType = node.opType ?? node.op_type;
    if (opType !== "BatchNormalization" || !node.attribute) return;

    // Opset 9: 'spatial' attribute removed (it's ignored/implied now).
    const idx = node.attribute.findIndex((a) => a.name === "spatial");
    if (idx !== -1) {
        node.attribute.splice(idx, 1);
    }
}

/**
 * Upgrades "Upsample" (Opset 7) to "Resize" standards
 */
function adaptResize(node: RawOnnxNode, graph: RawOnnxGraph) {
    const opType = node.opType ?? node.op_type;
    if (opType !== "Upsample" && opType !== "Resize") return;

    // Fast path: if the attribute is a single float
    moveAttributeToInput(node, graph, "scales", 1, "float");

    // Fallback path: if the attribute was a float array (floats)
    if (!node.attribute) return;
    const idx = node.attribute.findIndex((a) => a.name === "scales");
    if (idx !== -1) {
        const attr = node.attribute[idx];
        const vals = attr.floats || [];
        const name = `${node.name ?? "Resize"}_scales_${Math.random().toString(36).substr(2, 5)}`;

        const init: TensorProto = {
            name: name,
            dataType: DataType.FLOAT,
            dims: [vals.length],
            floatData: vals,
        };

        graph.initializer = graph.initializer || [];
        graph.initializer.push(init);

        node.input = node.input || [];
        while (node.input.length < 2) node.input.push("");
        node.input[1] = name; // input[1] is scales in older opsets (Input[2] in Opset 11+)

        node.attribute.splice(idx, 1);
    }
}

// --- Main Entry Point ---
export function applyAdapters(data: RawOnnxModel): void {
    if (!data?.graph?.node) return;
    const graph = data.graph;

    for (const node of graph.node) {
        // We call freezeOverridableInputs once per graph, not per node, to be safe.
        // It was inside the loop in the original code, but it operates on `data`,
        // so doing it on the first iteration is enough (or pulling it out of the loop).
        freezeOverridableInputs(data);

        adaptPad(node, graph);
        adaptClip(node, graph);
        adaptSlice(node, graph);
        adaptSqueezeUnsqueeze(node, graph);
        adaptReshape(node, graph);
        adaptSplit(node, graph);
        adaptBatchNormalization(node, graph);
        adaptResize(node, graph);
    }
}
