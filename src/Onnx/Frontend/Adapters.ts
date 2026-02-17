import { DataType } from "../OnnxTypes.js";

// --- Helpers ---
function createInt64Initializer(name: string, values: number[]): any {
    return {
        name: name,
        dataType: DataType.INT64,
        dims: [values.length],
        int64Data: values.map((v) => BigInt(v)),
    };
}

function createFloatInitializer(name: string, value: number): any {
    return {
        name: name,
        dataType: DataType.FLOAT,
        dims: [],
        floatData: [value],
    };
}

function moveAttributeToInput(
    node: any,
    graphProto: any,
    attrName: string,
    inputIndex: number,
    type: "int" | "float" | "ints" = "ints",
) {
    const idx = node.attribute?.findIndex((a: any) => a.name === attrName);
    if (idx !== undefined && idx !== -1) {
        const attr = node.attribute[idx];
        let val: any;
        let init: any;
        const name = `${node.name ?? node.opType}_${attrName}_${Math.random().toString(36).substr(2, 5)}`;

        if (type === "ints") {
            val = attr.ints || [];
            init = createInt64Initializer(name, val);
        } else if (type === "int") {
            val = [Number(attr.i)];
            init = createInt64Initializer(name, val);
        } else if (type === "float") {
            val = Number(attr.f);
            init = createFloatInitializer(name, val);
        }

        if (init) {
            graphProto.initializer = graphProto.initializer || [];
            graphProto.initializer.push(init);

            // Ensure inputs array is large enough
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
export function freezeOverridableInputs(data: any) {
    if (!data?.graph?.input || !data?.graph?.initializer) return;

    const initializerNames = new Set(data.graph.initializer.map((init: any) => init.name));

    // Filter out inputs that are actually initializers
    const newInputs = data.graph.input.filter((input: any) => {
        if (initializerNames.has(input.name)) {
            // console.log(`[Adapter] Freezing overridable input '${input.name}' to its constant initializer value.`);
            return false; // Remove from input list
        }
        return true; // Keep purely dynamic inputs
    });

    data.graph.input = newInputs;
}

function adaptPad(node: any, graph: any) {
    if (node.opType !== "Pad") return;
    moveAttributeToInput(node, graph, "pads", 1, "ints");
    moveAttributeToInput(node, graph, "value", 2, "float");
}

function adaptClip(node: any, graph: any) {
    if (node.opType !== "Clip") return;
    moveAttributeToInput(node, graph, "min", 1, "float");
    moveAttributeToInput(node, graph, "max", 2, "float");
}

function adaptSqueezeUnsqueeze(node: any, graph: any) {
    if (node.opType !== "Squeeze" && node.opType !== "Unsqueeze") return;
    // Opset 13: axes moved from attribute to input[1]
    moveAttributeToInput(node, graph, "axes", 1, "ints");
}

function adaptSlice(node: any, graph: any) {
    if (node.opType !== "Slice") return;
    // Opset 10: starts/ends/axes/steps moved to inputs
    moveAttributeToInput(node, graph, "starts", 1, "ints");
    moveAttributeToInput(node, graph, "ends", 2, "ints");
    moveAttributeToInput(node, graph, "axes", 3, "ints");
    moveAttributeToInput(node, graph, "steps", 4, "ints");
}

/**
 * Upgrades "Reshape" (Opset < 5 used 'shape' attribute)
 */
function adaptReshape(node: any, graph: any) {
    if (node.opType !== "Reshape") return;
    // Opset 5: shape moved from attribute to input[1]
    moveAttributeToInput(node, graph, "shape", 1, "ints");
}

/**
 * Upgrades "Split" (Opset < 13 used 'split' attribute)
 */
function adaptSplit(node: any, graph: any) {
    if (node.opType !== "Split") return;
    // Opset 13: split lengths moved from attribute to input[1]
    moveAttributeToInput(node, graph, "split", 1, "ints");
}

/**
 * Upgrades "BatchNormalization" (Opset < 9 used 'spatial' attribute)
 */
function adaptBatchNormalization(node: any, _graph: any) {
    if (node.opType !== "BatchNormalization") return;
    // Opset 9: 'spatial' attribute removed (it's ignored/implied now).
    // We just remove it so strict Schema validation doesn't fail.
    const idx = node.attribute?.findIndex((a: any) => a.name === "spatial");
    if (idx !== undefined && idx !== -1) {
        node.attribute.splice(idx, 1);
    }
}

/**
 * Upgrades "Upsample" (Opset 7) to "Resize" standards
 * Note: Real Resize (Opset 10/11/13) is complex. This handles the simple
 * case of old Upsample models using 'scales' as an attribute.
 */
function adaptResize(node: any, graph: any) {
    if (node.opType !== "Upsample" && node.opType !== "Resize") return;

    // Opset 7 Upsample used 'scales' attribute.
    // Opset 9 Upsample used 'scales' input.
    // Opset 10 Resize used 'scales' input.
    moveAttributeToInput(node, graph, "scales", 1, "float"); // usually float array, handled by helper

    const idx = node.attribute?.findIndex((a: any) => a.name === "scales");
    if (idx !== undefined && idx !== -1) {
        const attr = node.attribute[idx];
        const vals = attr.floats || [];
        const name = `${node.name ?? "Resize"}_scales_${Math.random().toString(36).substr(2, 5)}`;

        const init = {
            name: name,
            dataType: DataType.FLOAT,
            dims: [vals.length],
            floatData: vals,
        };

        graph.initializer = graph.initializer || [];
        graph.initializer.push(init);

        while (node.input.length < 2) node.input.push("");
        node.input[1] = name; // input[1] is scales in older opsets (Input[2] in Opset 11+)

        node.attribute.splice(idx, 1);
    }
}

// --- Main Entry Point ---
export function applyAdapters(data: any) {
    if (!data || !data.graph || !data.graph.node) return;
    const graph = data.graph;

    for (const node of graph.node) {
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
