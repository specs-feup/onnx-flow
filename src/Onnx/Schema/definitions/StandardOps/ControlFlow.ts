import { AttributeType, DataType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import type { OpSchema } from "../../OpSchema.js";
import { OpCategory, T_ANY } from "../../OpSchema.js";

export const Loop: OpSchema = {
    opType: "Loop",
    sinceVersion: 13, // Use 13 or 16
    category: OpCategory.ControlFlow,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "trip_count", typeConstraint: "I", optional: true },
        { name: "cond", typeConstraint: "B", optional: true },
        { name: "v_initial", typeConstraint: "V", variadic: true },
    ],
    outputs: [{ name: "v_final_and_scan_outputs", typeConstraint: "V", variadic: true }],
    attributes: {
        body: { name: "body", type: AttributeType.GRAPH, required: true },
    },

    inferShape: (inputs) => {
        // Base schema fallback. Actual shape inference for Loop is intercepted
        // by InferShapes.ts because it requires recursive subgraph traversal.
        if (inputs.length < 3) {
            return [{ shape: [], dtype: DataType.UNDEFINED }];
        }
        const initState = inputs[2];
        return [
            {
                shape: initState.shape.slice(),
                dtype: initState.dtype,
            },
        ];
    },
};

export const If: OpSchema = {
    opType: "If",
    sinceVersion: 13, // Opset 13 aligns with Loop/Scan graph semantics
    category: OpCategory.ControlFlow,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "cond", typeConstraint: "B" }, // Boolean scalar
    ],
    outputs: [{ name: "outputs", typeConstraint: T_ANY, variadic: true }],
    attributes: {
        then_branch: { name: "then_branch", type: AttributeType.GRAPH, required: true },
        else_branch: { name: "else_branch", type: AttributeType.GRAPH, required: true },
    },

    // Base schema fallback. Actual shape inference for If is intercepted
    // by InferShapes.ts because it requires recursive subgraph traversal.
    inferShape: () => [{ shape: [], dtype: DataType.UNDEFINED }],
};

export const Scan: OpSchema = {
    opType: "Scan",
    sinceVersion: 9,
    category: OpCategory.ControlFlow,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "sequence_lens", typeConstraint: "I", optional: true },
        { name: "initial_state_and_inputs", typeConstraint: "V", variadic: true },
    ],
    outputs: [{ name: "final_state_and_scan_outputs", typeConstraint: "V", variadic: true }],
    attributes: {
        body: { name: "body", type: AttributeType.GRAPH, required: true },
        num_scan_inputs: { name: "num_scan_inputs", type: AttributeType.INT, required: true },
        scan_input_directions: {
            name: "scan_input_directions",
            type: AttributeType.INTS,
            required: false,
        },
        scan_output_directions: {
            name: "scan_output_directions",
            type: AttributeType.INTS,
            required: false,
        },
        scan_input_axes: { name: "scan_input_axes", type: AttributeType.INTS, required: false },
        scan_output_axes: { name: "scan_output_axes", type: AttributeType.INTS, required: false },
    },

    // Base schema fallback. Actual shape inference for Scan is intercepted
    // by InferShapes.ts because it requires recursive subgraph traversal.
    inferShape: () => [{ shape: [], dtype: DataType.UNDEFINED }],
};

export const ControlFlowOps: OpSchema[] = [Loop, If, Scan];
