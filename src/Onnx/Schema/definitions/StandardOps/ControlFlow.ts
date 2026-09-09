import { AttributeType, DataType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import type { OpSchema, TensorInfo } from "../../OpSchema.js";
import { OpCategory, T_ANY } from "../../OpSchema.js";
import { propagateToRegion } from "@specs-feup/onnx-flow/Onnx/InferShapes";
import { UNKNOWN_SHAPE } from "@specs-feup/onnx-flow/Onnx/Utils";

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

    inferShape: (inputs, attrs, node, graph, inferSubgraphs) => {
        const body = node!.regions[0];
        propagateToRegion(graph!, body);

        const bodyInputs = body.getInputTensorNodes().toArray();

        // Push outer resolved info into the body's boundary tensors
        for (let i = 0; i < inputs.length - 2; i++) {
            const vInitInfo = inputs[i + 2];
            const vBody = bodyInputs[i + 2];
            vBody.setShape(vInitInfo.shape);
            vBody.setLiteralType(vInitInfo.dtype);
        }

        // Trigger inner shape inference using the injected callback
        inferSubgraphs!(body);

        const bodyOutputs = body.getOutputTensorNodes().toArray();
        const results: TensorInfo[] = [];

        for (let i = 0; i < bodyOutputs.length - 1; i++) {
            const bOut = bodyOutputs[i + 1];
            if (i < inputs.length - 2) {
                results.push({ shape: bOut.shape, dtype: bOut.literalType });
            } else {
                const tripCnt = inputs[0]?.constantValue?.[0] ?? UNKNOWN_SHAPE[0];
                results.push({ shape: [tripCnt, ...bOut.shape], dtype: bOut.literalType });
            }
        }
        return results;
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

    inferShape: (inputs, attrs, node, graph, inferSubgraphs) => {
        for (const region of node!.regions) {
            propagateToRegion(graph!, region);
            inferSubgraphs!(region);
        }

        const thenGraph = node!.regions[0];
        const thenOutputs = thenGraph.getOutputTensorNodes().toArray();

        // Output shapes match the 'then' branch
        return thenOutputs.map((out) => ({ shape: out.shape, dtype: out.literalType }));
    },
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
