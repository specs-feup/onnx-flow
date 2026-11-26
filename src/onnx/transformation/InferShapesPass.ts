import type OnnxGraph from "../ir/OnnxGraph.js";
import type { GraphPass } from "./PassManager.js";
import inferShapes from "../schema/InferShapes.js";

export class InferShapesPass implements GraphPass {
    public readonly name = "InferShapes";

    run(graph: OnnxGraph.Class): boolean {
        // Run the existing ONNX shape inference algorithm
        inferShapes(graph);

        // We return false because shape inference only updates metadata (shapes/types).
        // It doesn't change the topology of the graph, so it shouldn't trigger another PassManager loop.
        return false;
    }
}
