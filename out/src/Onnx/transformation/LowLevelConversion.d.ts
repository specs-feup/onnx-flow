import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../OnnxGraph.js";
export default class OnnxGraphTransformer implements Graph.Transformation<OnnxGraph.Class, OnnxGraph.Class> {
    apply(graph: OnnxGraph.Class): OnnxGraph.Class;
}
//# sourceMappingURL=LowLevelConversion.d.ts.map