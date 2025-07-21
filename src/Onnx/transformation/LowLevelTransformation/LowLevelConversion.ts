import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../OnnxGraph.js";
import TransformChain from "./TransformChain.js";

export default class OnnxGraphTransformer
  implements Graph.Transformation<OnnxGraph.Class,OnnxGraph.Class>
{
  apply(graph: OnnxGraph.Class): OnnxGraph.Class {
    return new TransformChain(true, false).apply(graph);
  }
}