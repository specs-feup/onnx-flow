import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../OnnxGraph.js";
import TransformChain from "./TransformChain.js";

export default class OnnxGraphTransformer
  implements Graph.Transformation<OnnxGraph.Class,OnnxGraph.Class>
{
  constructor(
    private fuse: boolean = true,
    private recurse: boolean = false,
    private coalesce: boolean = true,
    private decomposeForCgra: boolean = false
  ) {}

  apply(graph: OnnxGraph.Class): OnnxGraph.Class {
    return new TransformChain(this.fuse, this.recurse, this.coalesce, this.decomposeForCgra).apply(graph);
  }
}