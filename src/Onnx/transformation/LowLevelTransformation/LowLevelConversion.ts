import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../OnnxGraph.js";
import TransformChain from "./TransformChain.js";
import applyPreDecomposition from "../PreDecomposition/index.js";

export default class OnnxGraphTransformer
  implements Graph.Transformation<OnnxGraph.Class,OnnxGraph.Class>
{
  constructor(
    private fuse: boolean = true,
    private recurse: boolean = false,
    private coalesce: boolean = true
  ) {}

  apply(graph: OnnxGraph.Class): OnnxGraph.Class {
    // 1) Pre-decompose high-level ops (Slice now; Conv later)
    const pre = applyPreDecomposition(graph);

    // 2) Your existing LL path (TransformChain) -> BuildLoop, etc.
    return new TransformChain(this.fuse, this.recurse, this.coalesce).apply(pre);
  }
}