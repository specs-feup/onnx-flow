import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../OnnxGraph.js";
import TransformChain from "./TransformChain.js";
import applyCanonicalization from "../canonicalization/index.js";

export default class OnnxGraphTransformer
  implements Graph.Transformation<OnnxGraph.Class,OnnxGraph.Class>
{
  constructor(
    private fuse: boolean = true,
    private recurse: boolean = false,
    private coalesce: boolean = true
  ) {}

  apply(graph: OnnxGraph.Class): OnnxGraph.Class {
    // 1) Canonical version of high-level operations (no explicit Loop needed)
    const canon = applyCanonicalization(graph);

    // 2) Loop-lowering (creating explicit loop and applying some Loop optimizations)
    return new TransformChain(this.fuse, this.recurse, this.coalesce).apply(canon);
  }
}