/**********************************************************************
 * Graph-wide transformation – replace every linear chain of supported
 * ops with a single Loop.
 *********************************************************************/
import Graph          from "@specs-feup/flow/graph/Graph";
import OnnxGraph      from "../../OnnxGraph.js";
import OperationNode  from "../../OperationNode.js";
import { detectChain } from "./DetectChain.js";
import { buildLoopForChain } from "./BuildLoop.js";

export default class TransformChain
implements Graph.Transformation<OnnxGraph.Class, OnnxGraph.Class> {

  apply(g: OnnxGraph.Class): OnnxGraph.Class {
    const done = new Set<OperationNode.Class>();

    g.getOperationNodes().forEach(op => {
      if (done.has(op)) return;
      const chain = detectChain(op);
      if (chain.length === 0) return;
      chain.forEach(c => done.add(c));
      buildLoopForChain(chain, g);
    });

    return g;
  }
}
