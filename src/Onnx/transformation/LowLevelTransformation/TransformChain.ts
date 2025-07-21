/**********************************************************************
 * Graph-wide transformation – replace every tree of supported
 * ops with a single Loop if fusion is enabled, or one per op otherwise.
 *********************************************************************/
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../OnnxGraph.js";
import OperationNode from "../../OperationNode.js";
import { buildLoopForChain } from "./BuildLoop.js";
import TensorNode from "../../TensorNode.js";

const SUP = new Set(["Add", "Sub", "Mul", "Div", "MatMul"]);

function isSupportedNonScalarOp(op: OperationNode.Class): boolean {
  if (!SUP.has(op.type)) {
    console.log(`[${op.id}] ❌ Not in SUP`);
    return false;
  }

  const incs = op.getIncomers ?? [];

  // 1. First check edge shapes directly
  const edgeHasShape = incs.some(edge => edge.shape && edge.shape.length >= 1);
  if (edgeHasShape) {
    console.log(`[${op.id}] ✅ Edge has shape`);
    return true;
  }

  
  // 2. Check tensor input shapes
  const tensorInputs = op.getInputs()
    ?.filter(n => n.is(TensorNode))
    .map(n => n.as(TensorNode)) ?? [];

  
  // Reject ops whose input is a [1] tensor coming from a Gather
  for (const t of tensorInputs) {
    if (t.shape.length === 1) {
      const producer = t.getIncomers?.[0]?.source;
      if (producer?.is(OperationNode) && producer.as(OperationNode).type === "Gather") {
        console.log(`[${op.id}] ❌ Skipping due to [1] input from Gather (${producer.id})`);
        return false;
      }
    }
  }
  /*
  const inputHasShape = tensorInputs.some(t => t.shape.length >= 1);
  if (inputHasShape) {
    console.log(`[${op.id}] ✅ Tensor input has shape`, tensorInputs[0].shape);
    return true;
  }
  */

  // 3. Recursively check intermediates' producers
  for (const t of tensorInputs) {
    if (t.type !== "intermediate") continue;
    const interIncs = t.getIncomers ?? [];
    for (const edge of interIncs) {
      if (edge.shape && edge.shape.length >= 1) {
        console.log(`[${op.id}] ✅ Found shape in intermediate edge from ${edge.source.id}`, edge.shape);
        return true;
      }
      const prod = edge.source;
      if (prod.is(OperationNode)) {
        const outEdges = prod.getOutgoers ?? [];
        for (const outEdge of outEdges) {
          if (outEdge.shape && outEdge.shape.length >= 1) {
            console.log(`[${op.id}] ✅ Found shape in producer ${prod.id}'s output`);
            return true;
          }
        }
      }
    }
  }

  // Optional debug
  /*
  console.log([${op.id}] ❌ No shape info found);
  console.log("  Incoming edge count:", incs.length);
  console.log("  Tensor input ids:", tensorInputs.map(t => t.id).join(", ") || "none");
  tensorInputs.forEach(t => {
    console.log(  Tensor ${t.id} shape:, t.shape, "type:", t.type);
    const incs = t.getIncomers ?? [];
    incs.forEach(e => console.log(    → from ${e.source.id} with shape, e.shape));
  });
  */

  console.log(`[${op.id}] ❌ No shape found`);
  return false;
}

export default class TransformChain implements Graph.Transformation<OnnxGraph.Class, OnnxGraph.Class> {
  constructor(private fuse: boolean = true, private recurse: boolean = true) {}

  apply(g: OnnxGraph.Class): OnnxGraph.Class {

    if (!this.fuse) {
      const supported = new Set<string>();
      g.getOperationNodes().forEach(op => {
        if(isSupportedNonScalarOp(op)) supported.add(op.id);
      });
      // Fusion disabled: decompose one op at a time
      g.getOperationNodes().forEach(op => {
        if (!supported.has(op.id)) return;
        buildLoopForChain([op], g, this.fuse, this.recurse);
      });
      return g;
    }

    // Fusion enabled: collect and fuse full chains
    const chains = new Map<OperationNode.Class, OperationNode.Class[]>();

    function collectChain(op: OperationNode.Class, visited = new Set<OperationNode.Class>()): OperationNode.Class[] {
      if (!isSupportedNonScalarOp(op) || visited.has(op)) return [];
      visited.add(op);

      const chain = [op];

      op.getInputs()?.forEach(inp => {
        if (inp.is(TensorNode)) {
          const t = inp.as(TensorNode);
          if (["constant", "input", "initializer", "index", "index_aux"].includes(t.type)) return;

          if (t.getIncomers.length === 0) return;
          const producer = t.getIncomers[0].source;
          if (!producer.is(OperationNode)) return;

          const subchain = collectChain(producer.as(OperationNode), visited);
          chain.push(...subchain);
        } else if (inp.is(OperationNode)) {
          const subchain = collectChain(inp.as(OperationNode), visited);
          chain.push(...subchain);
        }
      });

      return chain;
    }

    // Step 1: build candidate chains
    g.getOperationNodes().forEach(op => {
      if (!chains.has(op)) {
        const visited = new Set<OperationNode.Class>();
        const chain = collectChain(op, visited);
        if (chain.length > 0) {
          const unique = Array.from(new Set(chain));
          chains.set(op, unique);
        }
      }
    });

    // Step 2: remove chains whose root is included in another chain
    const innerNodes = new Set<string>();
    chains.forEach((chainOps, key) => {
      chainOps.forEach(op => {
        if (op !== key) innerNodes.add(op.id);
      });
    });
    for (const key of [...chains.keys()]) {
      if (innerNodes.has(key.id)) {
        chains.delete(key);
      }
    }

    // Optional debug output
    /*
    console.log("CHAINS");
    chains.forEach((chain, key) => {
      console.log(key.id, chain.map(op => op.id).join(", "));
    });
    */

    // Step 3: Apply transformation
    for (const chain of chains.values()) {
      const chainOps = chain.reverse(); // reverse for safe ordering
      buildLoopForChain(chainOps, g, this.fuse, this.recurse);
    }

    return g;
  }
}
