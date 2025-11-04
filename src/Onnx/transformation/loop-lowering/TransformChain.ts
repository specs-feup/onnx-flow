/**********************************************************************
 * Graph-wide transformation – replace every tree of supported
 * ops with a single Loop if fusion is enabled, or one per op otherwise.
 *********************************************************************/
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../OnnxGraph.js";
import OperationNode from "../../OperationNode.js";
import { buildLoopForChain } from "./BuildLoop.js";
import TensorNode from "../../TensorNode.js";

const SUP = new Set([
  "Add","Sub","Mul","Div","MatMul","Transpose","Range",
  "Relu","Sigmoid","Tanh","Exp","Sum","Min","Max",
  "ReduceSum","ReduceMax", "ReduceMin", "ReduceProd", "ReduceMean", "ReduceSumSquare",
  "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp"
]);

function isSupportedNonScalarOp(op: OperationNode.Class): boolean {
  if (!SUP.has(op.type)) return false;
  if (op.type === "Range") return true;

  const incs = op.getIncomers ?? [];

  const edgeHasShape = incs.some(edge =>
    edge.shape && (edge.shape.length > 1 || (edge.shape.length == 1 && edge.shape[0] > 1))
  );
  if (edgeHasShape) return true;

  const tensorInputs = op.getInputs()
    ?.filter(n => n.is(TensorNode))
    .map(n => n.as(TensorNode)) ?? [];

  for (const t of tensorInputs) {
    if (t.shape.length === 1) {
      const producer = t.getIncomers?.[0]?.source;
      if (producer?.is(OperationNode) && producer.as(OperationNode).type === "Gather") {
        return false;
      }
    }
  }

  const inputHasShape = tensorInputs.some(t => t.shape.length >= 1);
  if (inputHasShape) return true;

  for (const t of tensorInputs) {
    if (t.type !== "intermediate") continue;
    const interIncs = t.getIncomers ?? [];
    for (const edge of interIncs) {
      if (edge.shape && (edge.shape.length > 1 || (edge.shape.length == 1 && edge.shape[0] > 1))) {
        return true;
      }
      const prod = edge.source;
      if (prod.is(OperationNode)) {
        const outEdges = prod.getOutgoers ?? [];
        for (const outEdge of outEdges) {
          if (outEdge.shape && (outEdge.shape.length > 1 || (outEdge.shape.length == 1 && outEdge.shape[0] > 1))) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

export default class TransformChain implements Graph.Transformation<OnnxGraph.Class, OnnxGraph.Class> {
  constructor(
    private fuse: boolean = true,
    private recurse: boolean = true,
    private coalesce: boolean = true
  ) {}

  apply(g: OnnxGraph.Class): OnnxGraph.Class {
    if (!this.fuse) {
      const supported = new Set<string>();
      g.getOperationNodes().forEach(op => {
        if (isSupportedNonScalarOp(op)) supported.add(op.id);
      });
      g.getOperationNodes().forEach(op => {
        if (!supported.has(op.id)) return;
        buildLoopForChain([op], g, this.fuse, this.recurse, this.coalesce);
      });
      return g;
    }

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

    const innerNodes = new Set<string>();
    chains.forEach((chainOps, key) => {
      chainOps.forEach(op => { if (op !== key) innerNodes.add(op.id); });
    });
    for (const key of [...chains.keys()]) {
      if (innerNodes.has(key.id)) chains.delete(key);
    }

    for (const chain of chains.values()) {
      const chainOps = chain.reverse();

      if (this.coalesce) {
        const matmuls = chainOps.filter(op => op.type === "MatMul");
        const hasMultipleMatMuls = matmuls.length > 1;

        const mm = matmuls[0];
        let hasPreOpsForMatMul = false;
        if (mm) {
          const matmulAncestors = new Set<string>();
          const idToOp = new Map(chainOps.map(op => [op.id, op]));
          const stack = [mm];

          while (stack.length) {
            const cur = stack.pop()!;
            cur.getInputs()?.forEach(inp => {
              if (inp.is(OperationNode)) {
                const prod = inp.as(OperationNode);
                if (idToOp.has(prod.id) && prod.id !== mm.id && !matmulAncestors.has(prod.id)) {
                  matmulAncestors.add(prod.id);
                  stack.push(prod);
                }
              } else if (inp.is(TensorNode)) {
                const t = inp.as(TensorNode);
                const e = t.getIncomers?.[0];
                if (e?.source?.is(OperationNode)) {
                  const prod = e.source.as(OperationNode);
                  if (idToOp.has(prod.id) && prod.id !== mm.id && !matmulAncestors.has(prod.id)) {
                    matmulAncestors.add(prod.id);
                    stack.push(prod);
                  }
                }
              }
            });
          }

          hasPreOpsForMatMul = matmulAncestors.size > 0;
        }

        if (hasMultipleMatMuls || hasPreOpsForMatMul) {
          for (const op of chainOps) {
            buildLoopForChain([op], g, /*fuse=*/false, this.recurse, this.coalesce);
          }
          continue;
        }
      }

      if (this.coalesce) {
        const matmuls = chainOps.filter(op => op.type === "MatMul");
        const mm = matmuls[0];
        if (mm) {
          const mmIdx = chainOps.indexOf(mm);
          const afterMM = chainOps.slice(mmIdx + 1);
          const hasLayoutChangingAfter = afterMM.some(op => op.type === "Transpose");
          if (hasLayoutChangingAfter) {
            for (const op of chainOps) {
              buildLoopForChain([op], g, /*fuse=*/false, this.recurse, this.coalesce);
            }
            continue;
          }
        }
      }

      buildLoopForChain(chainOps, g, this.fuse, this.recurse, this.coalesce);
    }

    return g;
  }
}
