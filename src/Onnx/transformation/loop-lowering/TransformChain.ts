/**********************************************************************
 * Graph-wide transformation – replace every tree of supported
 * ops with a single Loop if fusion is enabled, or one per op otherwise.
 *********************************************************************/
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../OnnxGraph.js";
import OperationNode from "../../OperationNode.js";
import { buildLoopForChain } from "./BuildLoop.js";
import TensorNode from "../../TensorNode.js";
import { toStaticShape } from "../../Utils.js";

function isBroadcastableTo(inDims: number[], outDims: number[]): boolean {
  const rI = inDims.length;
  const rO = outDims.length;
  const maxRank = Math.max(rI, rO);

  for (let i = 0; i < maxRank; i++) {
    const inDimRaw = inDims[rI - 1 - i] ?? 1;
    const outDimRaw = outDims[rO - 1 - i] ?? 1;

    const inDim  = inDimRaw  > 0 ? inDimRaw  : 0; // 0 = dynamic/unknown
    const outDim = outDimRaw > 0 ? outDimRaw : 0;

    // If either side is dynamic (0), assume it's okay.
    if (inDim === 0 || outDim === 0) continue;

    // ONNX broadcasting rule:
    // dimensions are compatible if they are equal, or one of them is 1.
    if (inDim !== 1 && outDim !== 1 && inDim !== outDim) {
      return false;
    }
  }
  return true;
}

function getSegmentOutShape(seg: OperationNode.Class[]): (number | String)[] | null {
  if (!seg.length) return null;

  const root = seg[seg.length - 1]; // last op = segment root
  const outT = root.getOutgoers.targets
    ?.filter((n) => n.is(TensorNode))
    .first()
    ?.as(TensorNode);

  return outT?.shape ?? null;
}

/**
 * A segment is "broadcast-safe" iff every tensor input that will be
 * scalarised inside the loop is broadcastable to the segment's final
 * output shape.
 *
 * We are intentionally conservative:
 *  - If we don't know the segment's out shape, we say "unsafe".
 *  - We skip obvious index tensors (type === "index"/"index_aux").
 *  - Scalars (shape length 0) are always considered fine.
 *
 * Unsafe segments fall back to per-op loop lowering.
 */
function isBroadcastSafeSegment(seg: OperationNode.Class[]): boolean {
  const outShape = getSegmentOutShape(seg);
  if (!outShape || !outShape.length) {
    // No reliable shape => don't risk fusion
    return false;
  }

  for (const op of seg) {
    const tensorInputs =
      op.getInputs()?.filter((n) => n.is(TensorNode)).map((n) => n.as(TensorNode)) ?? [];

    for (const t of tensorInputs) {
      // Index helpers never go through gatherWithBroadcast
      if (t.type === "index" || t.type === "index_aux") continue;

      const s = t.shape ?? [];
      if (!s.length) continue; // scalar or unknown, fine

      // If any non-scalar input cannot broadcast to the final outShape,
      // then this segment is not safe to fuse into a single loop.
      if (!isBroadcastableTo(toStaticShape(s), toStaticShape(outShape))) {
        return false;
      }
    }
  }

  return true;
}


const SUP = new Set([
  "Add","Sub","Mul","Div", "MatMul", "Range", "Transpose",
  "Relu","Sigmoid","Tanh","Exp","Sum","Min","Max",
  "ReduceSum","ReduceMax", "ReduceMin", "ReduceProd", "ReduceMean", "ReduceSumSquare",
  "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp",
  "Conv", "AveragePool"
]);

const REDUCE_SET = new Set([
  "ReduceSum","ReduceMax","ReduceMin","ReduceProd","ReduceMean",
  "ReduceSumSquare","ReduceL1","ReduceL2","ReduceLogSum","ReduceLogSumExp"
]);

function isReduce(op: OperationNode.Class) {
  return REDUCE_SET.has(op.type);
}

function isElementwiseUnary(op: OperationNode.Class) {
  return new Set(["Relu","Sigmoid","Tanh","Exp","Log","Clip"]).has(op.type);
}

function isElementwiseBinary(op: OperationNode.Class) {
  return new Set(["Add","Sub","Mul","Div","Min","Max","Sum"]).has(op.type);
}

function isAllowedNonReduce(op: OperationNode.Class) {
  // Any op your default/elemwise builders can already handle.
  // Keep this aligned with SUP (minus reduces).
  return SUP.has(op.type) && !isReduce(op);
}

function isSimpleMatMul(op: OperationNode.Class): boolean {
  if (op.type !== "MatMul") return false;

  const inputs = op.getInputs();
  if (!inputs || inputs.length < 2) return false;

  const aNode = inputs[0];
  const bNode = inputs[1];

  if (!aNode.is(TensorNode) || !bNode.is(TensorNode)) return false;

  const aShape = aNode.as(TensorNode).shape ?? [];
  const bShape = bNode.as(TensorNode).shape ?? [];

  // Bail if we have symbolic dims we don't understand
  if (aShape.some(d => typeof d === "string") ||
      bShape.some(d => typeof d === "string")) {
    return false;
  }

  const aDims = toStaticShape(aShape);
  const bDims = toStaticShape(bShape);

  // Only handle straightforward 2D MatMuls for now
  if (aDims.length !== 2 || bDims.length !== 2) return false;

  const K  = aDims[1];
  const Kb = bDims[0];

  // Require a known and matching inner dimension
  if (K <= 0 || Kb <= 0 || K !== Kb) return false;

  return true;
}

function sameOrBroadcastsTo(shapeA: (number|String) [], shapeB: (number|String)[]): boolean {
  // shapeA is the op output; shapeB is the reduced tensor shape
  // strict match or standard numpy/onnx broadcast to shapeB
  if(typeof shapeA == "string" || typeof shapeB == "string") return false;
  if (shapeA.length > shapeB.length) return false;
  // right-align and check each dim is either 1 or equal
  let i = shapeA.length - 1, j = shapeB.length - 1;
  for (; i >= 0; --i, --j) {
    const a = shapeA[i], b = shapeB[j];
    if (!(a === 1 || a === b)) return false;
  }
  // any leading dims in shapeB are fine (they’re target dims)
  return true;
}

function splitByMatMul(seg: OperationNode.Class[]): OperationNode.Class[][] {
  const out: OperationNode.Class[][] = [];
  let cur: OperationNode.Class[] = [];
  let curHasMatMul = false;

  const flush = () => {
    if (cur.length) out.push(cur);
    cur = [];
    curHasMatMul = false;
  };

  for (const op of seg) {
    if (op.type === "MatMul") {
      if (curHasMatMul) {
        // Close current group before starting a new MatMul group
        flush();
      }
      cur.push(op);
      curHasMatMul = true;
    } else {
      cur.push(op);
    }
  }
  flush();
  return out;
}

function canBeEpilogue(op: OperationNode.Class, reducedOutShape: (number|String)[]): boolean {
  if (isElementwiseUnary(op)) return true;
  if (isElementwiseBinary(op)) {
    // Check output shape is broadcast-compatible with reducedOutShape
    const outT = op.getOutgoers.targets?.filter(n => n.is(TensorNode)).first()?.as(TensorNode);
    if (!outT) return false;
    return sameOrBroadcastsTo(outT.shape, reducedOutShape);
  }
  return false;
}

function isSupportedNonScalarOp(op: OperationNode.Class): boolean {
  if (!SUP.has(op.type)) return false;
  /*
  if (op.type === "MatMul") {
    // Only decompose MatMul when we have a simple, consistent 2D case
    // like [M,K] x [K,N]. If shapes are weird or partially unknown,
    // leave it as a plain MatMul in the graph.
    if (!isSimpleMatMul(op)) return false;
  }
    */
  if (op.type === "Range" || op.type === "Conv" || op.type === "AveragePool") return true;

  const incs = op.getIncomers ?? [];

  const edgeHasShape = incs.some(edge =>
    edge.shape && (edge.shape.length > 1 || (edge.shape.length == 1 && edge.shape[0] > 1))
  );
  if (edgeHasShape) return true;

  const tensorInputs = op.getInputs()
    ?.filter(n => n.is(TensorNode))
    .map(n => n.as(TensorNode)) ?? [];

  /*
  for (const t of tensorInputs) {
    if (t.shape && t.shape.length === 1) {
      const producer = t.getIncomers?.[0]?.source;
      if (producer?.is(OperationNode) && producer.as(OperationNode).type === "Gather") {
        return false;
      }
    }
  }
  */

  const inputHasShape = tensorInputs.some(t => t.shape && t.shape.length >= 1);
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
  constructor(private fuse: boolean = true, private recurse: boolean = true, private coalesce: boolean = true) { }

  apply(g: OnnxGraph.Class): OnnxGraph.Class {
    //lowerLSTM(g);
    // Fast path: no fusion — build one Loop per supported op
    if (!this.fuse) {
      const supported = new Set<string>();
      g.getOperationNodes().forEach(op => {
        if (isSupportedNonScalarOp(op)) supported.add(op.id);
      });
      g.getOperationNodes().forEach(op => {
        if (!supported.has(op.id)) return;
        buildLoopForChain([op], g, /*fuse=*/false, this.recurse, this.coalesce);
      });
      return g;
    }

    // 1) Collect candidate chains (DAG backwalk), one chain per "root" op.
    const chains = new Map<OperationNode.Class, OperationNode.Class[]>();

    function collectChain(
      op: OperationNode.Class,
      visited = new Set<OperationNode.Class>()
    ): OperationNode.Class[] {
      if (!isSupportedNonScalarOp(op) || visited.has(op)) return [];
      visited.add(op);

      if (op.type === "Conv" || op.type === "AveragePool") {
        return [op];
      }

      const chain: OperationNode.Class[] = [op];

      op.getInputs()?.forEach(inp => {
        if (inp.is(TensorNode)) {
          const t = inp.as(TensorNode);
          // stop at non-intermediate sources
          if (["constant","input","initializer","index","index_aux"].includes(t.type)) return;
          if (t.getIncomers.length === 0) return;

          const prod = t.getIncomers[0].source;
          if (!prod.is(OperationNode)) return;
          if (prod.as(OperationNode).type === "Conv" || op.type === "AveragePool") {
            return;
          }
          chain.push(...collectChain(prod.as(OperationNode), visited));

        } else if (inp.is(OperationNode)) {
          if (inp.as(OperationNode).type === "Conv" || op.type === "AveragePool") {
            return;
          }
          chain.push(...collectChain(inp.as(OperationNode), visited));
        }
      });

      return chain;
    }

    g.getOperationNodes().forEach(op => {
      if (chains.has(op)) return;
      const visited = new Set<OperationNode.Class>();
      const ch = collectChain(op, visited);
      if (ch.length > 0) chains.set(op, Array.from(new Set(ch))); // de-dup
    });

    // Keep only roots (remove chains whose key is inside another chain)
    const innerIds = new Set<string>();
    chains.forEach((ops, key) => {
      ops.forEach(o => { if (o !== key) innerIds.add(o.id); });
    });
    for (const key of [...chains.keys()]) {
      if (innerIds.has(key.id)) chains.delete(key);
    }

    // 2) Transform each chain (topo-ish order: producers -> consumers)
    for (const chain of chains.values()) {
      const chainOps = chain.slice().reverse(); // inputs first

      // --- Reduce-aware segmentation ---
      // split on reduces; non-reduce stretches are fused normally;
      // each reduce is built as its own loop segment.
      const segments: OperationNode.Class[][] = [];
      let cur: OperationNode.Class[] = [];
      for (const node of chainOps) {
        if (isReduce(node)) {
          if (cur.length) segments.push(cur), (cur = []);
          segments.push([node]); // a single-reduce segment
        } else {
          cur.push(node);
        }
      }
      if (cur.length) segments.push(cur);

      // Optional: coalescing / special-casing MatMul layout barriers
      if (this.coalesce) {
        const matmuls = chainOps.filter(op => op.type === "MatMul");
        const mm = matmuls[0];
        if (mm) {
          const mmIdx = chainOps.indexOf(mm);
          const afterMM = chainOps.slice(mmIdx + 1);
          const hasLayoutChangeAfter = afterMM.some(op => op.type === "Transpose");
          if (hasLayoutChangeAfter) {
            // bail to per-op lowering if layout changes after MatMul
            for (const op of chainOps) {
              // Re-hydrate from the *current* graph and skip if it was already removed
              const cur = g.getNodeById(op.id);
              if (!cur || !cur.is(OperationNode)) continue;
              buildLoopForChain(
                [cur.as(OperationNode)],
                g,
                /* fuse = */ false,
                this.recurse,
                this.coalesce
              );
            }
            continue;
          }
        }
      }

      // 3) Build each segment in order (reduce-aware, plus MatMul splitting)
      for (const seg0 of segments) {
        // First, split so each subsegment has at most one MatMul
        const mmSegments = splitByMatMul(seg0);

        for (const mmSeg0 of mmSegments) {
          // Re-hydrate after prior mutations:
          const seg = mmSeg0
            .map(op => g.getNodeById(op.id))
            .filter(n => n && n.is(OperationNode))
            .map(n => n!.as(OperationNode));

          if (seg.length === 0) continue;

          const isSingleReduce = seg.length === 1 && REDUCE_SET.has(seg[0].type);

          // If this is not a singleton reduce and shapes are not
          // broadcast-safe, fall back to per-op loop lowering.
          if (!isSingleReduce && !isBroadcastSafeSegment(seg)) {
            for (const op of seg) {
              buildLoopForChain(
                [op],
                g,
                /* fuse = */ false,
                this.recurse,
                this.coalesce
              );
            }
            continue;
          }

          buildLoopForChain(
            seg,
            g,
            /* fuse = */ this.fuse && !isSingleReduce,
            this.recurse,
            this.coalesce
          );
        }
      }
    }

    return g;
  }
}
