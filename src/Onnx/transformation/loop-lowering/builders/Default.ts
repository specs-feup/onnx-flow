import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../OnnxGraph.js";
import TensorNode from "../../../TensorNode.js";
import OperationNode from "../../../OperationNode.js";
import OnnxEdge from "../../../OnnxEdge.js";
import { DataType } from "../../../OnnxTypes.js";
import { inferShapes } from "@specs-feup/onnx-flow/initGraph";
import {
  uniq, int64Vec, zeroTensor, bool, toStaticShape, Shape, makeTensorConst,
  scalarInt64
} from "../../../Utils.js";
import {
  LoopCtx, BuildResult, LoopBuilder, unsqueezeIdx, resolveFusedInput,
  broadcastShapes
} from "../BuildLoop.js";

// Handlers needed by the default builder only
import handleElementWiseOperation from "../handlers/ElementWiseOperations.js";
import handleTranspose from "../handlers/Transpose.js";

export default class DefaultBuilder implements LoopBuilder {
  canHandle(chain: OperationNode.Class[]) {
    // No Slice, no Range → handled here
    return !chain.some(op => op.type === "Slice" || op.type === "Range");
  }

  build(
    chain: OperationNode.Class[],
    outer: OnnxGraph.Class,
    opts: { fuse: boolean; recurse: boolean; coalesce: boolean }
  ): BuildResult {
    const lastOp = chain.at(-1)!;
    const outTensor = lastOp.getOutgoers.targets.filterIs(TensorNode).first();
    const elemTy = outTensor.literalType === DataType.UNDEFINED
      ? lastOp.getOutgoers.first().literalType
      : outTensor.literalType;

    // infer output shape statically (no Range here)
    const rawOutShape = Array.isArray(outTensor?.shape) ? outTensor.shape : [undefined];
    let staticOut = toStaticShape(rawOutShape as Shape);

    if (!staticOut || staticOut.length === 0 || staticOut.some(d => d === -1 || d === undefined)) {
      const inputShapes = [...new Map(
        chain.flatMap(op => (op.getInputs()?.filter(n => n.is(TensorNode)) ?? [])
          .map(t => [t.id, t.as(TensorNode)]))
      ).values()]
        .map(t => toStaticShape(t.shape as Shape))
        .filter(s => Array.isArray(s) && s.length > 0); // ignore scalars

      if (inputShapes.length > 0) {
        staticOut = broadcastShapes(inputShapes);
      } else {
        staticOut = []; // truly scalar as last resort
      }
    }

    const totalIters = staticOut.length <= 1 ? (staticOut[0] ?? 1) : staticOut.reduce((a, b) => a * b, 1);
    const carryLen = totalIters;

    const inputs = new Map<string, TensorNode.Class>();
    chain.forEach(op => op.getInputs()?.filter(n => n.is(TensorNode)).forEach(t => inputs.set(t.id, t.as(TensorNode))));

    // --- body graph skeleton
    const body = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
    const iter = body.addNode(uniq(body, "iter")).init(new TensorNode.Builder(DataType.INT64, [], "input")).as(TensorNode);
    const condIn = body.addNode(uniq(body, "cond_in")).init(new TensorNode.Builder(DataType.BOOL, [], "input")).as(TensorNode);
    const carry = body.addNode(uniq(body, "carry"))
      .init(new TensorNode.Builder(elemTy, [carryLen], "input", zeroTensor(elemTy, [carryLen])))
      .as(TensorNode);
    const axes = makeTensorConst(body, "axes", DataType.INT64, "constant", int64Vec([0]));

    const unsq = body.addNode(uniq(body, "unsq"))
      .init(new OperationNode.Builder("Unsqueeze", [iter, axes]))
      .as(OperationNode);
    const unsqOut = body.addNode(uniq(body, "unsq_out"))
      .init(new TensorNode.Builder(DataType.INT64, [1], "intermediate"))
      .as(TensorNode);
    body.addEdge(unsq, unsqOut).init(new OnnxEdge.Builder(unsqOut.literalType, unsqOut.shape)).as(OnnxEdge);

    const ctx: LoopCtx = {
      opMap: new Map(),
      iter,
      unsqIdx: unsqOut,
      carry,
      axes,
      outShape: rawOutShape,
      coalesce: opts.coalesce,
    };

    const handlers: Record<string, (op: OperationNode.Class, g: OnnxGraph.Class, ctx: LoopCtx) => TensorNode.Class> = {
      Add: handleElementWiseOperation,
      Sub: handleElementWiseOperation,
      Mul: handleElementWiseOperation,
      Div: handleElementWiseOperation,
      Relu: handleElementWiseOperation,
      Sigmoid: handleElementWiseOperation,
      Tanh: handleElementWiseOperation,
      Exp: handleElementWiseOperation,
      Sum: handleElementWiseOperation,
      Min: handleElementWiseOperation,
      Max: handleElementWiseOperation,
      Transpose: handleTranspose,
    };

    let indicesOut = unsqOut;
    for (const op of chain) {
      const h = handlers[op.type];
      if (!h) throw new Error(`DefaultBuilder: unsupported op ${op.type}`);
      const out = h(op, body, ctx);
      ctx.opMap.set(op, [op, out]);
    }

    let lastOut = ctx.opMap.get(lastOp)![1];
    if (lastOut.shape.length === 0) {
      lastOut = unsqueezeIdx(body, lastOut, ctx.axes, "updateUnsq");
    }

    inferShapes(outer);
    inferShapes(body);

    // Loop inputs for outer graph
    const trip = makeTensorConst(outer, `trip_count_${chain[0].id}`, DataType.INT64, "constant", scalarInt64(totalIters));
    const cond = makeTensorConst(outer, `cond_${chain[0].id}`, DataType.BOOL, "constant", bool(true));
    const v_initial = makeTensorConst(outer, "init_carry", elemTy, "initializer", zeroTensor(elemTy, [carryLen]));

    return {
      body,
      ctx,
      lastOut,
      indicesOut,
      elemTy,
      outShape: staticOut,
      inputs,
      outTensor,
      trip,
      cond,
      v_initial
    };
  }
}
