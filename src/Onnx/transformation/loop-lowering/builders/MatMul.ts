import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../OnnxGraph.js";
import TensorNode from "../../../TensorNode.js";
import OperationNode from "../../../OperationNode.js";
import { DataType } from "../../../OnnxTypes.js";
import { inferShapes } from "@specs-feup/onnx-flow/initGraph";
import {
  uniq, int64Vec, zeroTensor, bool, makeTensorConst, scalarInt64
} from "../../../Utils.js";
import {
  LoopCtx, BuildResult, LoopBuilder, unsqueezeIdx
} from "../BuildLoop.js";

// Handlers needed here
import handleElementWiseOperation from "../handlers/ElementWiseOperations.js";
import handleMatMul from "../handlers/MatMul.js";

export default class MatMulBuilder implements LoopBuilder {
  canHandle(chain: OperationNode.Class[]) {
    return chain.some(op => op.type === "MatMul");
  }

  build(
    chain: OperationNode.Class[],
    outer: OnnxGraph.Class,
    opts: { fuse: boolean; recurse: boolean; coalesce: boolean }
  ): BuildResult {
    const matmulIndex = chain.findIndex(op => op.type === "MatMul");
    const lastOp = chain.at(-1)!;
    const outTensor = lastOp.getOutgoers.targets.filterIs(TensorNode).first();
    const elemTy = outTensor.literalType === DataType.UNDEFINED
      ? lastOp.getOutgoers.first().literalType
      : outTensor.literalType;

    const mm = chain[matmulIndex];
    const lhs = mm.getInputs()![0].as(TensorNode);
    const rhs = mm.getInputs()![1].as(TensorNode);
    const M = Number(lhs.shape.at(-2)!);
    const K = Number(lhs.shape.at(-1)!);
    const N = Number(rhs.shape.at(-1)!);

    const totalIters = M * K * N;
    const carryLen = M * N;
    const finalOutShape = [M, N];

    const inputs = new Map<string, TensorNode.Class>();
    chain.forEach(op => op.getInputs()?.filter(n => n.is(TensorNode)).forEach(t => inputs.set(t.id, t.as(TensorNode))));

    const body = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
    const iter = body.addNode(uniq(body, "iter")).init(new TensorNode.Builder(DataType.INT64, [], "input")).as(TensorNode);
    body.addNode(uniq(body, "cond_in")).init(new TensorNode.Builder(DataType.BOOL, [], "input")).as(TensorNode); // wired by orchestrator

    // carry buffer flat [M*N]
    const carry = body.addNode(uniq(body, "carry"))
      .init(new TensorNode.Builder(elemTy, [carryLen], "input", zeroTensor(elemTy, [carryLen])))
      .as(TensorNode);

    const axes = makeTensorConst(body, "axes", DataType.INT64, "constant", int64Vec([0]));
    // Flat index (i,j,k) decoding + cached unsqueezed indices provided by handler
    const ctx: LoopCtx = {
      opMap: new Map(),
      iter,
      unsqIdx: null,         // provided per-path; not used for coalesced MatMul
      carry,
      axes,
      outShape: finalOutShape,
      coalesce: opts.coalesce,
      iU: null, jU: null, kU: null, flatU: null, kIdx: null, kM1: null,
      gateByK: opts.coalesce && chain.slice(matmulIndex + 1).some(op =>
        op.type === "Add" || op.type === "Sub" || op.type === "Mul" || op.type === "Div"),
      running: null,
    };

    const handlers: Record<string, (op: OperationNode.Class, g: OnnxGraph.Class, ctx: LoopCtx) => TensorNode.Class> = {
      MatMul: handleMatMul,
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
    };

    let indicesOut: TensorNode.Class | null = null;

    for (const op of chain) {
      const h = handlers[op.type];
      if (!h) throw new Error(`MatMulBuilder: unsupported op ${op.type}`);
      const out = h(op, body, ctx);
      ctx.opMap.set(op, [op, out]);
      if (op.type === "MatMul") {
        // after matmul handler runs, it should have filled flatU/iU/jU
        indicesOut = ctx.flatU ?? ctx.unsqIdx!;
      }
    }

    let lastOut = ctx.opMap.get(lastOp)![1];
    if (lastOut.shape.length === 0) {
      lastOut = unsqueezeIdx(body, lastOut, ctx.axes, "updateUnsq");
    }

    inferShapes(outer);
    inferShapes(body);

    // Loop inputs
    const trip = makeTensorConst(outer, `trip_count_${chain[0].id}`, DataType.INT64, "constant", scalarInt64(totalIters));
    const cond = makeTensorConst(outer, `cond_${chain[0].id}`, DataType.BOOL, "constant", bool(true));
    const v_initial = makeTensorConst(outer, "init_carry", elemTy, "initializer", zeroTensor(elemTy, [carryLen]));

    return {
      body,
      ctx,
      lastOut,
      indicesOut: indicesOut!, // set by handler
      elemTy,
      outShape: finalOutShape,
      inputs,
      outTensor,
      trip,
      cond,
      v_initial
    };
  }
}
