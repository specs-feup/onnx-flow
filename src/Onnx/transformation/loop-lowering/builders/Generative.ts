import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../OnnxGraph.js";
import TensorNode from "../../../TensorNode.js";
import OperationNode from "../../../OperationNode.js";
import OnnxEdge from "../../../OnnxEdge.js";
import { DataType, TensorProto } from "../../../OnnxTypes.js";
import { inferShapes } from "@specs-feup/onnx-flow/initGraph";
import {
  uniq, int64Vec, zeroTensor, bool, makeTensorConst
} from "../../../Utils.js";
import {
  LoopCtx, BuildResult, LoopBuilder, unsqueezeIdx
} from "../BuildLoop.js";

// Handlers needed here (Range + we allow trailing elementwise/transpose)
import handleRange from "../handlers/Range.js";
import handleElementWiseOperation from "../handlers/ElementWiseOperations.js";
import handleTranspose from "../handlers/Transpose.js";

export default class GenerativeBuilder implements LoopBuilder {
  canHandle(chain: OperationNode.Class[]) {
    return chain.some(op => op.type === "Range");
  }

  build(
    chain: OperationNode.Class[],
    outer: OnnxGraph.Class,
    opts: { fuse: boolean; recurse: boolean; coalesce: boolean }
  ): BuildResult {
    const lastOp = chain.at(-1)!;
    const outTensor = lastOp.getOutgoers.targets.filterIs(TensorNode).first();

    const rangeOp = chain.find(op => op.type === "Range")!;

    // Compute trip_count, cond, v_initial for Range at OUTER graph level
    const [startT, limitT, deltaT] = rangeOp.getInputs()!.map(n => n.as(TensorNode));
    const elemTy = startT.literalType;       // <- anchor carry type to Range dtype

    const outShape: (number | string)[] = [undefined];

    const inputs = new Map<string, TensorNode.Class>();
    chain.forEach(op => op.getInputs()?.filter(n => n.is(TensorNode)).forEach(t => inputs.set(t.id, t.as(TensorNode))));

    const body = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
    const iter = body.addNode(uniq(body, "iter")).init(new TensorNode.Builder(DataType.INT64, [], "input")).as(TensorNode);
    body.addNode(uniq(body, "cond_in")).init(new TensorNode.Builder(DataType.BOOL, [], "input")).as(TensorNode);

    // unknown length carry → declare [-1], *no* initializer
    const carry = body.addNode(uniq(body, "carry"))
      .init(new TensorNode.Builder(elemTy, [-1], "input"))
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
      outShape,
      coalesce: false
    };

    const handlers: Record<string, (op: OperationNode.Class, g: OnnxGraph.Class, ctx: LoopCtx) => TensorNode.Class> = {
      Range: handleRange,
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

    for (const op of chain) {
      const h = handlers[op.type];
      if (!h) throw new Error(`GenerativeBuilder: unsupported op ${op.type}`);
      const out = h(op, body, ctx);
      ctx.opMap.set(op, [op, out]);
    }

    let lastOut = ctx.opMap.get(lastOp)![1];
    if (lastOut.shape.length === 0) {
      lastOut = unsqueezeIdx(body, lastOut, ctx.axes, "updateUnsq");
    }

    // Cast the update to elemTy if needed (guards against float tails)
    if (lastOut.literalType !== elemTy) {
      const castU = body.addNode(uniq(body, `gen_cast_update_${lastOp.id}`))
        .init(new OperationNode.Builder("Cast", [lastOut], { to: elemTy }))
        .as(OperationNode);
      const castUOut = body.addNode(uniq(body, `gen_cast_update_out_${lastOp.id}`))
        .init(new TensorNode.Builder(elemTy, lastOut.shape, "intermediate"))
        .as(TensorNode);
      body.addEdge(castU, castUOut).init(new OnnxEdge.Builder(elemTy, lastOut.shape)).as(OnnxEdge);
      lastOut = castUOut;
    }

    inferShapes(outer);
    inferShapes(body);

    const subN = outer.addNode(uniq(outer, `range_sub_${chain[0].id}`))
      .init(new OperationNode.Builder("Sub", [limitT, startT])).as(OperationNode);
    const subOut = outer.addNode(uniq(outer, `range_sub_out_${chain[0].id}`))
      .init(new TensorNode.Builder(startT.literalType, [], "intermediate")).as(TensorNode);
    outer.addEdge(subN, subOut).init(new OnnxEdge.Builder(subOut.literalType, subOut.shape)).as(OnnxEdge);

    const subCastN = outer.addNode(uniq(outer, `range_subF_${chain[0].id}`))
      .init(new OperationNode.Builder("Cast", [subOut], { to: DataType.FLOAT })).as(OperationNode);
    const subF = outer.addNode(uniq(outer, `range_subF_out_${chain[0].id}`))
      .init(new TensorNode.Builder(DataType.FLOAT, [], "intermediate")).as(TensorNode);
    outer.addEdge(subCastN, subF).init(new OnnxEdge.Builder(subF.literalType, subF.shape)).as(OnnxEdge);

    const deltaCastN = outer.addNode(uniq(outer, `range_deltaF_${chain[0].id}`))
      .init(new OperationNode.Builder("Cast", [deltaT], { to: DataType.FLOAT })).as(OperationNode);
    const deltaF = outer.addNode(uniq(outer, `range_deltaF_out_${chain[0].id}`))
      .init(new TensorNode.Builder(DataType.FLOAT, [], "intermediate")).as(TensorNode);
    outer.addEdge(deltaCastN, deltaF).init(new OnnxEdge.Builder(deltaF.literalType, deltaF.shape)).as(OnnxEdge);

    const divN = outer.addNode(uniq(outer, `range_divF_${chain[0].id}`))
      .init(new OperationNode.Builder("Div", [subF, deltaF])).as(OperationNode);
    const divF = outer.addNode(uniq(outer, `range_divF_out_${chain[0].id}`))
      .init(new TensorNode.Builder(DataType.FLOAT, [], "intermediate")).as(TensorNode);
    outer.addEdge(divN, divF).init(new OnnxEdge.Builder(divF.literalType, divF.shape)).as(OnnxEdge);

    const ceilN = outer.addNode(uniq(outer, `range_ceilF_${chain[0].id}`))
      .init(new OperationNode.Builder("Ceil", [divF])).as(OperationNode);
    const ceilF = outer.addNode(uniq(outer, `range_ceilF_out_${chain[0].id}`))
      .init(new TensorNode.Builder(DataType.FLOAT, [], "intermediate")).as(TensorNode);
    outer.addEdge(ceilN, ceilF).init(new OnnxEdge.Builder(ceilF.literalType, ceilF.shape)).as(OnnxEdge);

    const zeroF = makeTensorConst(
      outer, `range_zeroF_${chain[0].id}`, DataType.FLOAT, "constant",
      { dataType: DataType.FLOAT, dims: [], floatData: [0] } as TensorProto
    );

    const maxN = outer.addNode(uniq(outer, `range_maxF_${chain[0].id}`))
      .init(new OperationNode.Builder("Max", [ceilF, zeroF])).as(OperationNode);
    const maxF = outer.addNode(uniq(outer, `range_maxF_out_${chain[0].id}`))
      .init(new TensorNode.Builder(DataType.FLOAT, [], "intermediate")).as(TensorNode);
    outer.addEdge(maxN, maxF).init(new OnnxEdge.Builder(maxF.literalType, maxF.shape)).as(OnnxEdge);

    const tripCastN = outer.addNode(uniq(outer, `range_trip_${chain[0].id}`))
      .init(new OperationNode.Builder("Cast", [maxF], { to: DataType.INT64 })).as(OperationNode);
    const tripScalar = outer.addNode(uniq(outer, `range_trip_out_${chain[0].id}`))
      .init(new TensorNode.Builder(DataType.INT64, [], "intermediate")).as(TensorNode);
    outer.addEdge(tripCastN, tripScalar).init(new OnnxEdge.Builder(tripScalar.literalType, tripScalar.shape)).as(OnnxEdge);

    const axes0 = makeTensorConst(outer, `axes0_${chain[0].id}`, DataType.INT64, "constant", int64Vec([0]));
    const tripUnsq = outer.addNode(uniq(outer, `range_trip_unsq_${chain[0].id}`))
      .init(new OperationNode.Builder("Unsqueeze", [tripScalar, axes0])).as(OperationNode);
    const tripVec = outer.addNode(uniq(outer, `range_trip_vec_${chain[0].id}`))
      .init(new TensorNode.Builder(DataType.INT64, [1], "intermediate")).as(TensorNode);
    outer.addEdge(tripUnsq, tripVec).init(new OnnxEdge.Builder(tripVec.literalType, tripVec.shape)).as(OnnxEdge);

    const init = outer.addNode(uniq(outer, `range_init_${chain[0].id}`))
      .init(new OperationNode.Builder("ConstantOfShape", [tripVec], {
        value: { type: "TENSOR", ...zeroTensor(elemTy, [1]) }
      }))
      .as(OperationNode);
    const v_initial = outer.addNode(uniq(outer, `range_init_out_${chain[0].id}`))
      .init(new TensorNode.Builder(elemTy, [undefined], "intermediate"))
      .as(TensorNode);
    outer.addEdge(init, v_initial).init(new OnnxEdge.Builder(v_initial.literalType, v_initial.shape)).as(OnnxEdge);

    const trip = tripScalar; // scalar
    const cond = makeTensorConst(outer, `cond_${chain[0].id}`, DataType.BOOL, "constant", bool(true));

    return {
      body,
      ctx,
      lastOut,
      indicesOut: unsqOut,
      elemTy,
      outShape,
      inputs,
      outTensor,
      trip,
      cond,
      v_initial
    };
  }
}
