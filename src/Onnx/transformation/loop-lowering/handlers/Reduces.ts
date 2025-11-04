// handlers/Reduces.ts
import OnnxGraph from "@specs-feup/onnx-flow/Onnx/OnnxGraph";
import { DataType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import OperationNode from "@specs-feup/onnx-flow/Onnx/OperationNode";
import TensorNode from "@specs-feup/onnx-flow/Onnx/TensorNode";
import OnnxEdge from "@specs-feup/onnx-flow/Onnx/OnnxEdge";
import { uniq, makeTensorConst } from "@specs-feup/onnx-flow/Onnx/Utils";
import { LoopCtx } from "../BuildLoop.js";

/**
 * Per-element reducer: returns a scalar [] equal to the bin's value to write this iteration.
 * We keep the carry in a form that:
 *  - yields correct math on the next iteration, and
 *  - equals the final reduced value on the last iteration (so the Loop’s output is correct),
 *    without changing BuildLoop.
 *
 * Ops:
 *  - ReduceSum        : acc + x
 *  - ReduceMean       : acc + x * meanScale        (meanScale = 1 / reduce_count) [from ctx]
 *  - ReduceMax        : Max(acc, x)
 *  - ReduceMin        : Min(acc, x)
 *  - ReduceProd       : Mul(acc, x)
 *  - ReduceSumSquare  : acc + x*x
 *  - ReduceL1         : acc + Abs(x)
 *  - ReduceL2         : carry stores sqrt(sum_sq). Update: sqrt(acc*acc + x*x)
 *  - ReduceLogSum     : carry stores log(sum).     Update: log( (acc==0?0:exp(acc)) + x )
 *  - ReduceLogSumExp  : carry stores log(sum_exp). Update: log( (acc==0?0:exp(acc)) + exp(x) )
 */
export default function handleReduceElem(
  op: OperationNode.Class,
  g: OnnxGraph.Class,
  ctx: LoopCtx,
  accScalar: TensorNode.Class, // []
  xScalar: TensorNode.Class,   // []
): TensorNode.Class {
  const elemTy =
    accScalar.literalType !== DataType.UNDEFINED ? accScalar.literalType :
    (xScalar.literalType !== DataType.UNDEFINED ? xScalar.literalType : DataType.FLOAT);

  const unary = (type: string, a: TensorNode.Class, name: string): TensorNode.Class => {
    const n = g.addNode(uniq(g, `${name}_${op.id}`))
      .init(new OperationNode.Builder(type, [a], {}))
      .as(OperationNode);
    const out = g.addNode(uniq(g, `${name}_out_${op.id}`))
      .init(new TensorNode.Builder(elemTy, [], "intermediate"))
      .as(TensorNode);
    g.addEdge(n, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);
    return out;
  };

  const bin = (type: string, a: TensorNode.Class, b: TensorNode.Class, name: string): TensorNode.Class => {
    const n = g.addNode(uniq(g, `${name}_${op.id}`))
      .init(new OperationNode.Builder(type, [a, b], {}))
      .as(OperationNode);
    const out = g.addNode(uniq(g, `${name}_out_${op.id}`))
      .init(new TensorNode.Builder(elemTy, [], "intermediate"))
      .as(TensorNode);
    g.addEdge(n, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);
    return out;
  };

  const where = (cond: TensorNode.Class, a: TensorNode.Class, b: TensorNode.Class, name: string): TensorNode.Class => {
    const n = g.addNode(uniq(g, `${name}_${op.id}`))
      .init(new OperationNode.Builder("Where", [cond, a, b], {}))
      .as(OperationNode);
    const out = g.addNode(uniq(g, `${name}_out_${op.id}`))
      .init(new TensorNode.Builder(elemTy, [], "intermediate"))
      .as(TensorNode);
    g.addEdge(n, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);
    return out;
  };

  switch (op.type) {
    case "ReduceSum":
      return bin("Add", accScalar, xScalar, "sum");

    case "ReduceMean": {
      if (!ctx.meanScale) throw new Error("[ReduceMean] missing meanScale in ctx");
      const scaled = bin("Mul", xScalar, ctx.meanScale, "mean_scale");
      return bin("Add", accScalar, scaled, "mean_acc");
    }

    case "ReduceMax":
      return bin("Max", accScalar, xScalar, "max");

    case "ReduceMin":
      return bin("Min", accScalar, xScalar, "min");

    case "ReduceProd":
      return bin("Mul", accScalar, xScalar, "prod");

    case "ReduceSumSquare": {
      const sq = bin("Mul", xScalar, xScalar, "sumsq_sq");
      return bin("Add", accScalar, sq, "sumsq_acc");
    }

    case "ReduceL1": {
      const absx = unary("Abs", xScalar, "l1_abs");
      return bin("Add", accScalar, absx, "l1_acc");
    }

    case "ReduceL2": {
      // carry keeps sqrt(sum_sq); update = sqrt((acc^2) + x^2)
      const accSq = bin("Mul", accScalar, accScalar, "l2_acc_sq");
      const xSq   = bin("Mul", xScalar, xScalar, "l2_x_sq");
      const sumSq = bin("Add", accSq, xSq, "l2_sum_sq");
      return unary("Sqrt", sumSq, "l2_sqrt");
    }

    case "ReduceLogSum": {
      // Carry meaning: log(sum_so_far)
      // Recurrence: acc' = log( exp(acc) + x )
      // Edge case: at first step acc==0 and if x==1 then log(1)=0 -> stays 0 forever.
      // Fix: inject a tiny epsilon into "sumPrev" only when (acc==0 && x==1).

      const zeroF = makeTensorConst(
        g, `f0_${op.id}`, DataType.FLOAT, "constant",
        { dataType: DataType.FLOAT, dims: [], floatData: [0] }
      );
      const oneF = makeTensorConst(
        g, `f1_${op.id}`, DataType.FLOAT, "constant",
        { dataType: DataType.FLOAT, dims: [], floatData: [1] }
      );
      const tinyF = makeTensorConst(
        g, `feps_${op.id}`, DataType.FLOAT, "constant",
        { dataType: DataType.FLOAT, dims: [], floatData: [1e-7] }
      );

      // eq0: acc == 0  (only guaranteed true on the very first loop step)
      const eq0Node = g.addNode(uniq(g, `ls_eq0_${op.id}`))
        .init(new OperationNode.Builder("Equal", [accScalar, zeroF], {}))
        .as(OperationNode);
      const eq0 = g.addNode(uniq(g, `ls_eq0_out_${op.id}`))
        .init(new TensorNode.Builder(DataType.BOOL, [], "intermediate"))
        .as(TensorNode);
      g.addEdge(eq0Node, eq0).init(new OnnxEdge.Builder(eq0.literalType, eq0.shape)).as(OnnxEdge);

      // eq1x: x == 1
      const eq1Node = g.addNode(uniq(g, `ls_eq1x_${op.id}`))
        .init(new OperationNode.Builder("Equal", [xScalar, oneF], {}))
        .as(OperationNode);
      const eq1x = g.addNode(uniq(g, `ls_eq1x_out_${op.id}`))
        .init(new TensorNode.Builder(DataType.BOOL, [], "intermediate"))
        .as(TensorNode);
      g.addEdge(eq1Node, eq1x).init(new OnnxEdge.Builder(eq1x.literalType, eq1x.shape)).as(OnnxEdge);

      // first_and_one = (acc==0 && x==1)
      const andNode = g.addNode(uniq(g, `ls_and_${op.id}`))
        .init(new OperationNode.Builder("And", [eq0, eq1x], {}))
        .as(OperationNode);
      const firstAndOne = g.addNode(uniq(g, `ls_and_out_${op.id}`))
        .init(new TensorNode.Builder(DataType.BOOL, [], "intermediate"))
        .as(TensorNode);
      g.addEdge(andNode, firstAndOne).init(new OnnxEdge.Builder(firstAndOne.literalType, firstAndOne.shape)).as(OnnxEdge);

      // basePrev = exp(acc)
      const expAcc = unary("Exp", accScalar, "logsum_exp_acc");

      // sumPrev = first_and_one ? tiny : (eq0 ? 0 : exp(acc))
      const prevIfEq0Node = g.addNode(uniq(g, `ls_prev_if_eq0_${op.id}`))
        .init(new OperationNode.Builder("Where", [eq0, zeroF, expAcc], {}))
        .as(OperationNode);
      const prevIfEq0 = g.addNode(uniq(g, `ls_prev_if_eq0_out_${op.id}`))
        .init(new TensorNode.Builder(elemTy, [], "intermediate"))
        .as(TensorNode);
      g.addEdge(prevIfEq0Node, prevIfEq0).init(new OnnxEdge.Builder(prevIfEq0.literalType, prevIfEq0.shape)).as(OnnxEdge);

      const prevNode = g.addNode(uniq(g, `ls_prev_${op.id}`))
        .init(new OperationNode.Builder("Where", [firstAndOne, tinyF, prevIfEq0], {}))
        .as(OperationNode);
      const sumPrev = g.addNode(uniq(g, `ls_prev_out_${op.id}`))
        .init(new TensorNode.Builder(elemTy, [], "intermediate"))
        .as(TensorNode);
      g.addEdge(prevNode, sumPrev).init(new OnnxEdge.Builder(sumPrev.literalType, sumPrev.shape)).as(OnnxEdge);

      // sumNow = sumPrev + x
      const sumNow = bin("Add", sumPrev, xScalar, "logsum_add");

      // acc' = log(sumNow)
      return unary("Log", sumNow, "logsum_log");
    }

    case "ReduceLogSumExp": {
      // carry keeps log(sum_exp); start acc==0 meaning "no sum yet"
      const zero = makeTensorConst(g, `rd0_${op.id}`, DataType.FLOAT, "constant", { dataType: DataType.FLOAT, dims: [], floatData: [0] });
      const eq0 = (() => {
        const n = g.addNode(uniq(g, `eq0_${op.id}`))
          .init(new OperationNode.Builder("Equal", [accScalar, zero], {}))
          .as(OperationNode);
        const out = g.addNode(uniq(g, `eq0_out_${op.id}`))
          .init(new TensorNode.Builder(DataType.BOOL, [], "intermediate"))
          .as(TensorNode);
        g.addEdge(n, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);
        return out;
      })();
      const expAcc = unary("Exp", accScalar, "lse_exp_acc");   // exp(log(sum_exp)) = sum_exp (except init)
      const base   = where(eq0, zero, expAcc, "lse_prev");     // 0 or sum_exp
      const ex     = unary("Exp", xScalar, "lse_exp_x");
      const sumExp = bin("Add", base, ex, "lse_add");
      return unary("Log", sumExp, "lse_log");
    }
  }

  // Fallback (shouldn't be hit)
  return bin("Add", accScalar, xScalar, "sum_fallback");
}
