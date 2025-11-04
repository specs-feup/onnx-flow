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
      // 1) normal running sum for the recurrence
      const sumNow = bin("Add", accScalar, xScalar, "logsum_accum");

      // 2) log of the final sum
      const logNow = unary("Log", sumNow, "logsum_log");

      // 3) detect last iteration without changing BuildLoop:
      // Prefer a ready-made boolean from ctx (if your BuildLoop already exposes one),
      // else derive it from the loop index and the trip count.
      let isLast: TensorNode.Class | null = null;

      if ((ctx as any).isFinalIter) {
        // boolean [] provided by BuildLoop subgraph
        isLast = (ctx as any).isFinalIter;
      } else if ((ctx as any).iterScalar && (ctx as any).reduceLenScalar) {
        // derive: iter == reduceLen-1
        const one = makeTensorConst(
          g,
          `one_${op.id}`,
          DataType.INT64,
          "constant",
          { dataType: DataType.INT64, dims: [], int64Data: [1n] }
        );
        const lastIdx = bin("Sub", (ctx as any).reduceLenScalar, one, "logsum_last_idx"); // reduceLen - 1
        isLast = bin("Equal", (ctx as any).iterScalar, lastIdx, "logsum_is_last"); // BOOL []
      } else {
        // Conservative fallback: behave like running sum (won't be exact for log-sum)
        return sumNow;
      }

      // 4) write sum during iterations, log(sum) only at the final one
      return where(isLast, logNow, sumNow, "logsum_select");
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
