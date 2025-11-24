import BaseNode from "@specs-feup/flow/graph/BaseNode";
import OnnxGraph from "./OnnxGraph.js";
import TensorNode from "./TensorNode.js";
import { AttributeType, DataType, TensorProto } from "./OnnxTypes.js";
import {
  broadcastShapes,
  getAttr,
  inferPoolDim,
  normalizeAxis,
  toNum,
  topologicalSortOperationNodes,
  toStaticShape,
} from "./Utils.js";
import OnnxEdge from "./OnnxEdge.js";
import OperationNode from "./OperationNode.js";

/** Helper: resolve a tensor's shape from its node or incoming edge */
function resolveTensorShape(t: TensorNode.Class): (number | String)[] {
  if (t.shape && t.shape.length) return t.shape as (number | String)[];
  const interEdge = t.getIncomers?.first() as OnnxEdge.Class | undefined;
  if (interEdge?.shape && interEdge.shape.length) {
    return interEdge.shape as (number | String)[];
  }
  return [];
}

/** Helper: read int array from a TensorProto, including rawData */
function tensorProtoToIntArray(t?: TensorProto): number[] {
  if (!t) return [];

  if (t.int64Data && t.int64Data.length) {
    return Array.from(t.int64Data, (n) => Number(n));
  }
  if (t.int32Data && t.int32Data.length) {
    return Array.from(t.int32Data, (n) => Number(n));
  }

  const raw : any = t.rawData ?? t.rawData;
  if (!raw) return [];

  let buf: Buffer;
  if (typeof raw === "string") {
    buf = Buffer.from(raw, "base64");
  } else {
    buf = Buffer.from(raw);
  }

  const out: number[] = [];
  if (t.dataType === DataType.INT64) {
    for (let i = 0; i + 8 <= buf.length; i += 8) {
      out.push(Number(buf.readBigInt64LE(i)));
    }
  } else if (t.dataType === DataType.INT32) {
    for (let i = 0; i + 4 <= buf.length; i += 4) {
      out.push(buf.readInt32LE(i));
    }
  }

  return out;
}

/** Main shape inference */
export default function inferShapes(graph: OnnxGraph.Class): void {
  const ops = topologicalSortOperationNodes(graph);

  for (const node of ops) {
    const inputs = node.getInputs?.() ?? [];

    const infos = inputs.map((inp) => {
      if (!inp) {
        return {
          shape: [] as (number | String)[],
          dtype: AttributeType.UNDEFINED,
        };
      }
      const tns = inp.tryAs(TensorNode);
      let interEdge;
      if (tns && tns.type === "intermediate") {
        interEdge = tns.getIncomers.sources.first();
      }
      const directEdge = graph.getEdge(inp.id, node.id)?.tryAs(OnnxEdge);

      return {
        shape:
          interEdge?.shape ??
          directEdge?.shape ??
          (tns?.shape as (number | String)[] | undefined) ??
          [],
        dtype:
          interEdge?.literalType ??
          directEdge?.literalType ??
          tns?.literalType ??
          AttributeType.UNDEFINED,
      };
    });

    let outShape: (number | String)[] = [];
    let outDtype = infos[0]?.dtype ?? AttributeType.UNDEFINED;

    switch (node.type) {
      /** ───── Elementwise (broadcast) ───── */
      case "Add":
      case "Sub":
      case "Mul":
      case "Div":
      case "Pow":
      case "Min":
      case "Max":
      case "And":
      case "Or":
      case "Xor":
      case "Greater":
      case "Less":
      case "GreaterOrEqual":
      case "LessOrEqual":
      case "Equal":
      case "NotEqual": {
        const shapes = infos.map((i) => toStaticShape(i.shape) ?? []);
        outShape = broadcastShapes(...shapes);
        if (
          [
            "Greater",
            "Less",
            "GreaterOrEqual",
            "LessOrEqual",
            "Equal",
            "NotEqual",
            "And",
            "Or",
            "Xor",
          ].includes(node.type)
        ) {
          outDtype = DataType.BOOL;
        }
        break;
      }

      /** ───── Unary activations (shape preserved) ───── */
      case "Relu":
      case "LeakyRelu":
      case "Sigmoid":
      case "Tanh":
      case "Exp":
      case "Sqrt":
      case "Abs":
      case "Neg":
      case "Clip": {
        outShape = infos[0]?.shape ?? [];
        break;
      }

      /** ───── Where ───── */
      case "Where": {
        const sc = infos[0]?.shape ?? [];
        const sx = infos[1]?.shape ?? [];
        const sy = infos[2]?.shape ?? [];
        outShape = broadcastShapes(toStaticShape(sc), toStaticShape(sx), toStaticShape(sy));
        outDtype = infos[1]?.dtype ?? infos[2]?.dtype ?? outDtype;
        if (infos[0]?.dtype !== DataType.BOOL) {
          console.warn("Where: condition input is not BOOL.");
        }
        break;
      }

      /** ───── MatMul (simple 2D) ───── */
      case "MatMul": {
        if (infos.length >= 2) {
          const [a, b] = infos;
          if (a.shape.length === 2 && b.shape.length === 2) {
            outShape = [a.shape[0], b.shape[1]];
          } else {
            console.warn("MatMul with non-2D tensors:", a.shape, b.shape);
            outShape = [];
          }
        }
        break;
      }

      /** ───── Gemm (2D MatMul + broadcast add) ───── */
      case "Gemm": {
        const a = infos[0]?.shape ?? [];
        const b = infos[1]?.shape ?? [];
        if (a.length === 2 && b.length === 2) {
          const mm: (number | String)[] = [a[0], b[1]];
          const c = infos[2]?.shape ?? [];
          outShape = c.length ? broadcastShapes(toStaticShape(mm), toStaticShape(c)) : mm;
        } else {
          outShape = [];
          console.warn("Gemm with non-2D inputs; inference skipped.");
        }
        break;
      }

      /** ───── Transpose (with perm attr, default = reverse) ───── */
      case "Transpose": {
        const inputShape = infos[0]?.shape ?? [];
        const perm = getAttr(
          node,
          "perm",
          inputShape.map((_, i) => i).reverse()
        ) as number[];
        outShape = perm.map((p) => inputShape[p] ?? 1);
        break;
      }

      /** ───── Reshape (ONNX rules: 0 / -1, product preserved) ───── */
      case "Reshape": {
        const inputShape = infos[0]?.shape ?? [];
        const shapeInput = inputs[1]?.tryAs(TensorNode);
        const shapeTensor =
          shapeInput?.constantValue ?? shapeInput?.originalInitializer;

        const target = tensorProtoToIntArray(shapeTensor);

        if (target.length > 0 && inputShape.length > 0) {
          const inNums = inputShape.map((d) => toNum(d) ?? 1);
          const prodIn =
            inNums.reduce((a, b) => a * (b || 1), 1) || 1;

          let inferIndex = -1;
          let knownProd = 1;
          const resolved = target.slice();

          // 0 → copy from input
          resolved.forEach((d, i) => {
            if (d === 0) {
              resolved[i] = inNums[i] ?? 1;
            }
          });

          // -1 → infer from remaining product
          resolved.forEach((d, i) => {
            if (d === -1) {
              if (inferIndex !== -1) {
                throw new Error(
                  "Reshape: multiple -1 in target shape not allowed"
                );
              }
              inferIndex = i;
            } else {
              knownProd *= d || 1;
            }
          });

          if (inferIndex !== -1) {
            const missing = prodIn / (knownProd || 1);
            resolved[inferIndex] = missing;
          }

          outShape = resolved;
        } else {
          outShape = inputShape.slice();
        }
        break;
      }

      /** ───── Unsqueeze / Squeeze ───── */
      case "Unsqueeze": {
        const tensorShape = infos[0]?.shape ?? [];
        const axesNode = inputs[1]?.tryAs(TensorNode);
        const raw =
          axesNode?.constantValue?.int64Data?.map(Number) ??
          axesNode?.constantValue?.int32Data?.map(Number) ??
          [];
        const axes = [...raw].sort((a, b) => a - b);
        outShape = [...tensorShape];
        for (const ax of axes) outShape.splice(ax, 0, 1);
        break;
      }

      case "Squeeze": {
        const inputShape = infos[0]?.shape ?? [];
        const axesNode = inputs[1]?.tryAs(TensorNode);
        const axes = axesNode?.constantValue?.int64Data?.map(Number);
        if (!axes || axes.length === 0) {
          outShape = inputShape.filter((d) => d !== 1);
        } else {
          const rank = inputShape.length;
          const norm = new Set(axes.map((a) => normalizeAxis(a, rank)));
          outShape = inputShape.filter(
            (dim, idx) => !norm.has(idx) || dim !== 1
          );
        }
        break;
      }

      /** ───── Gather / GatherElements / Scatter(Elements) ───── */
      case "Gather": {
        const dataShape = infos[0]?.shape ?? [];
        const indicesShape = infos[1]?.shape ?? [];
        const axisRaw = getAttr(node, "axis", 0) as number;
        const axis = normalizeAxis(axisRaw, dataShape.length);
        outShape = [
          ...dataShape.slice(0, axis),
          ...indicesShape,
          ...dataShape.slice(axis + 1),
        ];
        outDtype = infos[0]?.dtype ?? outDtype;
        break;
      }

      case "GatherElements": {
        const dataShape = infos[0]?.shape ?? [];
        const indicesShape = infos[1]?.shape ?? [];
        const rankData = dataShape.length;

        let axis = getAttr(node, "axis", 0) as number;
        if (rankData > 0) axis = normalizeAxis(axis, rankData);

        // ONNX: output shape == indices.shape
        outShape = indicesShape.slice();
        outDtype = infos[0]?.dtype ?? outDtype;
        break;
      }

      case "ScatterElements": {
        const dataShape = infos[0]?.shape ?? [];
        const indicesShape = infos[1]?.shape ?? [];
        const updatesShape = infos[2]?.shape ?? [];
        const rankData = dataShape.length;

        let axis = getAttr(node, "axis", 0) as number;
        if (rankData > 0) axis = normalizeAxis(axis, rankData);

        // Optional sanity checks (non-fatal)
        if (indicesShape.length && updatesShape.length) {
          if (
            indicesShape.length !== rankData ||
            updatesShape.length !== rankData
          ) {
            // console.warn(...) if you want
          }
          const axOk =
            indicesShape[axis] === undefined ||
            updatesShape[axis] === undefined ||
            indicesShape[axis] === updatesShape[axis];
          if (!axOk) {
            // console.warn(...)
          }
        }

        outShape = dataShape.slice();
        outDtype = infos[0]?.dtype ?? outDtype;
        break;
      }

      case "Scatter": {
        const dataShape = infos[0]?.shape ?? [];
        outShape = dataShape.slice();
        outDtype = infos[0]?.dtype ?? outDtype;
        break;
      }

      /** ───── Slice (with starts/ends/axes/steps) ───── */
      case "Slice": {
        const dataShape = infos[0]?.shape ?? [];
        const rank = dataShape.length;
        if (rank === 0) {
          outShape = [];
          break;
        }

        const startsNode = inputs[1]?.tryAs(TensorNode);
        const endsNode = inputs[2]?.tryAs(TensorNode);
        const axesNode = inputs[3]?.tryAs(TensorNode);
        const stepsNode = inputs[4]?.tryAs(TensorNode);

        const starts = tensorProtoToIntArray(
          startsNode?.constantValue ?? startsNode?.originalInitializer
        );
        const ends = tensorProtoToIntArray(
          endsNode?.constantValue ?? endsNode?.originalInitializer
        );
        let axes = tensorProtoToIntArray(
          axesNode?.constantValue ?? axesNode?.originalInitializer
        );
        const steps = tensorProtoToIntArray(
          stepsNode?.constantValue ?? stepsNode?.originalInitializer
        );

        const normAxis = (ax: number, r: number) =>
          r > 0 ? ((ax % r) + r) % r : 0;

        if (!axes.length) {
          axes = Array.from(
            { length: starts.length || rank },
            (_, i) => i
          );
        } else {
          axes = axes.map((a) => normAxis(a, rank));
        }

        const out = dataShape.slice();

        for (let i = 0; i < axes.length; i++) {
          const ax = axes[i];
          const len = dataShape[ax] ?? 0;
          if (len === 0) continue;

          let s = starts[i] ?? 0;
          let e = ends[i] ?? len;
          const step = steps[i] ?? 1;
          if (step === 0) continue;

          const normPos = (pos: number) =>
            pos < 0 ? Math.max(0, len + pos) : Math.min(len, pos);

          s = normPos(s);
          e = normPos(e);

          const size = Math.max(0, Math.ceil((e - s) / step));
          out[ax] = size;
        }

        outShape = out;
        break;
      }

      /** ───── Pad ───── */
      case "Pad": {
        const dataShape = infos[0]?.shape ?? [];
        const padsNode = inputs[1]?.tryAs(TensorNode);
        const pads =
          padsNode?.constantValue?.int64Data?.map(Number) ?? [];
        const rank = dataShape.length;
        outShape = dataShape.slice();
        if (pads.length === 2 * rank) {
          for (let i = 0; i < rank; i++) {
            outShape[i] =
              (toNum(outShape[i]) ?? 0) +
              (pads[i] ?? 0) +
              (pads[i + rank] ?? 0);
          }
        }
        break;
      }

      /** ───── Reduces (Sum/Mean/Prod/Min/Max) ───── */
      case "ReduceSum":
      case "ReduceMean":
      case "ReduceProd":
      case "ReduceMin":
      case "ReduceMax": {
        const inShape = infos[0]?.shape ?? [];
        const keepdims = !!getAttr(node, "keepdims", 1);

        let axesAttr = getAttr(node, "axes", undefined) as
          | number[]
          | number
          | undefined;
        let axes: number[] | undefined =
          Array.isArray(axesAttr)
            ? axesAttr.map(Number)
            : typeof axesAttr === "number"
            ? [Number(axesAttr)]
            : undefined;

        if (!axes) {
          const axesNode = inputs[1]?.tryAs(TensorNode);
          const cv = axesNode?.constantValue;
          const raw = cv?.int64Data ?? cv?.int32Data ?? cv?.uint64Data;
          if (raw && raw.length !== undefined) {
            axes = Array.from(raw).map((v) => Number(v));
          }
        }

        if (!axes || axes.length === 0) {
          outShape = keepdims ? inShape.map(() => 1) : [];
        } else {
          const rank = inShape.length;
          const norm = new Set(axes.map((a) => normalizeAxis(a, rank)));
          outShape = keepdims
            ? inShape.map((d, i) => (norm.has(i) ? 1 : d))
            : inShape.filter((_, i) => !norm.has(i));
        }
        outDtype = infos[0]?.dtype ?? outDtype;
        break;
      }

      /** ───── ReduceLogSumExp ───── */
      case "ReduceLogSumExp": {
        const dataShape = infos[0]?.shape ?? [];
        const rank = dataShape.length;

        if (rank === 0) {
          outShape = [];
          outDtype = infos[0]?.dtype ?? outDtype;
          break;
        }

        const attrs = (node.getAttributes() ?? node.attributes ?? {});
        let axes: number[] | undefined = attrs.axes;

        if (!Array.isArray(axes) || axes.length === 0) {
          axes = Array.from({ length: rank }, (_, i) => i);
        } else {
          axes = axes.map((a: number) =>
            a < 0 ? ((a % rank) + rank) % rank : a
          );
        }

        const keepdims = Number(attrs.keepdims ?? 1);

        if (keepdims) {
          outShape = dataShape.slice();
          for (const ax of axes) {
            outShape[ax] = 1;
          }
        } else {
          const axeSet = new Set(axes);
          outShape = dataShape.filter((_, i) => !axeSet.has(i));
        }

        outDtype = infos[0]?.dtype ?? outDtype;
        break;
      }

      /** ───── ArgMax / ArgMin ───── */
      case "ArgMax":
      case "ArgMin": {
        const inShape = infos[0]?.shape ?? [];
        const keepdims = !!getAttr(node, "keepdims", 1);
        const axis = normalizeAxis(getAttr(node, "axis", 0), inShape.length);
        if (keepdims) {
          outShape = inShape.map((d, i) => (i === axis ? 1 : d));
        } else {
          outShape = inShape.filter((_, i) => i !== axis);
        }
        outDtype = DataType.INT64;
        break;
      }

      /** ───── Shape (with start/end/ to) ───── */
      case "Shape": {
        const inputShape = infos[0]?.shape ?? [];
        const rank = inputShape.length;
        if (rank === 0) {
          outShape = [0];
          outDtype = DataType.INT64;
          break;
        }

        const attrs = node.getAttributes() ?? node.attributes ?? {};
        const hasStart = Object.prototype.hasOwnProperty.call(
          attrs,
          "start"
        );
        const hasEnd = Object.prototype.hasOwnProperty.call(attrs, "end");

        let start = hasStart ? Number(attrs.start) : 0;
        let end = hasEnd ? Number(attrs.end) : rank;

        const norm = (idx: number, r: number) =>
          r > 0 ? ((idx % r) + r) % r : 0;

        start = norm(start, rank);
        end = norm(end, rank);

        start = Math.max(0, Math.min(start, rank));
        end = Math.max(0, Math.min(end, rank));

        let length = Math.max(0, end - start);

        if (!hasStart && !hasEnd && length === 0 && rank > 0) {
          length = rank;
        }

        outShape = [length];

        const toAttr = attrs.to;
        if (typeof toAttr === "number") {
          outDtype = toAttr as number;
        } else {
          outDtype = DataType.INT64;
        }
        break;
      }

      /** ───── ConstantOfShape (with Shape(X) fallback) ───── */
      case "ConstantOfShape": {
        const shapeTensor = inputs[0]?.tryAs(TensorNode);
        let shape: (number | String)[] = [];

        if (shapeTensor) {
          const cv = shapeTensor.constantValue;

          if (cv?.int64Data?.length) {
            shape = Array.from(cv.int64Data, (n) => Number(n));
          } else if (cv?.int32Data?.length) {
            shape = Array.from(cv.int32Data, (n) => Number(n));
          }

          if (!shape.length) {
            const producers =
              shapeTensor.getIncomers?.sources ??
              graph.emptyCollection(BaseNode);
            const shapeOp = producers
              .filterIs(OperationNode)
              .filter((op) => op.type === "Shape")
              .first();

            if (shapeOp) {
              const shapeInputs = shapeOp.getInputs?.() ?? [];
              const xTensor = shapeInputs[0]?.tryAs(TensorNode);
              if (xTensor) {
                const xShape = resolveTensorShape(xTensor);
                if (xShape.length) shape = xShape.slice();
              }
            }
          }
        }

        // 🔧 NEW: if we still don't know the shape, reuse the existing output shape
        if (!shape.length) {
          const outs =
            node.getOutgoers?.targets ?? graph.emptyCollection(BaseNode);
          const outT = outs
            .filter((t) => t.is(TensorNode))
            .first()
            ?.as(TensorNode);

          if (outT && Array.isArray(outT.shape) && outT.shape.length > 0) {
            shape = [...outT.shape];
          }
        }

        outShape = shape;
        break;
      }

      /** ───── OneHot ───── */
      case "OneHot": {
        const indicesShape = infos[0]?.shape ?? [];

        const depthTensorNode = inputs[1]?.tryAs(TensorNode);
        const depthTensor =
          depthTensorNode?.constantValue ??
          depthTensorNode?.originalInitializer;
        const depthArr = tensorProtoToIntArray(depthTensor);
        const depth = depthArr[0] ?? 0;

        if (indicesShape.length > 0) {
          outShape = depth > 0 ? [...indicesShape, depth] : [...indicesShape, 1];
        } else {
          outShape = depth > 0 ? [depth] : [];
        }

        const valuesTensor = inputs[2]?.tryAs(TensorNode);
        outDtype = valuesTensor?.literalType ?? DataType.FLOAT;
        break;
      }

      /** ───── Concat ───── */
      case "Concat": {
        const axis = getAttr(node, "axis", 0) as number;
        const inputShapes = infos.map((i) => i.shape);
        const ref = inputShapes.find((s) => s.length) ?? [];
        outShape = [...ref];
        outShape[axis] = inputShapes.reduce(
          (sum, s) => sum + (toNum(s[axis]) ?? 0),
          0
        );
        break;
      }

      /** ───── Flatten ───── */
      case "Flatten": {
        const inputShape = infos[0]?.shape ?? [];
        const axis = getAttr(node, "axis", 1) as number;
        const d0 = inputShape
          .slice(0, axis)
          .reduce((a, b) => (toNum(a) ?? 1) * (toNum(b) ?? 1), 1);
        const d1 = inputShape
          .slice(axis)
          .reduce((a, b) => (toNum(a) ?? 1) * (toNum(b) ?? 1), 1);
        outShape = [d0, d1];
        break;
      }

      /** ───── Expand (with Expand(x, Shape(X)) pattern) ───── */
      case "Expand": {
        const dataShape = infos[0]?.shape ?? [];
        const shapeInput = inputs[1]?.tryAs(TensorNode);
        let targetShape: (String | number)[] | undefined;

        if (shapeInput) {
          const cv = shapeInput.constantValue;
          if (cv?.int64Data?.length) {
            targetShape = Array.from(cv.int64Data, (n) => Number(n));
          } else if (cv?.int32Data?.length) {
            targetShape = Array.from(cv.int32Data, (n) => Number(n));
          }

          if (!targetShape?.length) {
            const producers =
              shapeInput.getIncomers?.sources ??
              graph.emptyCollection(BaseNode);
            const shapeOp = producers
              .filterIs(OperationNode)
              .filter((op) => op.type === "Shape")
              .first();

            if (shapeOp) {
              const shapeInputs = shapeOp.getInputs?.() ?? [];
              const xTensor = shapeInputs[0]?.tryAs(TensorNode);
              if (xTensor) {
                const xShape = resolveTensorShape(xTensor);
                if (xShape.length) targetShape = xShape.slice();
              }
            }
          }
        }

        if (targetShape && targetShape.length > 0) {
          outShape = targetShape;
        } else if (dataShape.length > 0) {
          outShape = dataShape.slice();
        } else {
          outShape = [];
        }
        break;
      }

      /** ───── Conv (NCHW) with pads/auto_pad/dilations ───── */
      case "Conv": {
        const xShape = infos[0]?.shape ?? [];
        const wShape = infos[1]?.shape ?? [];

        if (xShape.length === 0 || wShape.length === 0) {
          const first = infos.find((i) => i.shape?.length);
          if (first) {
            outShape = first.shape.slice();
            outDtype = first.dtype ?? outDtype;
          }
          break;
        }

        if (xShape.length !== 4 || wShape.length !== 4) {
          outShape = xShape.slice();
          outDtype = infos[0]?.dtype ?? outDtype;
          break;
        }

        let [N, C, H, W] = xShape.map(toNum) as number[];
        let [M, Cw, kH, kW] = wShape.map(toNum) as number[];

        const attrs = node.getAttributes() ?? node.attributes ?? {};

        const group = Number(attrs.group ?? 1);

        let strides = attrs.strides as number[] | undefined;
        if (!Array.isArray(strides) || strides.length !== 2) {
          strides = [1, 1];
        }
        const [sH, sW] = strides.map(Number);

        let dilations = attrs.dilations as number[] | undefined;
        if (!Array.isArray(dilations) || dilations.length !== 2) {
          dilations = [1, 1];
        }
        const [dH, dW] = dilations.map(Number);

        let pads = attrs.pads as number[] | undefined;
        let padTop = 0,
          padLeft = 0,
          padBottom = 0,
          padRight = 0;

        const autoPad = (attrs.auto_pad ?? "NOTSET") as string;

        if (Array.isArray(pads) && pads.length === 4) {
          [padTop, padLeft, padBottom, padRight] = pads.map(Number);
        } else if (autoPad === "SAME_UPPER" || autoPad === "SAME_LOWER") {
          const kEffH = dH * (kH - 1) + 1;
          const kEffW = dW * (kW - 1) + 1;

          const outH = Math.ceil(H / sH);
          const outW = Math.ceil(W / sW);

          const totalPadH = Math.max(
            0,
            (outH - 1) * sH + kEffH - H
          );
          const totalPadW = Math.max(
            0,
            (outW - 1) * sW + kEffW - W
          );

          if (autoPad === "SAME_UPPER") {
            padTop = Math.floor(totalPadH / 2);
            padBottom = totalPadH - padTop;
            padLeft = Math.floor(totalPadW / 2);
            padRight = totalPadW - padLeft;
          } else {
            padBottom = Math.floor(totalPadH / 2);
            padTop = totalPadH - padBottom;
            padRight = Math.floor(totalPadW / 2);
            padLeft = totalPadW - padRight;
          }
        }

        const kEffH = dH * (kH - 1) + 1;
        const kEffW = dW * (kW - 1) + 1;

        const H_padded = H + padTop + padBottom;
        const W_padded = W + padLeft + padRight;

        const H_out = Math.floor((H_padded - kEffH) / sH + 1);
        const W_out = Math.floor((W_padded - kEffW) / sW + 1);

        if (group > 0 && Cw > 0) {
          const expectedC = Cw * group;
          if (C !== expectedC) {
            // you can console.warn here if you want
          }
        }

        outShape = [N, M, H_out, W_out];
        outDtype = infos[0]?.dtype ?? outDtype;
        break;
      }

      /** ───── MaxPool / AveragePool ───── */
      case "MaxPool":
      case "AveragePool": {
        const x = toStaticShape(infos[0]?.shape) ?? [];
        const n = x[0],
          c = x[1],
          h = x[2],
          wdim = x[3];
        const kernel = (getAttr(node, "kernel_shape", [1, 1]) ||
          []) as number[];
        const strides = (getAttr(node, "strides", [1, 1]) || []) as number[];
        const pads = (getAttr(node, "pads", [0, 0, 0, 0]) ||
          []) as number[];
        const dil = (getAttr(node, "dilations", [1, 1]) || []) as number[];

        const Hout = inferPoolDim(
          h ?? 0,
          kernel[0] ?? 1,
          strides[0] ?? 1,
          pads[0] ?? 0,
          pads[2] ?? 0,
          dil[0] ?? 1
        );
        const Wout = inferPoolDim(
          wdim ?? 0,
          kernel[1] ?? 1,
          strides[1] ?? 1,
          pads[1] ?? 0,
          pads[3] ?? 0,
          dil[1] ?? 1
        );
        outShape = [n, c, Hout, Wout];
        outDtype = infos[0]?.dtype ?? outDtype;
        break;
      }

      /** ───── BatchNormalization (shape preserved) ───── */
      case "BatchNormalization": {
        outShape = infos[0]?.shape ?? [];
        outDtype = infos[0]?.dtype ?? outDtype;
        break;
      }

      /** ───── Cast (dtype changes, shape preserved) ───── */
      case "Cast": {
        outShape = infos[0]?.shape ?? [];
        outDtype = getAttr(node, "to", outDtype) as number;
        break;
      }

      /** ───── Softmax (shape preserved) ───── */
      case "Softmax": {
        outShape = infos[0]?.shape ?? [];
        outDtype = infos[0]?.dtype ?? outDtype;
        break;
      }

      /** ───── Range (1D) ───── */
      case "Range": {
        const st = inputs[0]?.tryAs(TensorNode)?.constantValue;
        const ed = inputs[1]?.tryAs(TensorNode)?.constantValue;
        const dt = inputs[2]?.tryAs(TensorNode)?.constantValue;
        if (
          st &&
          ed &&
          dt &&
          st.int64Data &&
          ed.int64Data &&
          dt.int64Data
        ) {
          const start = Number(st.int64Data[0]);
          const end = Number(ed.int64Data[0]);
          const step = Number(dt.int64Data[0] || 1);
          const len = Math.max(0, Math.ceil((end - start) / step));
          outShape = [len];
        } else {
          outShape = [];
        }
        break;
      }

      /** ───── Loop (carry shape passthrough) ───── */
      case "Loop": {
        const initState = infos[2];
        if (initState && initState.shape) {
          outShape = initState.shape.slice();
          outDtype = initState.dtype ?? outDtype;
        } else {
          const outputs =
            node.getOutgoers?.targets ??
            graph.emptyCollection(BaseNode);
          const firstOutT = outputs.first()?.tryAs?.(TensorNode);
          outShape = (firstOutT?.shape as (number | String)[]) ?? [];
          outDtype = firstOutT?.literalType ?? outDtype;
        }
        break;
      }

      default: {
        const first = infos.find((i) => i.shape !== undefined);
        if (first) {
          outShape = first.shape;
          outDtype = first.dtype;
        }
      }
    }

    // Rewire edges with updated shapes/dtypes
    const outputs = node.getOutgoers.targets;
    const outputTensors = outputs.filter((t) => t.is(TensorNode));

    node.getOutgoers.forEach((e) =>
      graph.getEdgeById(e.id).remove()
    );

    for (const output of outputs) {
      graph
        .addEdge(node, output)
        .init(new OnnxEdge.Builder(outDtype, outShape));
    }

    if (Array.isArray(outShape) && outShape.length > 0) {
      for (const out of outputTensors) {
        const tn = out.tryAs(TensorNode);
        if (!tn) continue;
        if (tn.type === "intermediate" || !tn.shape?.length) {
          tn.setShape(outShape);
          tn.setLiteralType(outDtype);
        }
      }
    }
  }
}
