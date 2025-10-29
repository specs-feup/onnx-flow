import BaseNode from "@specs-feup/flow/graph/BaseNode";
import OnnxEdge from "./OnnxEdge.js";
import OnnxGraph from "./OnnxGraph.js";
import { AttributeType, DataType } from "./OnnxTypes.js";
import TensorNode from "./TensorNode.js";
import { broadcastShapes, getAttr, inferConvDim, inferPoolDim, normalizeAxis, toNum, topologicalSortOperationNodes } from "./Utils.js";

/** Main */
export function inferShapes(graph: OnnxGraph.Class): void {
  const ops = topologicalSortOperationNodes(graph);

  for (const node of ops) {
    const inputs = node.getInputs?.() ?? [];
    const infos = inputs.map(inp => {
      const tns = inp.tryAs(TensorNode);
      let interEdge = null as any;
      if (tns?.type === "intermediate") interEdge = tns.getIncomers.first;
      const directEdge = graph.getEdge(inp.id, node.id)?.tryAs(OnnxEdge);
      return {
        shape: interEdge?.shape ?? directEdge?.shape ?? tns?.shape ?? [],
        dtype: interEdge?.literalType ?? directEdge?.literalType ?? tns?.literalType ?? AttributeType.UNDEFINED
      };
    });

    let outShape: (number|String)[] = [];
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
        const shapes = infos.map(i => i.shape ?? []);
        outShape = broadcastShapes(...shapes);
        // Boolean comparisons yield BOOL
        if (["Greater","Less","GreaterOrEqual","LessOrEqual","Equal","NotEqual","And","Or","Xor"].includes(node.type))
          outDtype = DataType.BOOL;
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

      /** ───── Where (already present but keep) ───── */
      case "Where": {
        const sc = infos[0]?.shape ?? [];
        const sx = infos[1]?.shape ?? [];
        const sy = infos[2]?.shape ?? [];
        outShape = broadcastShapes(sc, sx, sy);
        outDtype = infos[1]?.dtype ?? infos[2]?.dtype ?? outDtype;
        if (infos[0]?.dtype !== DataType.BOOL) {
          console.warn("Where: condition input is not BOOL.");
        }
        break;
      }

      /** ───── MatMul (keep your 2D rule, warn for >2D) ───── */
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

      /** ───── Gemm (if present): Y = alpha*A*B + beta*C; treat like MatMul + broadcast add ───── */
      case "Gemm": {
        const a = infos[0]?.shape ?? [];
        const b = infos[1]?.shape ?? [];
        if (a.length === 2 && b.length === 2) {
          const mm = [a[0], b[1]];
          const c = infos[2]?.shape ?? [];
          outShape = c.length ? broadcastShapes(mm, c) : mm;
        } else {
          outShape = [];
          console.warn("Gemm with non-2D inputs; inference skipped.");
        }
        break;
      }

      /** ───── Transpose (you had two cases; keep the attribute-based one) ───── */
      case "Transpose": {
        const inputShape = infos[0]?.shape ?? [];
        const perm = getAttr(node, "perm", inputShape.map((_, i) => i).reverse());
        outShape = perm.map((p: number) => inputShape[p] ?? 1);
        break;
      }

      /** ───── Reshape (keep) ───── */
      case "Reshape": {
        const shapeInput = inputs[1]?.tryAs(TensorNode);
        const shapeProto = shapeInput?.constantValue;
        if (shapeProto?.int64Data) {
          outShape = Array.from(shapeProto.int64Data.map((n: any) => Number(n)));
        }
        break;
      }

      /** ───── Squeeze/Unsqueeze (keep, with axis norm) ───── */
      case "Unsqueeze": {
        const tensorShape = infos[0]?.shape ?? [];
        const axesNode = inputs[1]?.tryAs(TensorNode);
        const raw =
          axesNode?.constantValue?.int64Data?.map(Number) ??
          axesNode?.constantValue?.int32Data?.map(Number) ??
          [];
        const axes = [...raw].sort((a,b)=>a-b);
        outShape = [...tensorShape];
        for (const ax of axes) outShape.splice(ax, 0, 1);
        break;
      }
      case "Squeeze": {
        const inputShape = infos[0]?.shape ?? [];
        const axesNode = inputs[1]?.tryAs(TensorNode);
        const axes = axesNode?.constantValue?.int64Data?.map(Number);
        if (!axes || axes.length === 0) outShape = inputShape.filter(d => d !== 1);
        else {
          const rank = inputShape.length;
          const norm = new Set(axes.map(a => normalizeAxis(a, rank)));
          outShape = inputShape.filter((dim, idx) => !norm.has(idx) || dim !== 1);
        }
        break;
      }

      /** ───── Concat (keep) ───── */
      case "Concat": {
        const axis = getAttr(node, "axis", 0);
        const inputShapes = infos.map(i => i.shape);
        const ref = inputShapes.find(s => s.length) ?? [];
        outShape = [...ref];
        outShape[axis] = inputShapes.reduce((sum, s) => sum + (s[axis] ?? 0), 0);
        break;
      }

      /** ───── Flatten (keep) ───── */
      case "Flatten": {
        const inputShape = infos[0]?.shape ?? [];
        const axis = getAttr(node, "axis", 1);
        const d0 = inputShape.slice(0, axis).reduce((a, b) => a * b, 1);
        const d1 = inputShape.slice(axis).reduce((a, b) => a * b, 1);
        outShape = [d0, d1];
        break;
      }

      /** ───── Expand (keep) ───── */
      case "Expand": {
        const shapeInput = inputs[1]?.tryAs(TensorNode);
        const targetShape = shapeInput?.constantValue?.int64Data?.map(Number);
        if (targetShape && targetShape.length > 0) outShape = targetShape;
        break;
      }

      /** ───── Gather / GatherElements / ScatterElements / Scatter (keep) ───── */
      case "Gather": {
        const dataShape = infos[0]?.shape ?? [];
        const indicesShape = infos[1]?.shape ?? [];
        const axis = getAttr(node, "axis", 0);
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
        const rank = dataShape.length;
        const axis = normalizeAxis(getAttr(node, "axis", 0), rank);
        // Output shape is indices.shape
        outShape = indicesShape.slice();
        outDtype = infos[0]?.dtype ?? outDtype;
        break;
      }
      case "ScatterElements":
      case "Scatter": {
        const dataShape = infos[0]?.shape ?? [];
        outShape = dataShape.slice();
        outDtype = infos[0]?.dtype ?? outDtype;
        break;
      }

      /** ───── Slice ───── */
      case "Slice": {
        const dataShape = infos[0]?.shape ?? [];
        const starts = (inputs[1]?.tryAs(TensorNode)?.constantValue?.int64Data ?? []).map(Number);
        const ends   = (inputs[2]?.tryAs(TensorNode)?.constantValue?.int64Data ?? []).map(Number);
        const axes   = (inputs[3]?.tryAs(TensorNode)?.constantValue?.int64Data ?? []).map(Number);
        const steps  = (inputs[4]?.tryAs(TensorNode)?.constantValue?.int64Data ?? []).map(Number);
        const rank = dataShape.length;
        const normAxes = (axes.length ? axes : [...Array(rank).keys()]).map(a => normalizeAxis(Number(a), rank));
        const mapAxis = new Map<number, number>();
        normAxes.forEach((ax, i) => mapAxis.set(ax, i));

        outShape = dataShape.slice();
        for (let ax = 0; ax < rank; ax++) {
          if (!mapAxis.has(ax)) continue;
          const i = mapAxis.get(ax)!;
          const s = starts[i] ?? 0;
          const e = ends[i] ?? dataShape[ax];
          const st = steps[i] ?? 1;
          const len = Math.max(0, Math.ceil((e - s) / st));
          outShape[ax] = Number.isFinite(len) ? len : outShape[ax];
        }
        break;
      }

      /** ───── Pad ───── */
      case "Pad": {
        const dataShape = infos[0]?.shape ?? [];
        const padsNode = inputs[1]?.tryAs(TensorNode);
        const pads = padsNode?.constantValue?.int64Data?.map(Number) ?? [];
        // pads is [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        const rank = dataShape.length;
        outShape = dataShape.slice();
        if (pads.length === 2 * rank) {
          for (let i = 0; i < rank; i++) {
            outShape[i] = (toNum(outShape[i]) ?? 0) + (pads[i] ?? 0) + (pads[i + rank] ?? 0);
          }
        }
        break;
      }

      /** ───── Reduces (axes + keepdims) ───── */
      case "ReduceSum":
      case "ReduceMean":
      case "ReduceProd":
      case "ReduceMin":
      case "ReduceMax": {
        const inShape = infos[0]?.shape ?? [];
        const keepdims = !!getAttr(node, "keepdims", 1);
        let axes = getAttr(node, "axes", undefined) as number[] | undefined;
        if (!axes) {
          // axes may come as a second input in some models
          const axesNode = inputs[1]?.tryAs(TensorNode);
          axes = axesNode?.constantValue?.int64Data?.map(Number);
        }
        if (!axes || axes.length === 0) {
          // reduce all
          outShape = keepdims ? inShape.map(_ => 1) : [];
        } else {
          const rank = inShape.length;
          const norm = new Set(axes.map(a => normalizeAxis(a, rank)));
          if (keepdims) {
            outShape = inShape.map((d, i) => (norm.has(i) ? 1 : d));
          } else {
            outShape = inShape.filter((_, i) => !norm.has(i));
          }
        }
        // dtype follows input (except special ops like ArgMax, not here)
        outDtype = infos[0]?.dtype ?? outDtype;
        break;
      }

      /** ───── ArgMax / ArgMin (indices along axis, dtype INT64 by default) ───── */
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

      /** ───── Conv / Pooling (NCHW assumed; adjust if needed) ───── */
      case "Conv": {
        const x = infos[0]?.shape ?? []; // [N,Cin,H,W]
        const w = infos[1]?.shape ?? []; // [Cout,Cin,kH,kW] (default)
        const n = x[0], cin = x[1], h = x[2], wdim = x[3];
        const cout = w[0];
        const strides = getAttr(node, "strides", [1,1]);
        const pads = getAttr(node, "pads", [0,0,0,0]); // [top,left,bottom,right]
        const dil = getAttr(node, "dilations", [1,1]);
        const kH = w[2], kW = w[3];
        const Hout = inferConvDim(h ?? 0, kH ?? 1, strides[0] ?? 1, pads[0] ?? 0, pads[2] ?? 0, dil[0] ?? 1);
        const Wout = inferConvDim(wdim ?? 0, kW ?? 1, strides[1] ?? 1, pads[1] ?? 0, pads[3] ?? 0, dil[1] ?? 1);
        outShape = [n, cout, Hout, Wout];
        outDtype = infos[0]?.dtype ?? outDtype;
        break;
      }
      case "MaxPool":
      case "AveragePool": {
        const x = infos[0]?.shape ?? []; // [N,C,H,W]
        const n = x[0], c = x[1], h = x[2], wdim = x[3];
        const kernel = getAttr(node, "kernel_shape", [1,1]);
        const strides = getAttr(node, "strides", [1,1]);
        const pads = getAttr(node, "pads", [0,0,0,0]);
        const dil = getAttr(node, "dilations", [1,1]);
        const Hout = inferPoolDim(h ?? 0, kernel[0] ?? 1, strides[0] ?? 1, pads[0] ?? 0, pads[2] ?? 0, dil[0] ?? 1);
        const Wout = inferPoolDim(wdim ?? 0, kernel[1] ?? 1, strides[1] ?? 1, pads[1] ?? 0, pads[3] ?? 0, dil[1] ?? 1);
        outShape = [n, c, Hout, Wout];
        outDtype = infos[0]?.dtype ?? outDtype;
        break;
      }

      /** ───── BatchNormalization (shape preserved, dtype from input) ───── */
      case "BatchNormalization": {
        outShape = infos[0]?.shape ?? [];
        outDtype = infos[0]?.dtype ?? outDtype;
        break;
      }

      /** ───── Cast (dtype changes, shape preserved) ───── */
      case "Cast": {
        outShape = infos[0]?.shape ?? [];
        outDtype = getAttr(node, "to", outDtype);
        break;
      }

      /** ───── Shape (outputs 1D tensor of input rank) ───── */
      case "Shape": {
        const rank = (infos[0]?.shape ?? []).length;
        outShape = [rank];
        outDtype = DataType.INT64;
        break;
      }

      /** ───── Softmax (shape preserved) ───── */
      case "Softmax": {
        outShape = infos[0]?.shape ?? [];
        outDtype = infos[0]?.dtype ?? outDtype;
        break;
      }

      /** ───── Range (1D length if all constants available) ───── */
      case "Range": {
        const st = inputs[0]?.tryAs(TensorNode)?.constantValue;
        const ed = inputs[1]?.tryAs(TensorNode)?.constantValue;
        const dt = inputs[2]?.tryAs(TensorNode)?.constantValue;
        if (st && ed && dt && st.int64Data && ed.int64Data && dt.int64Data) {
          const start = Number(st.int64Data[0]);
          const end = Number(ed.int64Data[0]);
          const step = Number(dt.int64Data[0] || 1);
          const len = Math.max(0, Math.ceil((end - start) / step));
          outShape = [len];
        } else {
          outShape = []; // unknown
        }
        break;
      }

      /** ───── Loop / If / Scan (keep your existing Loop behavior; If/Scan pass shapes through best-effort) ───── */
      case "Loop": {
        const initState = infos[2];
        if (initState && initState.shape) {
          outShape = initState.shape.slice();
          outDtype = initState.dtype ?? outDtype;
        } else {
          const outputs = node.getOutgoers?.targets ?? graph.emptyCollection(BaseNode);
          const firstOutT = outputs.first()?.tryAs?.(TensorNode);
          outShape = firstOutT?.shape;
          outDtype = firstOutT?.literalType ?? outDtype;
        }
        break;
      }

      default: {
        // Fallback: copy first known input
        const first = infos.find(i => i.shape !== undefined);
        if (first) {
          outShape = first.shape;
          outDtype = first.dtype;
        }
      }
    }

    // Rewire edges (keep your current behavior)
    const outputs = node.getOutgoers.targets;
    const outputTensors = outputs.filter((t: any) => t.is(TensorNode));

    node.getOutgoers.forEach((e: any) => graph.getEdgeById(e.id).remove());
    for (const output of outputs) {
      graph.addEdge(node, output).init(new OnnxEdge.Builder(outDtype, outShape));
    }
    for (const out of outputTensors) {
      const tn = out.tryAs(TensorNode);
      if (tn && (tn.type === "intermediate" || (tn.shape?.length ?? 0) === 0)) {
        tn.setShape(outShape);
        tn.setLiteralType(outDtype);
      }
    }
  }
}
