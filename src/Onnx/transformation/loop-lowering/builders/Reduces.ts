import Graph from "@specs-feup/flow/graph/Graph";
import OnnxEdge from "@specs-feup/onnx-flow/Onnx/OnnxEdge";
import OnnxGraph from "@specs-feup/onnx-flow/Onnx/OnnxGraph";
import type {
    ConcreteValueNode,
    Shape,
    StaticShape,
    TensorProto,
} from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import { DataType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import OperationNode from "@specs-feup/onnx-flow/Onnx/OperationNode";
import TensorNode from "@specs-feup/onnx-flow/Onnx/TensorNode";
import {
    toStaticShape,
    uniq,
    makeTensorConst,
    int64Vec,
    computeStrides,
    scalarInt64,
    bool,
    readConstIntegerVectorFromTensorNode,
    asValueNode,
    scalarFloat,
} from "@specs-feup/onnx-flow/Onnx/Utils";
import type { LoopBuilder, BuildResult, LoopCtx } from "../BuildLoop.js";
import {
    resolveFusedInput,
    ensureFlatInput,
    decodeMixedRadix,
    buildLinearIndex,
    unsqueezeIdx,
    squeezeIfLen1,
} from "../BuildLoop.js";
import handleReduceElem from "../handlers/Reduces.js";
import inferShapes from "@specs-feup/onnx-flow/Onnx/InferShapes";

function normalizeAxes(axes: number[] | undefined, rank: number): number[] {
    if (!axes || axes.length === 0) return Array.from({ length: rank }, (_, i) => i);
    return Array.from(new Set(axes.map((a) => (a < 0 ? a + rank : a)))).sort((a, b) => a - b);
}

function makeOutShape(inShape: StaticShape, redAxes: number[], keepdims01: 0 | 1): StaticShape {
    if (keepdims01 === 1) {
        const out = inShape.slice();
        redAxes.forEach((a) => {
            out[a] = 1;
        });
        return out;
    }
    return inShape.filter((_, i) => !redAxes.includes(i));
}

function minSentinel(elemTy: DataType): { kind: "float" | "i32" | "i64"; value: number | bigint } {
    switch (elemTy) {
        case DataType.INT64:
            return { kind: "i64", value: BigInt("-9223372036854775808") };
        case DataType.INT32:
            return { kind: "i32", value: -2147483648 };
        default:
            return { kind: "float", value: -3.4028235e38 };
    }
}
function maxSentinel(elemTy: DataType): { kind: "float" | "i32" | "i64"; value: number | bigint } {
    switch (elemTy) {
        case DataType.INT64:
            return { kind: "i64", value: BigInt("9223372036854775807") };
        case DataType.INT32:
            return { kind: "i32", value: 2147483647 };
        default:
            return { kind: "float", value: 3.4028235e38 };
    }
}

// Build a TensorProto filled with a scalar value (size = product(dims))
function filledTensor(elemTy: DataType, dims: number[], scalar: number | bigint): TensorProto {
    const size = dims.reduce((a, b) => a * (b > 0 ? b : 1), 1);
    switch (elemTy) {
        case DataType.INT64:
            return {
                dataType: elemTy,
                dims,
                int64Data: Array(size).fill(
                    typeof scalar === "bigint" ? scalar : BigInt(Math.trunc(Number(scalar))),
                ),
            };
        case DataType.INT32:
            return {
                dataType: elemTy,
                dims,
                int32Data: Array(size).fill(Math.trunc(Number(scalar))),
            };
        default:
            return {
                dataType: elemTy,
                dims,
                floatData: Array(size).fill(Number(scalar)),
            };
    }
}

const _SUPPORTED = new Set([
    "ReduceSum",
    "ReduceMax",
    "ReduceMin",
    "ReduceProd",
    "ReduceMean",
    "ReduceSumSquare",
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
]);

const REDUCE_SET = new Set([
    "ReduceSum",
    "ReduceMax",
    "ReduceMin",
    "ReduceProd",
    "ReduceMean",
    "ReduceSumSquare",
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
]);

export default class ReducesBuilder implements LoopBuilder {
    canHandle(chain: OperationNode.Class[]): boolean {
        // Only handle if the FIRST node is a Reduce op.
        if (!chain.length) return false;
        return REDUCE_SET.has(chain[0].type);
    }

    build(
        chain: OperationNode.Class[],
        outer: OnnxGraph.Class,
        _opts: { fuse: boolean; recurse: boolean; coalesce: boolean },
    ): BuildResult {
        const ridx = chain.findIndex((n) => REDUCE_SET.has(n.type));
        if (ridx < 0) throw new Error("[Reduces] no Reduce op in chain");
        const op = chain[ridx]; // anchor reduce node

        const xInputRaw = op.getInputs()![0];
        const xInput = asValueNode(xInputRaw);

        inferShapes(outer);

        let elemTy = xInput.literalType;
        if (elemTy === DataType.UNDEFINED) elemTy = DataType.FLOAT;

        // Static input shape is mandatory for this lowering
        const inShape = toStaticShape(xInput.shape as Shape);
        if (inShape.some((d) => d === -1)) {
            throw new Error(`[ReducesBuilder] dynamic input shapes not supported for ${op.id}`);
        }
        const rank = inShape.length;

        let axesFromInput: number[] | undefined;
        const inputsRaw = op.getInputs();
        if (inputsRaw && inputsRaw.length > 1) {
            const axesNode = inputsRaw[1];
            axesFromInput = readConstIntegerVectorFromTensorNode(axesNode);
        }

        const atts = op.getAttributes();
        const axesAttr: number[] | undefined =
            axesFromInput ??
            (Array.isArray(atts["axes"])
                ? atts["axes"].map(Number)
                : typeof atts["axes"] === "number"
                  ? [Number(atts["axes"])]
                  : undefined);
        const keepAttr = atts["keepdims"];
        const keep01: 0 | 1 = Number(keepAttr) === 1 ? 1 : 0;

        // If axes/keepdims both missing, infer from the input vs *expected* out shape later;
        // we’ll compute the out shape ourselves, so we don’t need the op’s outgoer at all.
        const redAxes = normalizeAxes(axesAttr ?? [], rank);

        // If axes were empty, ONNX default is "reduce over all axes"
        const effAxes = redAxes.length ? redAxes : [...Array(rank).keys()];
        const keptAxes = [...Array(rank).keys()].filter((a) => !effAxes.includes(a));

        // Compute final static output shape from inShape/axes/keepdims
        const outStatic = makeOutShape(inShape, effAxes, keep01);

        // We no longer read the reduce op's outgoer here (it can be absent in a chain)
        let outTensor = op.getOutgoers.targets.filterIs(TensorNode).first(); // optional

        // Trip count & carry length
        const totalIters = inShape.reduce((a, b) => a * (b > 0 ? b : 1), 1);
        const carryLen = Math.max(
            1,
            outStatic.length ? outStatic.reduce((a, b) => a * (b > 0 ? b : 1), 1) : 1,
        );

        // ------------- Body graph -------------
        const body = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);

        // iter, cond, carry
        const iter = body
            .addNode(uniq(body, "iter"))
            .init(new TensorNode.Builder(DataType.INT64, [], "input"))
            .as(TensorNode);

        body.addNode(uniq(body, "cond_in"))
            .init(new TensorNode.Builder(DataType.BOOL, [], "input"))
            .as(TensorNode);

        // Carry buffer (flat out)
        const carry = body
            .addNode(uniq(body, "carry"))
            .init(new TensorNode.Builder(elemTy, [carryLen], "input"))
            .as(TensorNode);

        // Axes const [0] (for Unsqueeze/GatherElements)
        const axes0 = makeTensorConst(body, `axes_${op.id}`, int64Vec([0]));

        // Optional mean scale (per-iter contribution) → 1 / reduce_count
        let meanScale: ConcreteValueNode | undefined = undefined;
        if (op.type === "ReduceMean") {
            const reduceCount =
                redAxes.length === 0
                    ? 1
                    : redAxes.map((a) => inShape[a]).reduce((p, d) => p * (d > 0 ? d : 1), 1);
            meanScale = makeTensorConst(
                body,
                `rd_mean_scale_${op.id}`,
                scalarFloat(1.0 / Math.max(1, reduceCount)),
            );
        }

        const ctx: LoopCtx = {
            opMap: new Map(),
            iter,
            unsqIdx: null,
            carry,
            axes: axes0,
            outShape: outStatic.length ? outStatic : [],
            coalesce: false,
            // extra for handler
            meanScale,
        };

        // Resolve/flatten X
        const X = resolveFusedInput(
            body,
            xInput,
            ctx,
            op,
            /*flatten*/ false,
            /*returnGather*/ false,
        );
        const Xflat = ensureFlatInput(body, X);

        // Input linear index from iter
        const inDigits = decodeMixedRadix(
            body,
            iter,
            inShape.map((d) => (d > 0 ? d : 1)),
            `rd_${op.id}`,
        );
        const inStrides = computeStrides(inShape);
        const inLin = buildLinearIndex(body, inDigits, inStrides, `rd_inlin_${op.id}`);
        const inLinU = unsqueezeIdx(body, inLin, axes0, `rd_inlinU_${op.id}`); // [1]

        // x scalar: GatherElements(Xflat, inLinU) → [1] → squeeze → []
        const gX = body
            .addNode(uniq(body, `gather_x_${op.id}`))
            .init(new OperationNode.Builder("Gather", [Xflat, inLinU], { axis: 0 }))
            .as(OperationNode);
        const gXOut = body
            .addNode(uniq(body, `gather_x_out_${op.id}`))
            .init(new TensorNode.Builder(elemTy, [1], "intermediate"))
            .as(TensorNode);
        body.addEdge(gX, gXOut)
            .init(new OnnxEdge.Builder(gXOut.literalType, gXOut.shape))
            .as(OnnxEdge);
        const xScalar = squeezeIfLen1(body, gXOut, axes0, `x_sq_${op.id}`); // []

        // ---- build output linear index for this input position ----
        const oDigits: ConcreteValueNode[] = [];
        if (outStatic.length === 0) {
            const zeroIdx = makeTensorConst(body, `rd_out_zero_${op.id}`, scalarInt64(0));
            oDigits.push(zeroIdx);
        } else {
            for (let ax = 0; ax < rank; ax++) {
                if (keptAxes.includes(ax)) {
                    oDigits.push(inDigits[ax]);
                } else if (keep01 === 1) {
                    const z = makeTensorConst(body, `rd_zero_${op.id}_${ax}`, scalarInt64(0));
                    oDigits.push(z);
                }
            }
        }
        const outDimsForStrides = (outStatic.length ? outStatic : [1]).map((d) => (d > 0 ? d : 1));
        const outStrides = computeStrides(outDimsForStrides);
        const outLin = outStatic.length
            ? buildLinearIndex(body, oDigits, outStrides, `rd_outlin_${op.id}`)
            : makeTensorConst(body, `rd_outlin_scalar_${op.id}`, scalarInt64(0));
        const outLinU = unsqueezeIdx(body, outLin, axes0, `rd_outlinU_${op.id}`); // [1]

        // acc scalar: GatherElements(carry, outLinU) → [1] → squeeze → []
        const gAcc = body
            .addNode(uniq(body, `acc_prev_${op.id}`))
            .init(new OperationNode.Builder("Gather", [carry, outLinU], { axis: 0 }))
            .as(OperationNode);
        const gAccOut = body
            .addNode(uniq(body, `acc_prev_out_${op.id}`))
            .init(new TensorNode.Builder(elemTy, [1], "intermediate"))
            .as(TensorNode);
        body.addEdge(gAcc, gAccOut)
            .init(new OnnxEdge.Builder(gAccOut.literalType, gAccOut.shape))
            .as(OnnxEdge);
        const accScalar = squeezeIfLen1(body, gAccOut, axes0, `acc_sq_${op.id}`); // []

        // combine scalar (Sum/Max/Min/Prod/L1/L2/SumSq/Mean/LogSum/LogSumExp)
        const combinedScalar = handleReduceElem(op, body, ctx, accScalar, xScalar); // []

        // Unsqueeze to [1] so ScatterElements ranks match
        const combinedU = unsqueezeIdx(body, combinedScalar, axes0, `combined_unsq_${op.id}`); // [1]

        // use the Unsqueeze output directly
        const lastOut = combinedU; // [1]
        const indicesOut = outLinU; // [1]

        // shapes before outer loop
        inferShapes(outer);

        // ------------- Outer Loop & v_initial -------------
        const trip = makeTensorConst(outer, `trip_count_${op.id}`, scalarInt64(totalIters));
        const cond = makeTensorConst(outer, `cond_${op.id}`, bool(true));

        // v_initial: identity per op
        let initTensor: TensorProto;
        switch (op.type) {
            case "ReduceSum":
            case "ReduceMean":
            case "ReduceSumSquare":
            case "ReduceL1":
            case "ReduceL2":
            case "ReduceLogSum":
            case "ReduceLogSumExp":
                initTensor = filledTensor(elemTy, [carryLen], 0);
                break;
            case "ReduceProd":
                initTensor = filledTensor(elemTy, [carryLen], 1);
                break;
            case "ReduceMax":
                {
                    const s = minSentinel(elemTy);
                    initTensor = filledTensor(
                        elemTy,
                        [carryLen],
                        s.kind === "i64" ? (s.value as bigint) : (s.value as number),
                    );
                }
                break;
            case "ReduceMin":
                {
                    const s = maxSentinel(elemTy);
                    initTensor = filledTensor(
                        elemTy,
                        [carryLen],
                        s.kind === "i64" ? (s.value as bigint) : (s.value as number),
                    );
                }
                break;
            default:
                initTensor = filledTensor(elemTy, [carryLen], 0);
        }

        const v_initial = makeTensorConst(outer, `rd_init_out_${op.id}`, initTensor);

        // run shape inference to align with other builders
        inferShapes(outer);
        inferShapes(body);
        const finalShape = outStatic.length ? outStatic : [];

        // Ensure we always have an outer output tensor node for this reduce chain
        if (!outTensor) {
            const shapeForOut = finalShape;
            outTensor = outer
                .addNode(uniq(outer, `out_${op.id}`))
                .init(new TensorNode.Builder(elemTy, shapeForOut, "intermediate"))
                .as(TensorNode);

            outer
                .addEdge(op, outTensor)
                .init(new OnnxEdge.Builder(elemTy, shapeForOut))
                .as(OnnxEdge);
        }

        return {
            body,
            ctx,
            lastOut,
            indicesOut,
            elemTy,
            outShape: finalShape,
            outTensor,
            trip,
            cond,
            v_initial,
        };
    }
}
