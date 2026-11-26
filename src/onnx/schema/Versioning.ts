import type OnnxGraph from "../OnnxGraph.js";
import type OperationNode from "../OperationNode.js";
import { DataType } from "../OnnxTypes.js";
import { LATEST_ONNX_IR_VERSION, LATEST_ONNX_OPSET } from "../utils/Constants.js";
import { OpRegistry } from "./OpRegistry.js";

/**
 * Default minimum versions exported when a graph only uses ops/dtypes valid at
 * or below Opset 20 / IR 9 so exported models remain maximally portable across runtimes.
 */
export const DEFAULT_COMPAT_IR_VERSION = 9;
export const DEFAULT_COMPAT_OPSET = 20;

/**
 * Returns the minimum ONNX IR version required to encode a TensorProto.DataType.
 */
export function getDataTypeMinIrVersion(dtype: DataType): number {
    if (dtype === DataType.FLOAT6E2M3 || dtype === DataType.FLOAT6E3M2) return 14;
    if (dtype === DataType.UINT2 || dtype === DataType.INT2) return 13;
    if (dtype === DataType.FLOAT8E8M0) return 12;
    if (dtype === DataType.FLOAT4E2M1) return 11;
    if (dtype === DataType.UINT4 || dtype === DataType.INT4) return 10;
    if (
        dtype === DataType.FLOAT8E4M3FN ||
        dtype === DataType.FLOAT8E4M3FNUZ ||
        dtype === DataType.FLOAT8E5M2 ||
        dtype === DataType.FLOAT8E5M2FNUZ
    ) {
        return 9;
    }
    return 4;
}

/**
 * Maps an ONNX ai.onnx opset version to the minimum ONNX IR version that introduced/paired with it.
 */
export function getOpsetMinIrVersion(opset: number): number {
    if (opset >= 27) return 14;
    if (opset >= 25) return 13;
    if (opset === 24) return 12;
    if (opset >= 22) return 11;
    if (opset === 21) return 10;
    if (opset >= 19) return 9;
    if (opset >= 15) return 8;
    if (opset >= 12) return 7;
    if (opset >= 10) return 5;
    if (opset >= 9) return 4;
    return 3;
}

function inferMinOpsetForOp(op: OperationNode.Class, registry: OpRegistry): number {
    const minSince = registry.getMinSinceVersion(op.type);
    let minOpset = minSince !== undefined ? minSince : 1;

    const attrs = op.getAttributes();
    const ins = op.getInputs() ?? [];

    // Feature/attribute bumps within existing op names (Opsets 21..28)
    if (op.type === "QuantizeLinear" || op.type === "DequantizeLinear") {
        if ("block_size" in attrs) {
            minOpset = Math.max(minOpset, 21);
        }
        if ("output_dtype" in attrs) {
            minOpset = Math.max(minOpset, op.type === "DequantizeLinear" ? 23 : 21);
        }
        if ("precision" in attrs) {
            minOpset = Math.max(minOpset, 23);
        }
    } else if (op.type === "GroupNormalization") {
        if ("stash_type" in attrs) {
            minOpset = Math.max(minOpset, 21);
        }
        // Channel-last / scalar-bias GroupNormalization in Opset 21 vs Opset 18
        if (ins.length >= 2 && (ins[1].shape as unknown[] | undefined)?.length === 1) {
            minOpset = Math.max(minOpset, 18);
        }
    } else if (op.type === "Attention") {
        if ("local_window_size" in attrs) {
            minOpset = Math.max(minOpset, 25);
        }
    } else if (op.type === "QLinearMatMul" || op.type === "MatMulInteger") {
        if (ins.some((n) => n.literalType === DataType.UINT4 || n.literalType === DataType.INT4)) {
            minOpset = Math.max(minOpset, 21);
        }
    } else if (op.type === "Cast" || op.type === "CastLike") {
        const toAttr = attrs["to"] as DataType | undefined;
        if (
            toAttr === DataType.FLOAT6E2M3 ||
            toAttr === DataType.FLOAT6E3M2 ||
            toAttr === DataType.UINT2 ||
            toAttr === DataType.INT2
        ) {
            minOpset = Math.max(minOpset, 26);
        } else if (toAttr === DataType.FLOAT4E2M1 || toAttr === DataType.FLOAT8E8M0) {
            minOpset = Math.max(minOpset, 23);
        } else if (toAttr === DataType.UINT4 || toAttr === DataType.INT4) {
            minOpset = Math.max(minOpset, 21);
        }
    }

    // Dtypes inside op I/O may require newer opsets
    const nodes = [...ins, ...op.getOutputs()];
    for (const n of nodes) {
        if (n.literalType === DataType.FLOAT6E2M3 || n.literalType === DataType.FLOAT6E3M2) {
            minOpset = Math.max(minOpset, 27);
        } else if (n.literalType === DataType.UINT2 || n.literalType === DataType.INT2) {
            minOpset = Math.max(minOpset, 26);
        } else if (n.literalType === DataType.FLOAT8E8M0) {
            minOpset = Math.max(minOpset, 24);
        } else if (n.literalType === DataType.FLOAT4E2M1) {
            minOpset = Math.max(minOpset, 23);
        } else if (n.literalType === DataType.UINT4 || n.literalType === DataType.INT4) {
            minOpset = Math.max(minOpset, 21);
        }
    }

    return Math.min(minOpset, LATEST_ONNX_OPSET);
}

/**
 * Recursively computes the minimum required `(irVersion, opsetVersion)` for `graph`
 * while respecting any explicit `targetOpset` or `irVersion` requested on the graph or via `override`.
 */
export function resolveExportVersions(
    graph: OnnxGraph.Class,
    override?: { targetOpset?: number; irVersion?: number },
): { irVersion: number; opsetVersion: number } {
    const registry = OpRegistry.getInstance();
    let reqOpset = DEFAULT_COMPAT_OPSET;
    let reqIr = DEFAULT_COMPAT_IR_VERSION;

    const visit = (g: OnnxGraph.Class) => {
        for (const t of g.getTensorNodes()) {
            reqIr = Math.max(reqIr, getDataTypeMinIrVersion(t.literalType));
        }
        for (const c of g.getConstantNodes()) {
            reqIr = Math.max(reqIr, getDataTypeMinIrVersion(c.literalType));
        }
        for (const op of g.getOperationNodes()) {
            reqOpset = Math.max(reqOpset, inferMinOpsetForOp(op, registry));
            for (const region of op.regions) {
                visit(region);
            }
        }
    };
    visit(graph);

    const sourceOpset = graph.getSourceOpset();
    const baseOpset =
        sourceOpset !== undefined
            ? Math.min(LATEST_ONNX_OPSET, Math.max(DEFAULT_COMPAT_OPSET, sourceOpset))
            : DEFAULT_COMPAT_OPSET;

    const explicitOpset =
        override?.targetOpset ??
        (graph.hasExplicitTargetOpset() ? graph.getTargetOpset() : undefined);
    const opsetVersion = Math.min(
        LATEST_ONNX_OPSET,
        explicitOpset !== undefined
            ? Math.max(reqOpset, explicitOpset)
            : Math.max(reqOpset, baseOpset),
    );

    reqIr = Math.max(reqIr, getOpsetMinIrVersion(opsetVersion));

    const explicitIr =
        override?.irVersion ?? (graph.hasExplicitIrVersion() ? graph.getIrVersion() : undefined);
    const irVersion = Math.min(LATEST_ONNX_IR_VERSION, Math.max(reqIr, explicitIr ?? reqIr));

    return { irVersion, opsetVersion };
}
