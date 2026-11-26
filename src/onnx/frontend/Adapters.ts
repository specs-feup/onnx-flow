import type {
    RawOnnxGraph,
    RawOnnxModel,
    RawOnnxNode,
    RawOnnxTensorType,
    TensorProto,
} from "../ir/OnnxTypes.js";
import { AttributeType, DataType } from "../ir/OnnxTypes.js";
import { LATEST_ONNX_OPSET } from "../utils/Constants.js";

type MigratedAttrKind = "int" | "float" | "ints";

interface AttrToInputRule {
    attrName: string;
    inputIndex: number;
    kind: MigratedAttrKind;
    /** Optional opset cutoff; rule runs only when `sourceOpset === undefined || sourceOpset < maxOpset`. */
    maxOpset?: number;
}

/**
 * Declarative table of ONNX ops whose attributes migrated to tensor inputs in newer opsets.
 */
const ATTR_TO_INPUT_MIGRATIONS: Partial<Record<string, readonly AttrToInputRule[]>> = {
    Pad: [
        { attrName: "pads", inputIndex: 1, kind: "ints" },
        { attrName: "value", inputIndex: 2, kind: "float" },
    ],
    Clip: [
        { attrName: "min", inputIndex: 1, kind: "float" },
        { attrName: "max", inputIndex: 2, kind: "float" },
    ],
    Slice: [
        { attrName: "starts", inputIndex: 1, kind: "ints" },
        { attrName: "ends", inputIndex: 2, kind: "ints" },
        { attrName: "axes", inputIndex: 3, kind: "ints" },
        { attrName: "steps", inputIndex: 4, kind: "ints" },
    ],
    Squeeze: [{ attrName: "axes", inputIndex: 1, kind: "ints" }],
    Unsqueeze: [{ attrName: "axes", inputIndex: 1, kind: "ints" }],
    Reshape: [{ attrName: "shape", inputIndex: 1, kind: "ints" }],
    Split: [{ attrName: "split", inputIndex: 1, kind: "ints" }],
    Dropout: [{ attrName: "ratio", inputIndex: 1, kind: "float" }],
    TopK: [{ attrName: "k", inputIndex: 1, kind: "int" }],
    ReduceSum: [{ attrName: "axes", inputIndex: 1, kind: "ints", maxOpset: 13 }],
    ReduceMean: [{ attrName: "axes", inputIndex: 1, kind: "ints", maxOpset: 18 }],
    ReduceMax: [{ attrName: "axes", inputIndex: 1, kind: "ints", maxOpset: 18 }],
    ReduceMin: [{ attrName: "axes", inputIndex: 1, kind: "ints", maxOpset: 18 }],
    ReduceProd: [{ attrName: "axes", inputIndex: 1, kind: "ints", maxOpset: 18 }],
    ReduceLogSum: [{ attrName: "axes", inputIndex: 1, kind: "ints", maxOpset: 18 }],
    ReduceLogSumExp: [{ attrName: "axes", inputIndex: 1, kind: "ints", maxOpset: 18 }],
    ReduceSumSquare: [{ attrName: "axes", inputIndex: 1, kind: "ints", maxOpset: 18 }],
    ReduceL1: [{ attrName: "axes", inputIndex: 1, kind: "ints", maxOpset: 18 }],
    ReduceL2: [{ attrName: "axes", inputIndex: 1, kind: "ints", maxOpset: 18 }],
};

/** Legacy attributes stripped when upgrading older opsets. */
const DEPRECATED_ATTRS_BY_OP: Partial<Record<string, readonly string[]>> = {
    Dropout: ["is_test"],
    BatchNormalization: ["spatial"],
};

const PRE7_BROADCAST_OPS = new Set([
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "And",
    "Or",
    "Xor",
    "Equal",
    "Greater",
    "Less",
    "Min",
    "Max",
    "PRelu",
]);

const SOFTMAX_FAMILY = new Set(["Softmax", "LogSoftmax", "Hardmax"]);

/**
 * Frontend normalizer that upgrades legacy ONNX model protos to modern opset
 * semantics before ingestion into `OnnxGraph` by `OnnxGraphLoader`.
 *
 * Handles overridable initializers, attribute-to-input migrations across opsets,
 * `Upsample`/pre-11 `Resize`, pre-7 `broadcast`/`axis`, and pre-13 `Softmax` flattening.
 */
export class OnnxOpsetAdapter {
    /** Removes initializers from `graph.input` so constant weights are never overwritten as inputs. */
    public static freezeGraphOverridableInputs(graphProto: RawOnnxGraph): void {
        if (!graphProto.input || !graphProto.initializer) return;
        const initializerNames = new Set(graphProto.initializer.map((init) => init.name));
        graphProto.input = graphProto.input.filter((input) => !initializerNames.has(input.name));
    }

    /** Upgrades `data` in place to modern ONNX conventions. */
    public static adaptModel(data: RawOnnxModel, targetOpset: number = LATEST_ONNX_OPSET): void {
        if (!data.graph) return;
        const sourceOpset = OnnxOpsetAdapter.resolveSourceOpset(data);
        OnnxOpsetAdapter.adaptGraph(data.graph, sourceOpset, targetOpset);
    }

    private static createInt64Initializer(
        name: string,
        values: (number | string | bigint)[],
    ): TensorProto {
        return {
            name,
            dataType: DataType.INT64,
            dims: [values.length],
            int64Data: values.map((v) => BigInt(v)),
        };
    }

    private static createFloatInitializer(name: string, value: number): TensorProto {
        return {
            name,
            dataType: DataType.FLOAT,
            dims: [],
            floatData: [value],
        };
    }

    private static getOpType(node: RawOnnxNode): string | undefined {
        return node.opType ?? node.op_type;
    }

    private static createNameAllocator(graphProto: RawOnnxGraph): (prefix: string) => string {
        const used = new Set<string>();
        for (const init of graphProto.initializer ?? []) {
            if (init.name !== undefined && init.name !== "") used.add(init.name);
        }
        for (const vi of [
            ...(graphProto.input ?? []),
            ...(graphProto.output ?? []),
            ...(graphProto.valueInfo ?? graphProto.value_info ?? []),
        ]) {
            if (vi.name !== "") used.add(vi.name);
        }
        for (const node of graphProto.node ?? []) {
            for (const name of [...(node.input ?? []), ...(node.output ?? [])]) {
                if (name !== "") used.add(name);
            }
        }

        let counter = 0;
        return (prefix: string): string => {
            const sanitized = prefix.replace(/[^A-Za-z0-9_]/g, "_");
            let candidate = `${sanitized}_${counter++}`;
            while (used.has(candidate)) {
                candidate = `${sanitized}_${counter++}`;
            }
            used.add(candidate);
            return candidate;
        };
    }

    private static resolveTensorRank(
        graphProto: RawOnnxGraph,
        tensorName: string,
    ): number | undefined {
        const init = graphProto.initializer?.find((i) => i.name === tensorName);
        if (init?.dims) return init.dims.length;

        const allValueInfos = [
            ...(graphProto.input ?? []),
            ...(graphProto.valueInfo ?? graphProto.value_info ?? []),
            ...(graphProto.output ?? []),
        ];
        for (const vi of allValueInfos) {
            if (vi.name !== tensorName) continue;
            const tensorType: RawOnnxTensorType | undefined =
                vi.type?.tensorType ?? vi.type?.tensor_type;
            if (tensorType?.shape?.dim) {
                return tensorType.shape.dim.length;
            }
        }
        return undefined;
    }

    private static resolveSourceOpset(data: RawOnnxModel): number | undefined {
        const opsets = data.opsetImport ?? data.opset_import;
        if (!opsets || opsets.length === 0) return undefined;
        const defaultEntry = opsets.find((o) => (o.domain ?? "") === "" || o.domain === "ai.onnx");
        if (!defaultEntry || defaultEntry.version === undefined) return undefined;
        return Number(defaultEntry.version);
    }

    private static moveAttributeToInput(
        node: RawOnnxNode,
        graphProto: RawOnnxGraph,
        rule: AttrToInputRule,
        nextName: (prefix: string) => string,
    ): void {
        const idx = node.attribute?.findIndex((a) => a.name === rule.attrName);
        if (idx === undefined || idx === -1) return;

        const attr = node.attribute![idx];
        const opType = OnnxOpsetAdapter.getOpType(node) ?? "UnknownOp";
        const name = nextName(`${node.name ?? opType}_${rule.attrName}`);

        let init: TensorProto;
        if (rule.kind === "ints") {
            init = OnnxOpsetAdapter.createInt64Initializer(name, attr.ints ?? [0]);
        } else if (rule.kind === "int") {
            init = OnnxOpsetAdapter.createInt64Initializer(name, [Number(attr.i ?? 0)]);
        } else {
            init = OnnxOpsetAdapter.createFloatInitializer(name, Number(attr.f ?? 0));
        }

        graphProto.initializer ??= [];
        graphProto.initializer.push(init);

        node.input ??= [];
        while (node.input.length < rule.inputIndex) node.input.push("");
        node.input[rule.inputIndex] = name;

        node.attribute?.splice(idx, 1);
    }

    private static applyTableMigrations(
        node: RawOnnxNode,
        graph: RawOnnxGraph,
        sourceOpset: number | undefined,
        nextName: (prefix: string) => string,
    ): void {
        const opType = OnnxOpsetAdapter.getOpType(node);
        if (opType === undefined) return;

        const rules = ATTR_TO_INPUT_MIGRATIONS[opType];
        if (rules !== undefined) {
            for (const rule of rules) {
                if (
                    rule.maxOpset === undefined ||
                    sourceOpset === undefined ||
                    sourceOpset < rule.maxOpset
                ) {
                    OnnxOpsetAdapter.moveAttributeToInput(node, graph, rule, nextName);
                }
            }
        }

        const deprecated = DEPRECATED_ATTRS_BY_OP[opType];
        if (deprecated !== undefined && node.attribute !== undefined) {
            for (const attrName of deprecated) {
                const idx = node.attribute.findIndex((a) => a.name === attrName);
                if (idx !== -1) node.attribute.splice(idx, 1);
            }
        }
    }

    /** Upgrades `Upsample` (Opset 7/9) and pre-11 `Resize` to Opset 11+ `[X, roi, scales, sizes]`. */
    private static adaptResize(
        node: RawOnnxNode,
        graph: RawOnnxGraph,
        sourceOpset: number | undefined,
        nextName: (prefix: string) => string,
    ): void {
        const origOpType = OnnxOpsetAdapter.getOpType(node);
        const isUpsample = origOpType === "Upsample";
        const isLegacyResize =
            origOpType === "Resize" && sourceOpset !== undefined && sourceOpset < 11;
        if (!isUpsample && origOpType !== "Resize") return;

        if (isUpsample) {
            node.opType = "Resize";
            if ("op_type" in node) delete node.op_type;
        }

        const scalesAttrIdx = node.attribute?.findIndex((a) => a.name === "scales") ?? -1;
        if (scalesAttrIdx !== -1 && node.attribute) {
            const attr = node.attribute[scalesAttrIdx];
            const vals =
                attr.floats && attr.floats.length > 0
                    ? attr.floats
                    : attr.f !== undefined
                      ? [Number(attr.f)]
                      : [];
            const name = nextName(`${node.name ?? "Resize"}_scales`);
            const init: TensorProto = {
                name,
                dataType: DataType.FLOAT,
                dims: [vals.length],
                floatData: vals,
            };

            graph.initializer ??= [];
            graph.initializer.push(init);

            node.input ??= [];
            while (node.input.length < 3) node.input.push("");
            node.input[2] = name;
            node.attribute.splice(scalesAttrIdx, 1);
        } else if ((isUpsample || isLegacyResize) && node.input && node.input.length === 2) {
            const scalesInput = node.input[1];
            if (scalesInput !== "") {
                node.input[1] = "";
                node.input[2] = scalesInput;
            }
        }

        if (isUpsample || isLegacyResize) {
            node.attribute ??= [];
            const hasCoordMode = node.attribute.some(
                (a) => a.name === "coordinate_transformation_mode",
            );
            if (!hasCoordMode) {
                node.attribute.push({
                    name: "coordinate_transformation_mode",
                    type: AttributeType.STRING,
                    s: "asymmetric",
                });
            }
        }
    }

    /** Upgrades pre-Opset 7 binary `broadcast=1` + `axis` to explicit `Unsqueeze` + numpy broadcast. */
    private static adaptLegacyBroadcast(
        node: RawOnnxNode,
        graph: RawOnnxGraph,
        sourceOpset: number | undefined,
        nextName: (prefix: string) => string,
    ): RawOnnxNode[] | undefined {
        const opType = OnnxOpsetAdapter.getOpType(node);
        if (opType === undefined || !PRE7_BROADCAST_OPS.has(opType)) return undefined;
        if (sourceOpset !== undefined && sourceOpset >= 7) return undefined;
        if (!node.attribute || node.attribute.length === 0) return undefined;

        const bIdx = node.attribute.findIndex((a) => a.name === "broadcast");
        if (bIdx === -1) return undefined;

        const isLegacyBroadcast = Number(node.attribute[bIdx].i ?? 0) === 1;
        node.attribute.splice(bIdx, 1);

        const aIdx = node.attribute.findIndex((a) => a.name === "axis");
        let axisVal: number | undefined;
        if (aIdx !== -1) {
            axisVal = Number(node.attribute[aIdx].i ?? 0);
            node.attribute.splice(aIdx, 1);
        }

        if (!isLegacyBroadcast || axisVal === undefined || !node.input || node.input.length < 2) {
            return undefined;
        }

        const rankA = OnnxOpsetAdapter.resolveTensorRank(graph, node.input[0]);
        const rankB = OnnxOpsetAdapter.resolveTensorRank(graph, node.input[1]);
        if (rankA === undefined || rankB === undefined) return undefined;

        const normalizedAxis = axisVal < 0 ? rankA + axisVal : axisVal;
        const trailingOnes = rankA - (normalizedAxis + rankB);
        if (trailingOnes <= 0) return undefined;

        const unsqueezeAxes: number[] = [];
        for (let k = 0; k < trailingOnes; k++) {
            unsqueezeAxes.push(rankB + k);
        }

        const axesName = nextName(`${node.name ?? opType}_bcast_axes`);
        graph.initializer ??= [];
        graph.initializer.push(OnnxOpsetAdapter.createInt64Initializer(axesName, unsqueezeAxes));

        const unsqueezedName = nextName(`${node.input[1]}_bcast`);
        const unsqueezeNode: RawOnnxNode = {
            name: nextName(`${node.name ?? opType}_Unsqueeze`),
            opType: "Unsqueeze",
            input: [node.input[1], axesName],
            output: [unsqueezedName],
        };

        node.input[1] = unsqueezedName;
        return [unsqueezeNode, node];
    }

    /** Upgrades pre-Opset 13 flattened `Softmax`/`LogSoftmax`/`Hardmax` to `Flatten -> Op -> Reshape`. */
    private static adaptLegacySoftmax(
        node: RawOnnxNode,
        graph: RawOnnxGraph,
        sourceOpset: number | undefined,
        nextName: (prefix: string) => string,
    ): RawOnnxNode[] | undefined {
        const opType = OnnxOpsetAdapter.getOpType(node);
        if (opType === undefined || !SOFTMAX_FAMILY.has(opType)) return undefined;
        if (sourceOpset === undefined || sourceOpset >= 13) return undefined;
        if (
            !node.input ||
            node.input.length === 0 ||
            node.input[0] === "" ||
            !node.output ||
            node.output.length === 0 ||
            node.output[0] === ""
        ) {
            return undefined;
        }

        node.attribute ??= [];
        const axisAttr = node.attribute.find((a) => a.name === "axis");
        const legacyAxis =
            axisAttr !== undefined && axisAttr.i !== undefined ? Number(axisAttr.i) : 1;

        const inputName = node.input[0];
        const outputName = node.output[0];
        const rank = OnnxOpsetAdapter.resolveTensorRank(graph, inputName);

        if (rank !== undefined && rank >= 1) {
            const normalizedAxis = legacyAxis < 0 ? rank + legacyAxis : legacyAxis;
            if (normalizedAxis === rank - 1) {
                if (axisAttr) {
                    axisAttr.i = -1;
                } else {
                    node.attribute.push({ name: "axis", type: AttributeType.INT, i: -1 });
                }
                return undefined;
            }
        }

        const base = node.name ?? opType;
        const shapeOut = nextName(`${base}_orig_shape`);
        const flatOut = nextName(`${base}_flat`);
        const smOut = nextName(`${base}_sm2d`);

        return [
            {
                name: nextName(`${base}_Shape`),
                opType: "Shape",
                input: [inputName],
                output: [shapeOut],
            },
            {
                name: nextName(`${base}_Flatten`),
                opType: "Flatten",
                input: [inputName],
                output: [flatOut],
                attribute: [{ name: "axis", type: AttributeType.INT, i: legacyAxis }],
            },
            {
                name: node.name ?? nextName(`${base}_sm2d`),
                opType,
                input: [flatOut],
                output: [smOut],
                attribute: [{ name: "axis", type: AttributeType.INT, i: 1 }],
            },
            {
                name: nextName(`${base}_Reshape`),
                opType: "Reshape",
                input: [smOut, shapeOut],
                output: [outputName],
            },
        ];
    }

    private static adaptGraph(
        graph: RawOnnxGraph,
        sourceOpset: number | undefined,
        targetOpset: number,
    ): void {
        OnnxOpsetAdapter.freezeGraphOverridableInputs(graph);
        if (!graph.node || graph.node.length === 0) return;

        const nextName = OnnxOpsetAdapter.createNameAllocator(graph);
        const updatedNodes: RawOnnxNode[] = [];

        for (const node of graph.node) {
            for (const attr of node.attribute ?? []) {
                if (attr.g != null) {
                    OnnxOpsetAdapter.adaptGraph(attr.g, sourceOpset, targetOpset);
                }
                if (attr.graphs !== undefined) {
                    for (const subGraph of attr.graphs) {
                        OnnxOpsetAdapter.adaptGraph(subGraph, sourceOpset, targetOpset);
                    }
                }
            }

            OnnxOpsetAdapter.applyTableMigrations(node, graph, sourceOpset, nextName);
            OnnxOpsetAdapter.adaptResize(node, graph, sourceOpset, nextName);

            const bcastExpanded = OnnxOpsetAdapter.adaptLegacyBroadcast(
                node,
                graph,
                sourceOpset,
                nextName,
            );
            if (bcastExpanded !== undefined) {
                updatedNodes.push(...bcastExpanded);
                continue;
            }

            const softmaxExpanded = OnnxOpsetAdapter.adaptLegacySoftmax(
                node,
                graph,
                sourceOpset,
                nextName,
            );
            if (softmaxExpanded !== undefined) {
                updatedNodes.push(...softmaxExpanded);
                continue;
            }

            updatedNodes.push(node);
        }

        graph.node = updatedNodes;
    }
}

/** Strips initializer names from `data.graph.input` so constant weights stay `ConstantNode`s. */
export function freezeOverridableInputs(data: RawOnnxModel): void {
    if (!data.graph) return;
    OnnxOpsetAdapter.freezeGraphOverridableInputs(data.graph);
}

/** Main frontend normalization entry point upgrading legacy ONNX constructs to modern opsets. */
export function applyAdapters(data: RawOnnxModel, targetOpset: number = LATEST_ONNX_OPSET): void {
    OnnxOpsetAdapter.adaptModel(data, targetOpset);
}
