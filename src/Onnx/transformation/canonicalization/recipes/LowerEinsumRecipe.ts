import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import {
    asConcreteValueNode,
    int64Vec,
    makeTensorProto,
    scalarInt64,
    toStaticShape,
} from "../../../Utils.js";
import { extractMixedDims, mulMixed, packShapeVector } from "../../loop-lowering/RecipeUtils.js";

function readStringAttr(op: OperationNode.Class, name: string): string | undefined {
    const raw = op.attributes[name];
    if (typeof raw === "string") return raw;
    if (Buffer.isBuffer(raw)) return raw.toString("utf8");
    if (raw instanceof Uint8Array) return Buffer.from(raw).toString("utf8");
    return undefined;
}

interface ParsedEinsum {
    inLabels: string[][];
    outLabels: string[];
    labelSizes: Map<string, number>;
}

function parseEquation(equation: string, shapes: number[][]): ParsedEinsum | undefined {
    const clean = equation.replace(/\s+/g, "");
    const parts = clean.split("->");
    if (parts.length > 2) return undefined;

    const lhsTerms = parts[0].split(",");
    if (lhsTerms.length !== shapes.length) return undefined;

    // 1. Determine broadcast ellipsis rank
    let maxEllRank = 0;
    const hasEllipsis = lhsTerms.some((t) => t.includes("..."));
    for (let i = 0; i < lhsTerms.length; i++) {
        const term = lhsTerms[i];
        const ellParts = term.split("...");
        if (ellParts.length > 2) return undefined;
        if (ellParts.length === 2) {
            const explicitCount = ellParts[0].length + ellParts[1].length;
            const ellRank = shapes[i].length - explicitCount;
            if (ellRank < 0) return undefined;
            maxEllRank = Math.max(maxEllRank, ellRank);
        } else if (term.length !== shapes[i].length) {
            return undefined;
        }
    }

    const ellLabels = Array.from({ length: maxEllRank }, (_, idx) => `__e${idx}`);

    const expandTerm = (term: string, rank: number): string[] => {
        const ellParts = term.split("...");
        if (ellParts.length === 1) return [...term];
        const pre = [...ellParts[0]];
        const post = [...ellParts[1]];
        const ellRank = rank - pre.length - post.length;
        return [...pre, ...ellLabels.slice(maxEllRank - ellRank), ...post];
    };

    const inLabels = lhsTerms.map((term, i) => expandTerm(term, shapes[i].length));

    // 2. Map label -> static size (allowing size-1 broadcast and dynamic -1 dimensions)
    const labelSizes = new Map<string, number>();
    for (let i = 0; i < inLabels.length; i++) {
        for (let ax = 0; ax < inLabels[i].length; ax++) {
            const lbl = inLabels[i][ax];
            const dim = shapes[i][ax];
            const prev = labelSizes.get(lbl);
            if (prev === undefined || prev === 1 || (prev === -1 && dim !== 1 && dim !== -1)) {
                labelSizes.set(lbl, dim);
            } else if (dim !== 1 && dim !== -1 && dim !== prev) {
                return undefined;
            }
        }
    }

    // 3. Determine output labels
    let outLabels: string[];
    if (parts.length === 2) {
        const rhs = parts[1];
        const ellParts = rhs.split("...");
        if (ellParts.length > 2) return undefined;
        outLabels =
            ellParts.length === 2 ? [...ellParts[0], ...ellLabels, ...ellParts[1]] : [...rhs];
        // Ensure no duplicate output labels and all exist on LHS
        if (new Set(outLabels).size !== outLabels.length) return undefined;
        for (const lbl of outLabels) {
            if (!labelSizes.has(lbl)) return undefined;
        }
    } else {
        const counts = new Map<string, number>();
        for (const labels of inLabels) {
            for (const lbl of labels) {
                if (!lbl.startsWith("__e")) counts.set(lbl, (counts.get(lbl) ?? 0) + 1);
            }
        }
        const uniqueSorted = [...counts.entries()]
            .filter(([, c]) => c === 1)
            .map(([l]) => l)
            .sort();
        outLabels = hasEllipsis ? [...ellLabels, ...uniqueSorted] : uniqueSorted;
    }

    return { inLabels, outLabels, labelSizes };
}

export class LowerEinsumRecipe implements DecompositionRecipe {
    public readonly name = "LowerEinsum";
    public readonly targetOp = "Einsum";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["contraction"];
    public readonly producedOps = [
        "Shape",
        "Gather",
        "ScatterElements",
        "Unsqueeze",
        "Max",
        "Transpose",
        "Reshape",
        "MatMul",
        "Mul",
        "ReduceSum",
        "Range",
        "Equal",
        "Where",
        "Squeeze",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Einsum") return false;
        const eq = readStringAttr(op, "equation");
        const inputs = op.getInputs();
        if (eq === undefined || inputs === undefined || inputs.length === 0) return false;
        const shapes: number[][] = [];
        for (const inp of inputs) {
            const node = asConcreteValueNode(inp);
            if (!Array.isArray(node.shape)) return false;
            shapes.push(toStaticShape(node.shape));
        }
        return parseEquation(eq, shapes) !== undefined;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const inputs = op.getInputs()!.map((n) => asConcreteValueNode(n));
        const Y = op.getOutputs()[0];
        const dtype = inputs[0].literalType as DataType;
        const shapes = inputs.map((n) => toStaticShape(n.shape));
        const parsed = parseEquation(readStringAttr(op, "equation")!, shapes)!;

        let tag = 0;
        const uid = (p: string) => `ein_${p}_${op.id}_${tag++}`;
        type MixedDim = number | ConcreteValueNode;
        const toKnown = (dims: MixedDim[]): KnownShape =>
            dims.map((d) => (typeof d === "number" ? d : -1)) as KnownShape;
        const maxMixed = (a: MixedDim, b: MixedDim, mTag: string): MixedDim => {
            if (a === b || (typeof a !== "number" && typeof b !== "number" && a.id === b.id)) {
                return a;
            }
            if (a === 1) return b;
            if (b === 1) return a;
            if (typeof a === "number" && typeof b === "number") return Math.max(a, b);
            const aN =
                typeof a === "number" ? builder.createConstant(`${mTag}_a`, scalarInt64(a)) : a;
            const bN =
                typeof b === "number" ? builder.createConstant(`${mTag}_b`, scalarInt64(b)) : b;
            return builder.createOp("Max", [aN, bN])[0];
        };
        const productMixed = (dims: MixedDim[], pTag: string): MixedDim => {
            let acc: MixedDim = 1;
            for (let i = 0; i < dims.length; i++) {
                acc = mulMixed(builder, acc, dims[i], `${pTag}_${i}`);
            }
            return acc;
        };

        const transposeTo = (
            node: ConcreteValueNode,
            curLabels: string[],
            curDims: MixedDim[],
            targetLabels: string[],
        ): { node: ConcreteValueNode; dims: MixedDim[] } => {
            const perm = targetLabels.map((l) => curLabels.indexOf(l));
            if (perm.every((p, i) => p === i)) return { node, dims: curDims };
            const newDims = perm.map((p) => curDims[p]);
            const [t] = builder.createOp("Transpose", [node], { perm }, [
                { type: dtype, shape: toKnown(newDims) },
            ]);
            return { node: t, dims: newDims };
        };

        const reduceAxes = (
            node: ConcreteValueNode,
            curLabels: string[],
            curDims: MixedDim[],
            dropIndices: number[],
        ): { node: ConcreteValueNode; labels: string[]; dims: MixedDim[] } => {
            if (dropIndices.length === 0) return { node, labels: curLabels, dims: curDims };
            const dropSet = new Set(dropIndices);
            const nextLabels = curLabels.filter((_, i) => !dropSet.has(i));
            const nextDims = curDims.filter((_, i) => !dropSet.has(i));
            const axesC = builder.createConstant(uid("rax"), int64Vec(dropIndices));
            const [r] = builder.createOp("ReduceSum", [node, axesC], { keepdims: 0 }, [
                { type: dtype, shape: toKnown(nextDims) },
            ]);
            return { node: r, labels: nextLabels, dims: nextDims };
        };

        // Step 1: Resolve repeated indices (trace/diagonal) inside each operand
        const operands = inputs.map((inp, idx) => {
            let cur = inp;
            let labels = [...parsed.inLabels[idx]];
            let dims: MixedDim[] = extractMixedDims(builder, inp, shapes[idx], uid(`in${idx}`));

            while (new Set(labels).size < labels.length) {
                const seen = new Map<string, number>();
                let a = -1;
                let b = -1;
                for (let i = 0; i < labels.length; i++) {
                    if (seen.has(labels[i])) {
                        a = seen.get(labels[i])!;
                        b = i;
                        break;
                    }
                    seen.set(labels[i], i);
                }
                const dim = typeof dims[a] === "number" ? dims[a] : dims[b];
                const dimNode =
                    typeof dim === "number"
                        ? builder.createConstant(uid("rN"), scalarInt64(dim))
                        : dim;
                const [rVec] = builder.createOp(
                    "Range",
                    [
                        builder.createConstant(uid("r0"), scalarInt64(0)),
                        dimNode,
                        builder.createConstant(uid("r1"), scalarInt64(1)),
                    ],
                    {},
                    [{ type: DataType.INT64, shape: toKnown([dim]) }],
                );
                const shA: MixedDim[] = new Array<MixedDim>(dims.length).fill(1);
                shA[a] = dim;
                const shB: MixedDim[] = new Array<MixedDim>(dims.length).fill(1);
                shB[b] = dim;
                const [rA] = builder.createOp(
                    "Reshape",
                    [rVec, packShapeVector(builder, shA, uid("shA"))],
                    {},
                    [{ type: DataType.INT64, shape: toKnown(shA) }],
                );
                const [rB] = builder.createOp(
                    "Reshape",
                    [rVec, packShapeVector(builder, shB, uid("shB"))],
                    {},
                    [{ type: DataType.INT64, shape: toKnown(shB) }],
                );
                const maskDims = shA.map((v, i) => (i === a || i === b ? dim : v));
                const [mask] = builder.createOp("Equal", [rA, rB], {}, [
                    { type: DataType.BOOL, shape: toKnown(maskDims) },
                ]);
                const zeroC = builder.createConstant(uid("z"), makeTensorProto(dtype, [], [0]));
                const [masked] = builder.createOp("Where", [mask, cur, zeroC], {}, [
                    { type: dtype, shape: toKnown(dims) },
                ]);
                const res = reduceAxes(masked, labels, dims, [b]);
                cur = res.node;
                labels = res.labels;
                dims = res.dims;
            }
            return { node: cur, labels, dims };
        });

        // Step 2: Fold operands left-to-right
        let acc = operands[0];
        for (let idx = 1; idx < operands.length; idx++) {
            const rhs = operands[idx];
            const futureNeeded = new Set<string>(parsed.outLabels);
            for (let j = idx + 1; j < operands.length; j++) {
                for (const l of operands[j].labels) futureNeeded.add(l);
            }

            const lhsSet = new Set(acc.labels);
            const rhsSet = new Set(rhs.labels);
            const batch = acc.labels.filter((l) => rhsSet.has(l) && futureNeeded.has(l));
            const contract = acc.labels.filter((l) => rhsSet.has(l) && !futureNeeded.has(l));
            const left = acc.labels.filter((l) => !rhsSet.has(l));
            const right = rhs.labels.filter((l) => !lhsSet.has(l));

            const exactBatchAndContract =
                contract.length > 0 &&
                [...batch, ...contract].every((l) => {
                    const da = acc.dims[acc.labels.indexOf(l)];
                    const db = rhs.dims[rhs.labels.indexOf(l)];
                    const sa = typeof da === "number" ? da : -1;
                    const sb = typeof db === "number" ? db : -1;
                    return sa === sb;
                });

            if (exactBatchAndContract) {
                const aOrder = [...batch, ...left, ...contract];
                const bOrder = [...batch, ...contract, ...right];
                const aT = transposeTo(acc.node, acc.labels, acc.dims, aOrder);
                const bT = transposeTo(rhs.node, rhs.labels, rhs.dims, bOrder);

                const bDims = batch.map((l, i) =>
                    maxMixed(
                        acc.dims[acc.labels.indexOf(l)],
                        rhs.dims[rhs.labels.indexOf(l)],
                        uid(`bd_${i}`),
                    ),
                );
                const lDims = left.map((l) => acc.dims[acc.labels.indexOf(l)]);
                const rDims = right.map((l) => rhs.dims[rhs.labels.indexOf(l)]);
                const kDim = productMixed(
                    contract.map((l) => acc.dims[acc.labels.indexOf(l)]),
                    uid("kd"),
                );
                const mDim = productMixed(lDims, uid("md"));
                const nDim = productMixed(rDims, uid("nd"));

                const aMatDims: MixedDim[] = [...bDims, mDim, kDim];
                const bMatDims: MixedDim[] = [...bDims, kDim, nDim];
                const mmOutDims: MixedDim[] = [...bDims, mDim, nDim];
                const finalLabels = [...batch, ...left, ...right];
                const finalDims: MixedDim[] = [...bDims, ...lDims, ...rDims];

                const [aR] = builder.createOp(
                    "Reshape",
                    [aT.node, packShapeVector(builder, aMatDims, uid("ash"))],
                    {},
                    [{ type: dtype, shape: toKnown(aMatDims) }],
                );
                const [bR] = builder.createOp(
                    "Reshape",
                    [bT.node, packShapeVector(builder, bMatDims, uid("bsh"))],
                    {},
                    [{ type: dtype, shape: toKnown(bMatDims) }],
                );
                const [mm] = builder.createOp("MatMul", [aR, bR], {}, [
                    { type: dtype, shape: toKnown(mmOutDims) },
                ]);
                let resNode: ConcreteValueNode = mm;
                if (finalDims.length === 0) {
                    const sqAxes = mmOutDims.map((_, i) => i);
                    [resNode] = builder.createOp(
                        "Squeeze",
                        [mm, builder.createConstant(uid("sq"), int64Vec(sqAxes))],
                        {},
                        [{ type: dtype, shape: [] as KnownShape }],
                    );
                } else {
                    [resNode] = builder.createOp(
                        "Reshape",
                        [mm, packShapeVector(builder, finalDims, uid("osh"))],
                        {},
                        [{ type: dtype, shape: toKnown(finalDims) }],
                    );
                }
                acc = { node: resNode, labels: finalLabels, dims: finalDims };
            } else {
                // General alignment: Mul + ReduceSum
                const unionLabels = [...acc.labels, ...right];
                const aAlignedDims: MixedDim[] = [
                    ...acc.dims,
                    ...new Array<number>(right.length).fill(1),
                ];
                const bOrderedLabels = unionLabels.filter((l) => rhsSet.has(l));
                const bT = transposeTo(rhs.node, rhs.labels, rhs.dims, bOrderedLabels);
                const bAlignedDims: MixedDim[] = unionLabels.map((l) =>
                    rhsSet.has(l) ? rhs.dims[rhs.labels.indexOf(l)] : 1,
                );
                const mulDims = aAlignedDims.map((d, i) =>
                    maxMixed(d, bAlignedDims[i], uid(`md_${i}`)),
                );

                const aAligned =
                    right.length > 0
                        ? builder.createOp(
                              "Reshape",
                              [acc.node, packShapeVector(builder, aAlignedDims, uid("aAl"))],
                              {},
                              [{ type: dtype, shape: toKnown(aAlignedDims) }],
                          )[0]
                        : acc.node;
                const bAligned = builder.createOp(
                    "Reshape",
                    [bT.node, packShapeVector(builder, bAlignedDims, uid("bAl"))],
                    {},
                    [{ type: dtype, shape: toKnown(bAlignedDims) }],
                )[0];
                const [prod] = builder.createOp("Mul", [aAligned, bAligned], {}, [
                    { type: dtype, shape: toKnown(mulDims) },
                ]);
                const dropIdx = unionLabels
                    .map((l, i) => (!futureNeeded.has(l) ? i : -1))
                    .filter((i) => i >= 0);
                acc = reduceAxes(prod, unionLabels, mulDims, dropIdx);
            }
        }

        // Step 3: Final reduction & output axis permutation
        const outSet = new Set(parsed.outLabels);
        const dropFinal = acc.labels.map((l, i) => (!outSet.has(l) ? i : -1)).filter((i) => i >= 0);
        acc = reduceAxes(acc.node, acc.labels, acc.dims, dropFinal);
        const finalT = transposeTo(acc.node, acc.labels, acc.dims, parsed.outLabels);

        builder.replaceAllUsesWith(Y, finalT.node);
        builder.removeNode(op);
    }
}
