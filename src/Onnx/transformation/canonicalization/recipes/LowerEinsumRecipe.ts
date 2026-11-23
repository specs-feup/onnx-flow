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

    // 2. Map label -> static size (allowing size-1 broadcast)
    const labelSizes = new Map<string, number>();
    for (let i = 0; i < inLabels.length; i++) {
        for (let ax = 0; ax < inLabels[i].length; ax++) {
            const lbl = inLabels[i][ax];
            const dim = shapes[i][ax];
            const prev = labelSizes.get(lbl);
            if (prev === undefined || prev === 1) {
                labelSizes.set(lbl, dim);
            } else if (dim !== 1 && dim !== prev) {
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
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = [
        "Transpose",
        "Reshape",
        "MatMul",
        "Mul",
        "ReduceSum",
        "Range",
        "Equal",
        "Where",
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
            const dims = toStaticShape(node.shape);
            if (dims.includes(-1)) return false;
            shapes.push(dims);
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

        const transposeTo = (
            node: ConcreteValueNode,
            curLabels: string[],
            curShape: number[],
            targetLabels: string[],
        ): { node: ConcreteValueNode; shape: number[] } => {
            const perm = targetLabels.map((l) => curLabels.indexOf(l));
            if (perm.every((p, i) => p === i)) return { node, shape: curShape };
            const newShape = perm.map((p) => curShape[p]);
            const [t] = builder.createOp("Transpose", [node], { perm }, [
                { type: dtype, shape: newShape as KnownShape },
            ]);
            return { node: t, shape: newShape };
        };

        const reduceAxes = (
            node: ConcreteValueNode,
            curLabels: string[],
            curShape: number[],
            dropIndices: number[],
        ): { node: ConcreteValueNode; labels: string[]; shape: number[] } => {
            if (dropIndices.length === 0) return { node, labels: curLabels, shape: curShape };
            const dropSet = new Set(dropIndices);
            const nextLabels = curLabels.filter((_, i) => !dropSet.has(i));
            const nextShape = curShape.filter((_, i) => !dropSet.has(i));
            const axesC = builder.createConstant(uid("rax"), int64Vec(dropIndices));
            const [r] = builder.createOp("ReduceSum", [node, axesC], { keepdims: 0 }, [
                { type: dtype, shape: nextShape as KnownShape },
            ]);
            return { node: r, labels: nextLabels, shape: nextShape };
        };

        // Step 1: Resolve repeated indices (trace/diagonal) inside each operand
        const operands = inputs.map((inp, idx) => {
            let cur = inp;
            let labels = [...parsed.inLabels[idx]];
            let shape = [...shapes[idx]];

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
                const dim = shape[a];
                const [rVec] = builder.createOp(
                    "Range",
                    [
                        builder.createConstant(uid("r0"), scalarInt64(0)),
                        builder.createConstant(uid("rN"), scalarInt64(dim)),
                        builder.createConstant(uid("r1"), scalarInt64(1)),
                    ],
                    {},
                    [{ type: DataType.INT64, shape: [dim] as KnownShape }],
                );
                const shA = new Array<number>(shape.length).fill(1);
                shA[a] = dim;
                const shB = new Array<number>(shape.length).fill(1);
                shB[b] = dim;
                const [rA] = builder.createOp(
                    "Reshape",
                    [rVec, builder.createConstant(uid("shA"), int64Vec(shA))],
                    {},
                    [{ type: DataType.INT64, shape: shA as KnownShape }],
                );
                const [rB] = builder.createOp(
                    "Reshape",
                    [rVec, builder.createConstant(uid("shB"), int64Vec(shB))],
                    {},
                    [{ type: DataType.INT64, shape: shB as KnownShape }],
                );
                const maskShape = shA.map((v, i) => Math.max(v, shB[i]));
                const [mask] = builder.createOp("Equal", [rA, rB], {}, [
                    { type: DataType.BOOL, shape: maskShape as KnownShape },
                ]);
                const zeroC = builder.createConstant(uid("z"), makeTensorProto(dtype, [], [0]));
                const [masked] = builder.createOp("Where", [mask, cur, zeroC], {}, [
                    { type: dtype, shape: shape as KnownShape },
                ]);
                const res = reduceAxes(masked, labels, shape, [b]);
                cur = res.node;
                labels = res.labels;
                shape = res.shape;
            }
            return { node: cur, labels, shape };
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
                [...batch, ...contract].every(
                    (l) => acc.shape[acc.labels.indexOf(l)] === rhs.shape[rhs.labels.indexOf(l)],
                );

            if (exactBatchAndContract) {
                const aOrder = [...batch, ...left, ...contract];
                const bOrder = [...batch, ...contract, ...right];
                const aT = transposeTo(acc.node, acc.labels, acc.shape, aOrder);
                const bT = transposeTo(rhs.node, rhs.labels, rhs.shape, bOrder);

                const bDims = batch.map((l) => acc.shape[acc.labels.indexOf(l)]);
                const lDims = left.map((l) => acc.shape[acc.labels.indexOf(l)]);
                const rDims = right.map((l) => rhs.shape[rhs.labels.indexOf(l)]);
                const kDim = contract.reduce((p, l) => p * acc.shape[acc.labels.indexOf(l)], 1);
                const mDim = lDims.reduce((p, d) => p * d, 1);
                const nDim = rDims.reduce((p, d) => p * d, 1);

                const aMatShape = [...bDims, mDim, kDim];
                const bMatShape = [...bDims, kDim, nDim];
                const mmOutShape = [...bDims, mDim, nDim];
                const finalLabels = [...batch, ...left, ...right];
                const finalShape = [...bDims, ...lDims, ...rDims];

                const [aR] = builder.createOp(
                    "Reshape",
                    [aT.node, builder.createConstant(uid("ash"), int64Vec(aMatShape))],
                    {},
                    [{ type: dtype, shape: aMatShape as KnownShape }],
                );
                const [bR] = builder.createOp(
                    "Reshape",
                    [bT.node, builder.createConstant(uid("bsh"), int64Vec(bMatShape))],
                    {},
                    [{ type: dtype, shape: bMatShape as KnownShape }],
                );
                const [mm] = builder.createOp("MatMul", [aR, bR], {}, [
                    { type: dtype, shape: mmOutShape as KnownShape },
                ]);
                let resNode: ConcreteValueNode = mm;
                if (finalShape.length === 0) {
                    const sqAxes = mmOutShape.map((_, i) => i);
                    [resNode] = builder.createOp(
                        "Squeeze",
                        [mm, builder.createConstant(uid("sq"), int64Vec(sqAxes))],
                        {},
                        [{ type: dtype, shape: [] as KnownShape }],
                    );
                } else {
                    [resNode] = builder.createOp(
                        "Reshape",
                        [mm, builder.createConstant(uid("osh"), int64Vec(finalShape))],
                        {},
                        [{ type: dtype, shape: finalShape as KnownShape }],
                    );
                }
                acc = { node: resNode, labels: finalLabels, shape: finalShape };
            } else {
                // General alignment: Mul + ReduceSum
                const unionLabels = [...acc.labels, ...right];
                const aAlignedShape = [...acc.shape, ...new Array<number>(right.length).fill(1)];
                const bOrderedLabels = unionLabels.filter((l) => rhsSet.has(l));
                const bT = transposeTo(rhs.node, rhs.labels, rhs.shape, bOrderedLabels);
                const bAlignedShape = unionLabels.map((l) =>
                    rhsSet.has(l) ? rhs.shape[rhs.labels.indexOf(l)] : 1,
                );
                const mulShape = aAlignedShape.map((d, i) => Math.max(d, bAlignedShape[i]));

                const aAligned =
                    right.length > 0
                        ? builder.createOp(
                              "Reshape",
                              [
                                  acc.node,
                                  builder.createConstant(uid("aAl"), int64Vec(aAlignedShape)),
                              ],
                              {},
                              [{ type: dtype, shape: aAlignedShape as KnownShape }],
                          )[0]
                        : acc.node;
                const bAligned = builder.createOp(
                    "Reshape",
                    [bT.node, builder.createConstant(uid("bAl"), int64Vec(bAlignedShape))],
                    {},
                    [{ type: dtype, shape: bAlignedShape as KnownShape }],
                )[0];
                const [prod] = builder.createOp("Mul", [aAligned, bAligned], {}, [
                    { type: dtype, shape: mulShape as KnownShape },
                ]);
                const dropIdx = unionLabels
                    .map((l, i) => (!futureNeeded.has(l) ? i : -1))
                    .filter((i) => i >= 0);
                acc = reduceAxes(prod, unionLabels, mulShape, dropIdx);
            }
        }

        // Step 3: Final reduction & output axis permutation
        const outSet = new Set(parsed.outLabels);
        const dropFinal = acc.labels.map((l, i) => (!outSet.has(l) ? i : -1)).filter((i) => i >= 0);
        acc = reduceAxes(acc.node, acc.labels, acc.shape, dropFinal);
        const finalT = transposeTo(acc.node, acc.labels, acc.shape, parsed.outLabels);

        builder.replaceAllUsesWith(Y, finalT.node);
        builder.removeNode(op);
    }
}
