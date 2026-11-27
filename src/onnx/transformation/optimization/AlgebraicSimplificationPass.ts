import type OnnxGraph from "../../ir/OnnxGraph.js";
import OperationNode from "../../ir/OperationNode.js";
import ConstantNode from "../../ir/ConstantNode.js";
import RegionArgumentNode from "../../ir/RegionArgumentNode.js";
import type { ValueNode } from "../../ir/OnnxTypes.js";
import { DataType } from "../../ir/OnnxTypes.js";
import { GraphBuilder } from "../../ir/GraphBuilder.js";
import {
    getIntAttr,
    getIntsAttr,
    makeTensorProto,
    readTensorData,
    topologicalSortOperationNodes,
    toStaticShape,
} from "../../utils/index.js";
import { BaseGraphPass } from "../GraphPass.js";
import {
    asRewritableIntermediate,
    cleanupOrphanConstant,
    collectCapturedIds,
} from "./OptimizationUtils.js";

const INT_DTYPES = new Set<DataType>([DataType.INT32, DataType.INT64]);

function isAllValue(node: ValueNode, expected: number): boolean {
    if (!node.is(ConstantNode)) return false;
    const vals = readTensorData(node);
    return vals !== undefined && vals.length > 0 && vals.every((v) => v === expected);
}

function sameKnownShape(a: ValueNode, b: ValueNode): boolean {
    const sa = toStaticShape(a.shape);
    const sb = toStaticShape(b.shape);
    if (sa.length !== sb.length) return false;
    if (sa.includes(-1) || sb.includes(-1)) return false;
    return sa.every((d, i) => d === sb[i]);
}

function isIdentityPerm(perm: readonly number[]): boolean {
    return perm.every((p, i) => p === i);
}

/**
 * Performs safe, semantics-preserving algebraic identity and view-chain simplifications
 * across all nested regions:
 * - `Add(x, 0)`, `Add(0, x)`, `Sub(x, 0)`, `Mul(x, 1)`, `Mul(1, x)`, `Div(x, 1)` → `x`
 * - Integer `Mul(x, 0)` / `Mul(0, x)` → zero constant (preserving IEEE float NaN/Inf rules)
 * - No-op `Cast(x, to = x.dtype)` → `x`
 * - Identity `Transpose(x, [0..r-1])` → `x` and chained `Transpose(Transpose(x, p1), p2)` composition
 * - Chained `Reshape(Reshape(x, s1), s2)` → `Reshape(x, s2)` when `s2` has no `0` copy-dims
 */
export class AlgebraicSimplificationPass extends BaseGraphPass {
    public override readonly name = "AlgebraicSimplification";

    constructor(private readonly maxZeroFillElements: number = 4096) {
        super();
    }

    public override run(graph: OnnxGraph.Class): boolean {
        let globalChanged = this.runOnRegions(graph);
        const capturedIds = collectCapturedIds(graph);

        let localChanged = true;
        while (localChanged) {
            localChanged = false;
            const ops = topologicalSortOperationNodes(graph);

            for (const op of ops) {
                if (!graph.hasNode(op.id)) continue;
                const outputs = op.getOutputs();
                if (outputs.length !== 1) continue;

                const outTensor = asRewritableIntermediate(outputs[0], capturedIds);
                if (outTensor === undefined || outTensor.outgoers.length === 0) continue;

                if (this.trySimplifyOp(op, outTensor, graph, capturedIds)) {
                    localChanged = true;
                    globalChanged = true;
                }
            }
        }

        return globalChanged;
    }

    private bypassWith(
        op: OperationNode.Class,
        outTensor: ValueNode,
        replacement: ValueNode,
        graph: OnnxGraph.Class,
        capturedIds: Set<string>,
    ): boolean {
        if (!graph.hasNode(replacement.id) || replacement.is(RegionArgumentNode)) return false;
        const inputs = op.getInputs() ?? [];
        const builder = new GraphBuilder(graph, `alg_${op.id}`);
        builder.replaceAllUsesWith(outTensor, replacement);
        op.remove();
        if (graph.hasNode(outTensor.id) && outTensor.outgoers.length === 0) {
            outTensor.remove();
        }
        for (const inp of inputs) {
            if (inp.id !== replacement.id) {
                cleanupOrphanConstant(inp, graph, capturedIds);
            }
        }
        return true;
    }

    private trySimplifyOp(
        op: OperationNode.Class,
        outTensor: ValueNode,
        graph: OnnxGraph.Class,
        capturedIds: Set<string>,
    ): boolean {
        const inputs = op.getInputs() ?? [];

        // 1. Additive identities: Add(x, 0), Add(0, x), Sub(x, 0)
        if ((op.type === "Add" || op.type === "Sub") && inputs.length === 2) {
            const [a, b] = inputs;
            if (isAllValue(b, 0) && sameKnownShape(a, outTensor)) {
                return this.bypassWith(op, outTensor, a, graph, capturedIds);
            }
            if (op.type === "Add" && isAllValue(a, 0) && sameKnownShape(b, outTensor)) {
                return this.bypassWith(op, outTensor, b, graph, capturedIds);
            }
        }

        // 2. Multiplicative identities & integer zero annihilation
        if ((op.type === "Mul" || op.type === "Div") && inputs.length === 2) {
            const [a, b] = inputs;
            if (isAllValue(b, 1) && sameKnownShape(a, outTensor)) {
                return this.bypassWith(op, outTensor, a, graph, capturedIds);
            }
            if (op.type === "Mul" && isAllValue(a, 1) && sameKnownShape(b, outTensor)) {
                return this.bypassWith(op, outTensor, b, graph, capturedIds);
            }
            if (
                op.type === "Mul" &&
                INT_DTYPES.has(outTensor.literalType) &&
                (isAllValue(a, 0) || isAllValue(b, 0))
            ) {
                const outShape = toStaticShape(outTensor.shape);
                const size = outShape.reduce((acc, d) => acc * d, 1);
                if (!outShape.includes(-1) && size <= this.maxZeroFillElements) {
                    const builder = new GraphBuilder(graph, `alg_zero_${op.id}`);
                    const zeroConst = builder.createConstant(
                        `zero_${op.id}`,
                        makeTensorProto(outTensor.literalType, outShape, new Array(size).fill(0)),
                    );
                    return this.bypassWith(op, outTensor, zeroConst, graph, capturedIds);
                }
            }
        }

        // 3. No-op Cast
        if (op.type === "Cast" && inputs.length === 1) {
            const to = getIntAttr(op, "to", DataType.UNDEFINED) as DataType;
            if (to !== DataType.UNDEFINED && inputs[0].literalType === to) {
                return this.bypassWith(op, outTensor, inputs[0], graph, capturedIds);
            }
        }

        // 4. Identity & chained Transpose composition
        if (op.type === "Transpose" && inputs.length === 1) {
            const src = inputs[0];
            const rank = src.shape.length;
            if (rank > 0) {
                const defaultPerm = Array.from({ length: rank }, (_, i) => rank - 1 - i);
                const perm = getIntsAttr(op, "perm", defaultPerm);
                if (perm.length === rank && isIdentityPerm(perm)) {
                    return this.bypassWith(op, outTensor, src, graph, capturedIds);
                }

                // Compose Transpose(Transpose(x, p1), p2) -> Transpose(x, p2[p1])
                const prod = src.incomers.sources.filterIs(OperationNode).first();
                if (prod !== undefined && prod.type === "Transpose") {
                    const grandIn = prod.getInputs()?.[0];
                    if (
                        grandIn !== undefined &&
                        graph.hasNode(grandIn.id) &&
                        !grandIn.is(RegionArgumentNode)
                    ) {
                        const p1 = getIntsAttr(prod, "perm", defaultPerm);
                        if (p1.length === rank && perm.length === rank) {
                            const composed = perm.map((p) => p1[p]);
                            if (isIdentityPerm(composed)) {
                                return this.bypassWith(op, outTensor, grandIn, graph, capturedIds);
                            }
                            const builder = new GraphBuilder(graph, `alg_tr_${op.id}`);
                            const [newTr] = builder.createOp(
                                "Transpose",
                                [grandIn],
                                { perm: composed },
                                [
                                    {
                                        type: outTensor.literalType,
                                        shape: toStaticShape(outTensor.shape),
                                    },
                                ],
                            );
                            return this.bypassWith(op, outTensor, newTr, graph, capturedIds);
                        }
                    }
                }
            }
        }

        // 5. Chained Reshape collapse: Reshape(Reshape(x, s1), s2) -> Reshape(x, s2)
        if (op.type === "Reshape" && inputs.length === 2) {
            const [src, shapeSpec] = inputs;
            const prod = src.incomers.sources.filterIs(OperationNode).first();
            if (prod !== undefined && prod.type === "Reshape") {
                const grandIn = prod.getInputs()?.[0];
                const allowZero = getIntAttr(op, "allowzero", 0);
                const specVals = shapeSpec.is(ConstantNode) ? readTensorData(shapeSpec) : undefined;
                const safeToCollapse =
                    allowZero === 1 || (specVals !== undefined && !specVals.includes(0));
                if (
                    safeToCollapse &&
                    grandIn !== undefined &&
                    graph.hasNode(grandIn.id) &&
                    !grandIn.is(RegionArgumentNode)
                ) {
                    const builder = new GraphBuilder(graph, `alg_rs_${op.id}`);
                    const [newRs] = builder.createOp(
                        "Reshape",
                        [grandIn, shapeSpec],
                        { ...op.attributes },
                        [{ type: outTensor.literalType, shape: toStaticShape(outTensor.shape) }],
                    );
                    return this.bypassWith(op, outTensor, newRs, graph, capturedIds);
                }
            }
        }

        return false;
    }
}
