import type OnnxGraph from "../../ir/OnnxGraph.js";
import OperationNode from "../../ir/OperationNode.js";
import TensorNode from "../../ir/TensorNode.js";
import type { GraphBuilder } from "../../ir/GraphBuilder.js";
import type { ValueNode, KnownShape, StaticShape, ConcreteValueNode } from "../../ir/OnnxTypes.js";
import { DataType } from "../../ir/OnnxTypes.js";
import { getRegionScopeTag, toStaticShape, int64Vec } from "../../utils/index.js";
import { buildForLoopRegion, createAccumulatorCarry, type LoopScanOutput } from "./LoopBuilder.js";
import {
    buildLinearIndex,
    buildMixedStrides,
    decodeMixedRadix,
    extractMixedDims,
} from "./RecipeUtils.js";
import { lookupScheduleMap } from "../schedule/Schedule.js";
import { IndexingMode } from "../../../DecompositionOptions.js";
import type { LoopChainFinder, LoopFusionOptions } from "./LoopChainFinder.js";

/**
 * Lowers a topologically ordered chain of fuseable operations into a single ONNX `Loop`,
 * supporting AGU pointer vs. radix indexing, schedule axis permutations, carry-tensor vs.
 * scan outputs, provenance metadata stamping, and post-loop reshaping/rewiring.
 */
export class LoopChainLowerer {
    constructor(
        private readonly finder: LoopChainFinder,
        private readonly options: LoopFusionOptions,
    ) {}

    /**
     * Lowers `chain` into a single ONNX `Loop` inside `graph` using `outerBuilder`,
     * post-processing and reshaping all live cluster outputs back to their original shapes.
     */
    public lowerChain(
        graph: OnnxGraph.Class,
        chain: OperationNode.Class[],
        outerBuilder: GraphBuilder,
    ): void {
        const rootOp = chain[chain.length - 1];
        const rootRecipe = this.finder.getRecipeFor(rootOp)!;

        // Fast-path for standalone recipes (e.g., SoftmaxLoop, DetLoop, TopKScalarLoop)
        if (chain.length === 1 && rootRecipe.lowerStandalone !== undefined) {
            const prevLoops = new Set(
                graph
                    .getOperationNodes()
                    .toArray()
                    .filter((o) => o.type === "Loop")
                    .map((o) => o.id),
            );
            rootRecipe.lowerStandalone(rootOp, outerBuilder, this.options.indexingMode);
            const newLoops = graph
                .getOperationNodes()
                .toArray()
                .filter((o) => o.type === "Loop" && !prevLoops.has(o.id));
            this.stampLoopMetadata(chain, newLoops);
            return;
        }

        const rootOutNodeRaw = rootOp.getOutgoers.targets.filterIs(TensorNode).first()!;
        const chainSet = new Set(chain.map((o) => o.id));
        const { chainOutputs, seenOutIds } = this.collectExternalChainOutputs(
            chain,
            chainSet,
            rootOutNodeRaw,
        );

        const originalOutShape: KnownShape = [...(rootOutNodeRaw.shape as KnownShape)];
        const elemTy = rootOutNodeRaw.literalType || DataType.FLOAT;

        const bounds = this.finder.getBoundsFor(rootOp);
        const totalIters = bounds.totalIters;
        const carryShape = bounds.carryShape as StaticShape;
        const isSelfCarry = rootRecipe.tags?.includes("self-carry") === true;
        const useScan = this.options.useScanOutputs === true && !isSelfCarry;
        const axisPerm = this.resolveAxisPermutation(chain, originalOutShape, isSelfCarry);

        // 1. Initialize Loop region (incorporating any nested-region scope tag to prevent ID collisions)
        const regionScope = getRegionScopeTag(graph, "");
        const scopePrefix = regionScope.length > 0 ? `${regionScope}_` : "";
        const loopTag = `${scopePrefix}Loop_${rootOp.id}`;
        const loopScopeTag = `${scopePrefix}lowering_${rootOp.id}`;
        const effectiveIndexingMode =
            axisPerm !== undefined ? IndexingMode.Radix : this.options.indexingMode;
        const { innerBuilder, trip, vInitial, loopOutput, loopOutputs, loopCtx, finalize } =
            buildForLoopRegion(
                outerBuilder,
                totalIters,
                elemTy,
                carryShape,
                loopTag,
                loopScopeTag,
                effectiveIndexingMode,
                !useScan,
            );
        const body = innerBuilder.graph;

        const effectiveIter: ConcreteValueNode =
            axisPerm !== undefined
                ? this.buildPermutedIter(
                      innerBuilder,
                      trip,
                      axisPerm,
                      originalOutShape,
                      bounds.targetShape as ValueNode | undefined,
                      rootOp.id,
                  )
                : trip;

        const axes = innerBuilder.createConstant("axes", int64Vec([0]));
        const valueMap = new Map<string, ValueNode>();
        let finalNextCarry: ValueNode | null = null;

        // 2. Lower each operation in the chain into the loop body
        for (const op of chain) {
            const recipe = this.finder.getRecipeFor(op)!;
            const applyRes = recipe.apply(
                op,
                body,
                valueMap,
                effectiveIter,
                axes,
                originalOutShape,
                vInitial,
                bounds.targetShape as ValueNode | undefined,
                loopCtx,
            );

            let scalarOut: ValueNode;
            if (typeof applyRes === "object" && "nextCarry" in applyRes) {
                scalarOut = applyRes.resultNode;
                if (op.id === rootOp.id) finalNextCarry = applyRes.nextCarry;
            } else {
                scalarOut = applyRes as ValueNode;
            }

            const outTensors = op.getOutgoers.targets.filterIs(TensorNode).toArray();
            if (outTensors.length > 0) {
                valueMap.set(outTensors[0].id, scalarOut);
            }
        }

        // 3. Finalize Loop outputs (native scan_outputs vs ScatterElements carry buffers)
        const rawLoopOutputs: ValueNode[] = new Array(chainOutputs.length);
        let emittedLoopOp: OperationNode.Class;

        if (useScan && finalNextCarry === null) {
            const scanOutputs: LoopScanOutput[] = chainOutputs.map((outT, idx) => {
                const scalar = valueMap.get(outT.id)!;
                const scanVal =
                    (scalar.shape as unknown[]).length === 0
                        ? scalar
                        : innerBuilder.createOp("Reshape", [
                              scalar,
                              innerBuilder.createConstant(`scan_0d_${idx}`, int64Vec([])),
                          ])[0];
                return {
                    tag: idx === 0 ? "out" : `out_${idx}`,
                    node: scanVal as ConcreteValueNode,
                    elemTy: outT.literalType || elemTy,
                    shape: [],
                };
            });
            emittedLoopOp = finalize([], scanOutputs);
            const scanBase = loopOutputs.length - chainOutputs.length;
            for (let j = 0; j < chainOutputs.length; j++) {
                rawLoopOutputs[j] = loopOutputs[scanBase + j];
            }
        } else {
            const oneShape = innerBuilder.createConstant("one_shape", int64Vec([1]));
            const sharedStoreIdx: ConcreteValueNode = effectiveIter;
            const [iterUnsq] = !finalNextCarry
                ? innerBuilder.createOp("Reshape", [sharedStoreIdx, oneShape])
                : [sharedStoreIdx];

            let primaryCarryOut: ValueNode;
            if (finalNextCarry) {
                primaryCarryOut = finalNextCarry;
            } else {
                const finalScalar0 = valueMap.get(chainOutputs[0].id)!;
                const [finalUnsq0] = innerBuilder.createOp("Reshape", [finalScalar0, oneShape]);
                [primaryCarryOut] = innerBuilder.createOp(
                    "ScatterElements",
                    [vInitial, iterUnsq, finalUnsq0],
                    { axis: 0 },
                );
            }

            const extraOutIndices: number[] = [];
            if (!finalNextCarry && chainOutputs.length > 1) {
                for (let j = 1; j < chainOutputs.length; j++) {
                    const outT = chainOutputs[j];
                    const outTy = outT.literalType || elemTy;
                    const extraDef = createAccumulatorCarry(
                        outerBuilder,
                        outTy,
                        carryShape,
                        `out${j}_${rootOp.id}`,
                    );
                    const cIn = loopCtx.addCarry(
                        `out_${j}_${rootOp.id}`,
                        extraDef.initialVal,
                        extraDef.elemTy,
                        extraDef.shape,
                    );
                    extraOutIndices.push(loopOutputs.length - 1);
                    const finalScalarJ = valueMap.get(outT.id)!;
                    const [finalUnsqJ] = innerBuilder.createOp("Reshape", [finalScalarJ, oneShape]);
                    const [nextCarryJ] = innerBuilder.createOp(
                        "ScatterElements",
                        [cIn, iterUnsq, finalUnsqJ],
                        { axis: 0 },
                    );
                    loopCtx.setNextState(cIn, nextCarryJ);
                }
            }

            emittedLoopOp = finalize([primaryCarryOut as ConcreteValueNode]);
            rawLoopOutputs[0] = loopOutput;
            extraOutIndices.forEach((outIdx, k) => {
                rawLoopOutputs[k + 1] = loopOutputs[outIdx];
            });
        }

        this.stampLoopMetadata(chain, [emittedLoopOp]);
        emittedLoopOp.setMetadata("scalarizedLoop", true);
        this.rewireAbsorbedShapeProbes(chain, chainSet, rootOp, seenOutIds, outerBuilder);

        // 4. Post-process, reshape, and reconnect each output of the cluster
        for (let j = 0; j < chainOutputs.length; j++) {
            const outT = chainOutputs[j];
            const outShapeJ: KnownShape = [...(outT.shape as KnownShape)];
            const outTyJ = outT.literalType || elemTy;
            let processedOut: ValueNode = rawLoopOutputs[j];
            if (j === 0 && rootRecipe.postProcess !== undefined) {
                processedOut = rootRecipe.postProcess(rootOp, outerBuilder, processedOut);
            }

            const staticCarryTotal = Array.isArray(carryShape)
                ? (carryShape as number[]).reduce((a, b) => a * b, 1)
                : -1;
            const isScalarOutput = outShapeJ.length === 0 && staticCarryTotal === 1;
            const outIsDynamic =
                outShapeJ.includes(-1) || (outShapeJ.length === 0 && !isScalarOutput);

            let targetShapeNode: ValueNode;
            if (j === 0 && bounds.targetShape !== undefined) {
                targetShapeNode = bounds.targetShape as ValueNode;
            } else if (Array.isArray(carryShape)) {
                const staticCarryShape = carryShape as number[];
                targetShapeNode = outerBuilder.createConstant(
                    `final_shape_${j}`,
                    int64Vec(outIsDynamic ? staticCarryShape : (outShapeJ as number[])),
                );
            } else {
                targetShapeNode = carryShape as ValueNode;
            }

            let finalOutput: ValueNode;
            if (isScalarOutput) {
                const squeezeAxis = outerBuilder.createConstant(
                    `squeeze_scalar_axis_${rootOp.id}_${j}`,
                    int64Vec([0]),
                );
                [finalOutput] = outerBuilder.createOp("Squeeze", [processedOut, squeezeAxis], {}, [
                    { type: outTyJ, shape: [] },
                ]);
            } else if (useScan && axisPerm !== undefined) {
                const permShapeKnown = axisPerm.map((p) => outShapeJ[p]) as KnownShape;
                const invPerm: number[] = new Array(axisPerm.length);
                axisPerm.forEach((p, idx) => {
                    invPerm[p] = idx;
                });
                const permShapeConst = outerBuilder.createConstant(
                    `perm_shape_${rootOp.id}_${j}`,
                    int64Vec(permShapeKnown as number[]),
                );
                const [permReshaped] = outerBuilder.createOp(
                    "Reshape",
                    [processedOut, permShapeConst],
                    {},
                    [{ type: outTyJ, shape: permShapeKnown }],
                );
                [finalOutput] = outerBuilder.createOp(
                    "Transpose",
                    [permReshaped],
                    { perm: invPerm },
                    [{ type: outTyJ, shape: outShapeJ }],
                );
            } else {
                const [reshaped] = outerBuilder.createOp(
                    "Reshape",
                    [processedOut, targetShapeNode],
                    {},
                    [{ type: outTyJ, shape: outShapeJ }],
                );
                finalOutput = reshaped;
            }

            outerBuilder.replaceAllUsesWith(outT, finalOutput);
        }

        chain.forEach((op) => outerBuilder.removeNode(op));
    }

    /**
     * Stamps provenance metadata (`sourceOps`, `sourceOpTypes`) onto newly emitted
     * `Loop` nodes so downstream `Schedule` passes can match them by original op ID/type.
     */
    private stampLoopMetadata(chain: OperationNode.Class[], loops: OperationNode.Class[]): void {
        const srcOpsSet = new Set<string>();
        const srcTypesSet = new Set<string>();
        for (const o of chain) {
            srcOpsSet.add(o.id);
            srcTypesSet.add(o.type);
            const prevOps = o.getMetadata("sourceOps");
            if (Array.isArray(prevOps)) prevOps.forEach((s) => srcOpsSet.add(String(s)));
            const prevTypes = o.getMetadata("sourceOpTypes");
            if (Array.isArray(prevTypes)) prevTypes.forEach((s) => srcTypesSet.add(String(s)));
        }
        const srcOps = Array.from(srcOpsSet);
        const srcTypes = Array.from(srcTypesSet);
        for (const l of loops) {
            l.setMetadata("sourceOps", srcOps);
            l.setMetadata("sourceOpTypes", srcTypes);
        }
    }

    /**
     * Collects all tensors produced inside `chain` that must remain live after fusion
     * (always starting with `rootOutNode`, plus any graph output or externally consumed tensor).
     */
    private collectExternalChainOutputs(
        chain: OperationNode.Class[],
        chainSet: Set<string>,
        rootOutNode: TensorNode.Class,
    ): { chainOutputs: TensorNode.Class[]; seenOutIds: Set<string> } {
        const chainOutputs: TensorNode.Class[] = [rootOutNode];
        const seenOutIds = new Set<string>([rootOutNode.id]);

        for (const op of chain) {
            for (const t of op.getOutgoers.targets.filterIs(TensorNode).toArray()) {
                if (seenOutIds.has(t.id)) continue;
                const hasExternalConsumer = t.outgoers.targets
                    .filterIs(OperationNode)
                    .toArray()
                    .some((c) => !chainSet.has(c.id));
                if (t.type === "output" || hasExternalConsumer) {
                    seenOutIds.add(t.id);
                    chainOutputs.push(t);
                }
            }
        }

        return { chainOutputs, seenOutIds };
    }

    /** Resolves and validates any `Schedule.reorder(...)` axis permutation targeting `chain`. */
    private resolveAxisPermutation(
        chain: OperationNode.Class[],
        originalOutShape: KnownShape,
        isSelfCarry: boolean,
    ): number[] | undefined {
        if (isSelfCarry || originalOutShape.length <= 1) return undefined;

        const rawAxisPerm = chain
            .map((o) => lookupScheduleMap(this.options.schedule?.axisPerms, o))
            .find((p) => p !== undefined);

        if (
            rawAxisPerm !== undefined &&
            rawAxisPerm.length === originalOutShape.length &&
            rawAxisPerm.every(
                (p, idx, arr) =>
                    Number.isInteger(p) &&
                    p >= 0 &&
                    p < originalOutShape.length &&
                    arr.indexOf(p) === idx,
            )
        ) {
            return rawAxisPerm;
        }
        return undefined;
    }

    /**
     * Maps the flat loop iteration counter `trip` through a permuted axis order
     * back to the canonical row-major linear offset expected by body recipes.
     */
    private buildPermutedIter(
        innerBuilder: GraphBuilder,
        trip: ConcreteValueNode,
        axisPerm: number[],
        originalOutShape: KnownShape,
        targetShape: ValueNode | undefined,
        rootOpId: string,
    ): ConcreteValueNode {
        const innerOutDims: (number | ConcreteValueNode)[] = targetShape
            ? extractMixedDims(
                  innerBuilder,
                  targetShape as ConcreteValueNode,
                  [originalOutShape.length],
                  `perm_t_${rootOpId}`,
              )
            : originalOutShape.map((d) => (typeof d === "number" && d > 0 ? d : 1));
        const permDims = axisPerm.map((p) => innerOutDims[p]);
        const origStrides = buildMixedStrides(innerBuilder, innerOutDims);
        const permCoords = decodeMixedRadix(innerBuilder, trip, permDims, `perm_dec_${rootOpId}`);
        const origCoords: ConcreteValueNode[] = new Array(axisPerm.length);
        axisPerm.forEach((p, idx) => {
            origCoords[p] = permCoords[idx];
        });
        return buildLinearIndex(innerBuilder, origCoords, origStrides, `perm_idx_${rootOpId}`);
    }

    /**
     * Redirects surviving external `Shape` probes on absorbed intermediate tensors
     * to an equivalent same-shape external input before removing fused chain ops.
     */
    private rewireAbsorbedShapeProbes(
        chain: OperationNode.Class[],
        chainSet: Set<string>,
        rootOp: OperationNode.Class,
        seenOutIds: Set<string>,
        outerBuilder: GraphBuilder,
    ): void {
        const findExternalShapeSource = (
            t: TensorNode.Class,
            seen = new Set<string>(),
        ): ValueNode | undefined => {
            if (seen.has(t.id)) return undefined;
            seen.add(t.id);
            const prod = t.incomers.sources.filterIs(OperationNode).first();
            if (!prod || !chainSet.has(prod.id)) return t;
            const outRank = (t.shape as unknown[] | undefined)?.length ?? 0;
            const tSig = toStaticShape(t.shape).join(",");
            const nonScalar = (prod.getInputs() ?? []).filter((inp) => {
                const sh = inp.shape as (number | string)[] | undefined;
                if (!sh || sh.length === 0) return false;
                if (sh.length <= outRank && sh.every((d) => d === 1)) return false;
                return toStaticShape(sh).join(",") === tSig;
            });
            for (const inp of nonScalar) {
                if (inp.is(TensorNode)) {
                    const ext = findExternalShapeSource(inp.as(TensorNode), seen);
                    if (ext !== undefined) return ext;
                } else {
                    return inp;
                }
            }
            return undefined;
        };

        for (const op of chain) {
            if (op.id === rootOp.id) continue;
            for (const t of op.getOutgoers.targets.filterIs(TensorNode).toArray()) {
                if (seenOutIds.has(t.id)) continue;
                const remaining = t.outgoers.targets.filterIs(OperationNode).toArray();
                if (remaining.some((c) => !chainSet.has(c.id))) {
                    const ext = findExternalShapeSource(t);
                    if (ext !== undefined) outerBuilder.replaceAllUsesWith(t, ext);
                }
            }
        }
    }
}
