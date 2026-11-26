import type OnnxGraph from "../../OnnxGraph.js";
import OperationNode from "../../OperationNode.js";
import TensorNode from "../../TensorNode.js";
import type { GraphBuilder } from "../../GraphBuilder.js";
import type { ValueNode, KnownShape, StaticShape, ConcreteValueNode } from "../../OnnxTypes.js";
import { DataType } from "../../OnnxTypes.js";
import {
    asStaticDims,
    toStaticShape,
    UNKNOWN_SHAPE,
    topologicalSortOperationNodes,
    int64Vec,
} from "../../utils/index.js";
import { buildForLoopRegion, createAccumulatorCarry, type LoopScanOutput } from "./LoopBuilder.js";
import {
    buildLinearIndex,
    buildMixedStrides,
    decodeMixedRadix,
    extractMixedDims,
} from "./RecipeUtils.js";
import { lookupScheduleMap, matchesScheduleTarget, type ScheduleConfig } from "../Schedule.js";
import { OpCategory } from "../../schema/OpSchema.js";
import { OpRegistry } from "../../schema/OpRegistry.js";
import { TransformationOpportunity } from "../TransformationOpportunity.js";

import { IndexingMode } from "../../../DecompositionOptions.js";
import type { LoopLoweringRecipe } from "./LoopLoweringRecipe.js";
import { createDefaultLoopLoweringRecipes } from "../recipes.js";

/**
 * Discovers single-op and fused multi-op loop-lowering chains in an `OnnxGraph`
 * and lowers each chain into a single ONNX `Loop` (supporting AGU pointer vs.
 * radix indexing, schedule axis permutations, and carry-tensor vs. scan outputs).
 */
export class LoopFusionMatcher {
    private boundsCache = new Map<
        string,
        { totalIters: number | ValueNode; carryShape: KnownShape | ValueNode }
    >();
    private recipes: LoopLoweringRecipe[];

    constructor(
        private options: {
            coalesce: boolean;
            fuse: boolean;
            indexingMode: IndexingMode;
            useScanOutputs?: boolean;
            fuseMultiOutput?: boolean;
            schedule?: ScheduleConfig;
        },
        customRecipes?: LoopLoweringRecipe[],
    ) {
        this.recipes =
            customRecipes !== undefined
                ? [...customRecipes]
                : createDefaultLoopLoweringRecipes(this.options.coalesce);

        this.recipes.sort((a, b) => b.priority - a.priority);
    }

    public findOpportunities(graph: OnnxGraph.Class): TransformationOpportunity[] {
        this.boundsCache.clear();
        const opportunities: TransformationOpportunity[] = [];

        // 1. Find the chains
        const chains = this.options.fuse
            ? this.findFuseableChains(graph)
            : this.findSingleOpChains(graph);

        // 2. Wrap each chain in a TransformationOpportunity
        for (const chain of chains) {
            const rootOp = chain[chain.length - 1];

            opportunities.push(
                new TransformationOpportunity(
                    "LoopLowering",
                    chain.map((o) => o.id).join(","), // The target is the whole chain!
                    `Fuse ${chain.length} ${chain.length == 1 ? "operation" : "operations"} into an ONNX Loop (Root: ${rootOp.type})`,
                    (builder: GraphBuilder) => {
                        // This closure executes the lowering when the opportunity is applied
                        this.lowerFlatLoopChain(graph, chain, builder);
                    },
                ),
            );
        }

        return opportunities;
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
            rawAxisPerm &&
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
                    if (ext) return ext;
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
                    if (ext) outerBuilder.replaceAllUsesWith(t, ext);
                }
            }
        }
    }

    /**
     * Lowers a topologically ordered `chain` of fuseable ops into a single ONNX `Loop`,
     * post-processing and reshaping all live cluster outputs back to their original shapes.
     */
    private lowerFlatLoopChain(
        graph: OnnxGraph.Class,
        chain: OperationNode.Class[],
        outerBuilder: GraphBuilder,
    ): void {
        const rootOp = chain[chain.length - 1];
        const rootRecipe = this.getRecipeFor(rootOp)!;

        // Fast-path for standalone recipes (e.g., SoftmaxLoop, DetLoop, TopKScalarLoop)
        if (chain.length === 1 && rootRecipe.lowerStandalone) {
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

        const bounds = this.getBoundsFor(rootOp);
        const totalIters = bounds.totalIters;
        const carryShape = bounds.carryShape as StaticShape;
        const isSelfCarry = rootRecipe.tags?.includes("self-carry") === true;
        const useScan = this.options.useScanOutputs === true && !isSelfCarry;
        const axisPerm = this.resolveAxisPermutation(chain, originalOutShape, isSelfCarry);

        // 1. Initialize Loop region
        const effectiveIndexingMode =
            axisPerm !== undefined ? IndexingMode.Radix : this.options.indexingMode;
        const { innerBuilder, trip, vInitial, loopOutput, loopOutputs, loopCtx, finalize } =
            buildForLoopRegion(
                outerBuilder,
                totalIters,
                elemTy,
                carryShape,
                `Loop_${rootOp.id}`,
                `lowering_${rootOp.id}`,
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
            const recipe = this.getRecipeFor(op)!;
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
        this.rewireAbsorbedShapeProbes(chain, chainSet, rootOp, seenOutIds, outerBuilder);

        // 4. Post-process, reshape, and reconnect each output of the cluster
        for (let j = 0; j < chainOutputs.length; j++) {
            const outT = chainOutputs[j];
            const outShapeJ: KnownShape = [...(outT.shape as KnownShape)];
            const outTyJ = outT.literalType || elemTy;
            let processedOut: ValueNode = rawLoopOutputs[j];
            if (j === 0 && rootRecipe.postProcess) {
                processedOut = rootRecipe.postProcess(rootOp, outerBuilder, processedOut);
            }

            const staticCarryTotal = Array.isArray(carryShape)
                ? (carryShape as number[]).reduce((a, b) => a * b, 1)
                : -1;
            const isScalarOutput = outShapeJ.length === 0 && staticCarryTotal === 1;
            const outIsDynamic =
                outShapeJ.includes(-1) || (outShapeJ.length === 0 && !isScalarOutput);

            let targetShapeNode: ValueNode;
            if (j === 0 && bounds.targetShape) {
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

    // --- Helper methods (copied exactly from your old Pass) ---
    private getRecipeFor(op: OperationNode.Class): LoopLoweringRecipe | undefined {
        const stopPhase = lookupScheduleMap(this.options.schedule?.stopAt, op);
        if (stopPhase === "none" || stopPhase === "canonicalization") {
            return undefined;
        }
        const preferred = lookupScheduleMap(this.options.schedule?.recipeOverrides, op);
        if (preferred !== undefined) {
            const match = this.recipes.find((r) => r.name === preferred && r.match(op));
            if (match) return match;
        }
        return this.recipes.find((recipe) => recipe.match(op));
    }

    private isSupported(op: OperationNode.Class): boolean {
        return this.getRecipeFor(op) !== undefined;
    }

    private getBoundsFor(op: OperationNode.Class): {
        totalIters: number | ValueNode;
        carryShape: KnownShape | ValueNode;
        targetShape?: KnownShape | ValueNode;
    } {
        if (this.boundsCache.has(op.id)) return this.boundsCache.get(op.id)!;

        const recipe = this.getRecipeFor(op)!;
        const outTensors = op.getOutgoers.targets.filterIs(TensorNode).toArray();
        const originalOutShape: KnownShape =
            outTensors.length > 0 ? [...(outTensors[0].shape as KnownShape)] : UNKNOWN_SHAPE;
        const safeOut = asStaticDims(originalOutShape);

        if (recipe.getLoopBounds) {
            const result = recipe.getLoopBounds(op, originalOutShape);
            this.boundsCache.set(op.id, result);
            return result;
        }

        if (
            (originalOutShape.includes(-1) || originalOutShape.length === 0) &&
            outTensors.length > 0
        ) {
            throw new Error(`Dynamic output shape missing bounds in recipe for ${op.type}`);
        }

        const totalIters = safeOut.length === 0 ? 1 : safeOut.reduce((a, b) => a * b, 1);
        const result = { totalIters, carryShape: [totalIters] };
        this.boundsCache.set(op.id, result);
        return result;
    }

    private compareIters(a: number | ValueNode, b: number | ValueNode): boolean {
        if (typeof a === "number" && typeof b === "number") return a === b;
        if (typeof a !== "number" && typeof b !== "number") return a.id === b.id;
        return false;
    }

    private compareCarryShapes(a: KnownShape | ValueNode, b: KnownShape | ValueNode): boolean {
        const isArrayA = Array.isArray(a),
            isArrayB = Array.isArray(b);
        if (isArrayA && isArrayB)
            return (a as KnownShape).join(",") === (b as KnownShape).join(",");
        if (!isArrayA && !isArrayB) return (a as ValueNode).id === (b as ValueNode).id;
        return false;
    }

    private findSingleOpChains(graph: OnnxGraph.Class): OperationNode.Class[][] {
        return topologicalSortOperationNodes(graph)
            .filter((op) => this.isSupported(op))
            .map((op) => [op]);
    }

    /**
     * Traces a chain of ElementWise ops whose non-scalar input determines the exact runtime shape,
     * enabling dynamic diamond DAG fusion without false broadcast merges.
     */
    private getUnaryShapeOrigin(
        tensor: ValueNode,
        opset: number,
        registry: OpRegistry,
        originCache: Map<string, string>,
        visiting = new Set<string>(),
    ): string {
        const cached = originCache.get(tensor.id);
        if (cached !== undefined) return cached;
        if (visiting.has(tensor.id)) return tensor.id;
        visiting.add(tensor.id);

        let result = tensor.id;
        const prod = tensor.is(TensorNode)
            ? tensor.incomers.sources.filterIs(OperationNode).first()
            : undefined;
        if (prod) {
            const schema = registry.get(prod.type, opset);
            if (schema?.category === OpCategory.ElementWise) {
                const outRank = (tensor.shape as unknown[] | undefined)?.length ?? 0;
                const nonScalarInputs = (prod.getInputs() ?? []).filter((inp) => {
                    const sh = inp.shape as (number | string)[] | undefined;
                    if (!sh || sh.length === 0) return false;
                    return !(sh.length <= outRank && sh.every((d) => d === 1));
                });
                if (nonScalarInputs.length >= 1) {
                    const origins = nonScalarInputs.map((inp) =>
                        this.getUnaryShapeOrigin(inp, opset, registry, originCache, visiting),
                    );
                    if (origins.every((o) => o === origins[0])) {
                        result = origins[0];
                    }
                }
            }
        }

        visiting.delete(tensor.id);
        originCache.set(tensor.id, result);
        return result;
    }

    private findFuseableChains(graph: OnnxGraph.Class): OperationNode.Class[][] {
        const chains: OperationNode.Class[][] = [];
        const ops = topologicalSortOperationNodes(graph);
        const visited = new Set<string>();
        const registry = OpRegistry.getInstance();
        const opset = graph.getTargetOpset();
        const originCache = new Map<string, string>();

        const isBarrierRecipe = (recipe: LoopLoweringRecipe): boolean =>
            recipe.tags?.includes("self-carry") === true ||
            recipe.postProcess !== undefined ||
            recipe.lowerStandalone !== undefined;

        const isStaticTensorShape = (t: TensorNode.Class | undefined): boolean => {
            if (!t || !Array.isArray(t.shape) || t.shape.length === 0) return false;
            return !(t.shape as (number | string)[]).includes(-1);
        };

        const boundsCompatible = (
            prod: OperationNode.Class,
            root: OperationNode.Class,
        ): boolean => {
            const prodOut = prod.getOutgoers.targets.filterIs(TensorNode).first();
            const rootOut = root.getOutgoers.targets.filterIs(TensorNode).first();
            if (!prodOut || !rootOut) return false;

            // Dynamic diamond check FIRST (pure read-only, zero graph mutations)
            if (
                (!isStaticTensorShape(prodOut) || !isStaticTensorShape(rootOut)) &&
                this.getUnaryShapeOrigin(prodOut, opset, registry, originCache) ===
                    this.getUnaryShapeOrigin(rootOut, opset, registry, originCache)
            ) {
                return true;
            }

            if (isStaticTensorShape(prodOut) && isStaticTensorShape(rootOut)) {
                const prodBounds = this.getBoundsFor(prod);
                const rootBounds = this.getBoundsFor(root);
                return (
                    this.compareIters(prodBounds.totalIters, rootBounds.totalIters) &&
                    this.compareCarryShapes(prodBounds.carryShape, rootBounds.carryShape)
                );
            }

            return false;
        };

        const fuseMulti = this.options.fuseMultiOutput === true;
        const computeAtPairs = this.options.schedule?.computeAtList ?? [];
        const hasComputeAt = (
            prod: OperationNode.Class,
            clusterOps: OperationNode.Class[],
        ): boolean =>
            computeAtPairs.some(
                (p) =>
                    matchesScheduleTarget(p.producer, prod) &&
                    clusterOps.some((c) => matchesScheduleTarget(p.consumer, c)),
            );

        // 0. Pre-form explicit Schedule.fuse(...) multi-op / multi-output clusters
        for (const groupTargets of this.options.schedule?.fuseGroups ?? []) {
            const members: OperationNode.Class[] = [];
            for (const op of ops) {
                if (visited.has(op.id)) continue;
                if (!groupTargets.some((t) => matchesScheduleTarget(t, op))) continue;
                const r = this.getRecipeFor(op);
                if (!r || isBarrierRecipe(r)) continue;
                if (registry.get(op.type, opset)?.category !== OpCategory.ElementWise) continue;
                members.push(op);
            }
            if (members.length >= 2) {
                const anchor = members[members.length - 1];
                const compatible = members.filter(
                    (m) => m.id === anchor.id || boundsCompatible(m, anchor),
                );
                if (compatible.length >= 2) {
                    compatible.forEach((m) => visited.add(m.id));
                    chains.push(this.localTopologicalSort(compatible));
                }
            }
        }

        // Grow convex single-output DAG clusters in reverse topological order (consumers before producers).
        // Supports diamond DAGs (e.g. Swish, Gelu, Reciprocal) while stopping cleanly at multi-output forks
        // unless fuseMultiOutput=true or explicitly scheduled via Schedule.fuse / Schedule.computeAt.
        for (let i = ops.length - 1; i >= 0; i--) {
            const root = ops[i];
            if (visited.has(root.id)) continue;
            const rootRecipe = this.getRecipeFor(root);
            if (rootRecipe === undefined) continue;

            if (rootRecipe.lowerStandalone) {
                visited.add(root.id);
                chains.push([root]);
                continue;
            }

            const clusterSet = new Set<string>([root.id]);
            const cluster: OperationNode.Class[] = [root];
            visited.add(root.id);

            const rootIsElementWise =
                registry.get(root.type, opset)?.category === OpCategory.ElementWise &&
                !isBarrierRecipe(rootRecipe);

            // Only chains whose consumers are elementwise can absorb compatible producers
            let grew = rootIsElementWise;
            while (grew) {
                grew = false;
                for (const member of [...cluster]) {
                    const inTensors =
                        member
                            .getInputs()
                            ?.filter(
                                (n) => n.is(TensorNode) && n.as(TensorNode).type === "intermediate",
                            ) ?? [];

                    for (const t of inTensors) {
                        const producers = t.incomers.sources.filterIs(OperationNode).toArray();
                        for (const prod of producers) {
                            if (visited.has(prod.id)) continue;
                            const prodRecipe = this.getRecipeFor(prod);
                            if (prodRecipe === undefined || isBarrierRecipe(prodRecipe)) continue;

                            const prodOuts = prod.getOutgoers.targets
                                .filterIs(TensorNode)
                                .toArray();
                            if (prodOuts.length !== 1) continue;
                            const isOutputAllowed = fuseMulti || hasComputeAt(prod, cluster);
                            if (prodOuts[0].type === "output" && !isOutputAllowed) continue;

                            const consumers = prodOuts[0].outgoers.targets
                                .filterIs(OperationNode)
                                .toArray();
                            if (consumers.length === 0) continue;

                            const outsideConsumers = consumers.filter((c) => !clusterSet.has(c.id));
                            if (outsideConsumers.length > 0) {
                                if (!fuseMulti) continue;
                                const canAbsorbSiblings = outsideConsumers.every((c) => {
                                    if (visited.has(c.id)) return false;
                                    const cr = this.getRecipeFor(c);
                                    const cSchema = registry.get(c.type, opset);
                                    return (
                                        cr !== undefined &&
                                        !isBarrierRecipe(cr) &&
                                        cSchema?.category === OpCategory.ElementWise &&
                                        boundsCompatible(c, root)
                                    );
                                });
                                if (!canAbsorbSiblings) continue;
                            }

                            const allClusterConsumersEW = consumers
                                .filter((c) => clusterSet.has(c.id))
                                .every(
                                    (c) =>
                                        registry.get(c.type, opset)?.category ===
                                        OpCategory.ElementWise,
                                );
                            if (!allClusterConsumersEW) continue;
                            if (!isStaticTensorShape(prodOuts[0])) {
                                const prodSig = toStaticShape(prodOuts[0].shape).join(",");
                                const hasSameShapeInput = (prod.getInputs() ?? []).some(
                                    (inp) => toStaticShape(inp.shape).join(",") === prodSig,
                                );
                                if (!hasSameShapeInput) continue;
                            }
                            if (!boundsCompatible(prod, root)) continue;

                            visited.add(prod.id);
                            clusterSet.add(prod.id);
                            cluster.push(prod);
                            for (const sib of outsideConsumers) {
                                visited.add(sib.id);
                                clusterSet.add(sib.id);
                                cluster.push(sib);
                            }
                            grew = true;
                        }
                    }
                }
            }

            chains.push(this.localTopologicalSort(cluster));
        }

        // Preserve topological execution order across clusters
        const pos = new Map<string, number>();
        ops.forEach((o, idx) => pos.set(o.id, idx));
        chains.sort(
            (a, b) => (pos.get(a[a.length - 1].id) ?? 0) - (pos.get(b[b.length - 1].id) ?? 0),
        );
        return chains;
    }

    private localTopologicalSort(ops: OperationNode.Class[]): OperationNode.Class[] {
        const visited = new Set<string>();
        const sorted: OperationNode.Class[] = [];

        const visit = (node: OperationNode.Class) => {
            if (visited.has(node.id)) return;
            visited.add(node.id);

            const inTensors = node.getInputs()?.filter((v) => v.is(TensorNode)) || [];
            for (const t of inTensors) {
                const producers = t.incomers.sources.filterIs(OperationNode).toArray();
                for (const prod of producers) {
                    if (ops.some((o) => o.id === prod.id)) visit(prod);
                }
            }
            sorted.push(node);
        };

        for (const op of ops) visit(op);
        return sorted;
    }
}
