import type Graph from "@specs-feup/flow/graph/Graph";
import type OnnxGraph from "../OnnxGraph.js";
import { initializeSchemaRegistry } from "../Schema/index.js";
import { PassManager } from "../PassManager.js";
import { CanonicalizationPass } from "./canonicalization/CanonicalizationPass.js";
import { InferShapesPass } from "./InferShapesPass.js";
import { LoopLoweringPass } from "./loop-lowering/LoopLoweringPass.js";
import transformForCgra from "./cgra-decomposition-example/TransformForCgra.js";
import { HistoryManager } from "./tracking/HistoryManager.js";
import type { DecompositionOptions, IndexingMode } from "../../DecompositionOptions.js";
import { defaultDecompositionOptions } from "../../DecompositionOptions.js";
import { LoopTilingPass } from "./loop-lowering/LoopTilingPass.js";
import { ShapeOptimizationPass } from "./optimization/ShapeOptimizationPass.js";
import { DeadCodeEliminationPass } from "./optimization/DeadCodeEliminationPass.js";
import { Schedule, resolveScheduleConfig, type ScheduleConfig } from "./Schedule.js";
import { DataType, type ConcreteValueNode, type KnownShape } from "../OnnxTypes.js";
import type OperationNode from "../OperationNode.js";
import ConstantNode from "../ConstantNode.js";
import { OpRegistry } from "../Schema/OpRegistry.js";
import { OpCategory } from "../Schema/OpSchema.js";
import { TrackedGraphBuilder } from "./tracking/TrackedGraphBuilder.js";
import { readTensorData, toStaticShape } from "../Utils.js";

export default class OnnxGraphTransformer implements Graph.Transformation<
    OnnxGraph.Class,
    OnnxGraph.Class
> {
    private canonicalize: boolean;
    private fuse: boolean;
    private recurse: boolean;
    private coalesce: boolean;
    private loopLowering: boolean;
    private optimize: boolean;
    private decomposeForCgra: boolean;
    private indexingMode: IndexingMode;
    private tileSize?: number | undefined;
    private targetOpset?: number | undefined;
    private irVersion?: number | undefined;
    private useScanOutputs: boolean;
    private fuseMultiOutput: boolean;
    private scheduleConfig?: ScheduleConfig | undefined;

    // Overload signatures (for TypeScript type checking)
    constructor();
    constructor(options: Partial<DecompositionOptions>);
    constructor(
        canonicalize: boolean,
        fuse: boolean,
        recurse: boolean,
        coalesce: boolean,
        optimize: boolean,
        decomposeForCgra: boolean,
        loopLowering?: boolean,
        indexingMode?: IndexingMode,
        tileSize?: number,
    );

    constructor(
        fuseOrOptions: boolean | Partial<DecompositionOptions> = defaultDecompositionOptions.fuse,
        canonicalize: boolean = defaultDecompositionOptions.canonicalize,
        recurse: boolean = defaultDecompositionOptions.recurse,
        coalesce: boolean = defaultDecompositionOptions.coalesce,
        optimize: boolean = defaultDecompositionOptions.optimize,
        loopLowering: boolean = defaultDecompositionOptions.loopLowering,
        decomposeForCgra: boolean = defaultDecompositionOptions.decomposeForCgra,
        indexingMode: IndexingMode = defaultDecompositionOptions.indexingMode,
        tileSize?: number,
    ) {
        if (typeof fuseOrOptions === "boolean") {
            this.canonicalize = canonicalize;
            this.fuse = fuseOrOptions;
            this.recurse = recurse;
            this.coalesce = coalesce;
            this.loopLowering = loopLowering;
            this.optimize = optimize;
            this.decomposeForCgra = decomposeForCgra;
            this.indexingMode = indexingMode;
            this.tileSize = tileSize;
            this.useScanOutputs = defaultDecompositionOptions.useScanOutputs ?? false;
            this.fuseMultiOutput = defaultDecompositionOptions.fuseMultiOutput ?? false;
        } else {
            this.canonicalize =
                fuseOrOptions.canonicalize ?? defaultDecompositionOptions.canonicalize;
            this.fuse = fuseOrOptions.fuse ?? defaultDecompositionOptions.fuse;
            this.recurse = fuseOrOptions.recurse ?? defaultDecompositionOptions.recurse;
            this.coalesce = fuseOrOptions.coalesce ?? defaultDecompositionOptions.coalesce;
            this.loopLowering =
                fuseOrOptions.loopLowering ?? defaultDecompositionOptions.loopLowering;
            this.optimize = fuseOrOptions.optimize ?? defaultDecompositionOptions.optimize;
            this.decomposeForCgra =
                fuseOrOptions.decomposeForCgra ?? defaultDecompositionOptions.decomposeForCgra;
            this.indexingMode =
                fuseOrOptions.indexingMode ?? defaultDecompositionOptions.indexingMode;
            this.tileSize = fuseOrOptions.tileSize ?? defaultDecompositionOptions.tileSize;
            this.targetOpset = fuseOrOptions.targetOpset;
            this.irVersion = fuseOrOptions.irVersion;
            this.useScanOutputs =
                fuseOrOptions.useScanOutputs ?? defaultDecompositionOptions.useScanOutputs ?? false;
            this.fuseMultiOutput =
                fuseOrOptions.fuseMultiOutput ??
                defaultDecompositionOptions.fuseMultiOutput ??
                false;
            this.scheduleConfig = resolveScheduleConfig(fuseOrOptions.schedule);
        }
    }

    apply(graph: OnnxGraph.Class, printSummary: boolean = true): OnnxGraph.Class {
        initializeSchemaRegistry();
        if (this.targetOpset !== undefined) {
            graph.setTargetOpset(this.targetOpset, true);
        }
        if (this.irVersion !== undefined) {
            graph.setIrVersion(this.irVersion, true);
        }
        const pm = new PassManager();
        const historyManager = new HistoryManager(graph);

        if (this.decomposeForCgra) {
            // =========================================================================
            // PHASE 0: CGRA Decomposition Example
            // =========================================================================
            // We run a simple CGRA Decomposition example instead of the other transformations
            pm.addPass({
                name: "TransformForCgraExample",
                run: (g) => {
                    transformForCgra(g);
                    return false; // Return false so the PassManager doesn't loop it infinitely
                },
            });
            pm.run(graph, historyManager);
            return graph;
        }

        const scheduleCfg = this.scheduleConfig;
        if (scheduleCfg?.cacheReads && scheduleCfg.cacheReads.length > 0) {
            pm.addPass({
                name: "ScheduleCacheRead",
                run: (g, hm) => Schedule.applyCacheReads(g, scheduleCfg, hm),
            });
        }

        if (this.canonicalize) {
            // =========================================================================
            // PHASE 1: Normalization (Canonicalization)
            // =========================================================================
            // We run this first to break down complex ops (like Gemm -> MatMul + Add)
            // or Softmax into their primitive mathematical components.
            pm.addPass(
                new CanonicalizationPass(undefined, {
                    loopLowering: this.loopLowering,
                    recurse: this.recurse,
                    ...(scheduleCfg !== undefined ? { schedule: scheduleCfg } : {}),
                }),
            );

            // Sweep dangling dead nodes left by canonicalization (e.g. final unused bits in
            // Exp/Reciprocal range reduction) before LoopLowering fusion inspects consumer graphs.
            if (this.optimize) {
                pm.addPass(new DeadCodeEliminationPass());
            }

            // Ensure shapes are strictly inferred before we try to lower anything,
            // as our loop recipes rely heavily on broadcasting shape math.
            pm.addPass(new InferShapesPass());
        }

        if (this.loopLowering) {
            // =========================================================================
            // PHASE 2: Lowering & Early Fusion
            // =========================================================================
            // This looks at our clean, primitive graph and finds chains (Transpose -> Add)
            // and lowers them directly into single, highly optimized ONNX Loops.
            pm.addPass(
                new LoopLoweringPass({
                    coalesce: this.coalesce,
                    fuse: this.fuse,
                    indexingMode: this.indexingMode,
                    useScanOutputs: this.useScanOutputs,
                    fuseMultiOutput: this.fuseMultiOutput,
                    ...(scheduleCfg !== undefined ? { schedule: scheduleCfg } : {}),
                }),
            );

            if (this.canonicalize && this.recurse) {
                pm.addPass({
                    name: "PostLoopDepth0Lower",
                    run: (g, hm) => {
                        const POST_LOOP_SCALAR_OPS = new Set([
                            "Div",
                            "Reciprocal",
                            "Sqrt",
                            "Log",
                            "Ceil",
                        ]);
                        const POST_LOOP_TENSOR_OPS = new Set(["Mul", "Add", "Sub", "Pad"]);
                        const isFloat = (dt?: DataType): boolean =>
                            dt === undefined ||
                            dt === DataType.UNDEFINED ||
                            dt === DataType.FLOAT ||
                            dt === DataType.DOUBLE ||
                            dt === DataType.FLOAT16 ||
                            dt === DataType.BFLOAT16;
                        const isNon0D = (op: OperationNode.Class): boolean =>
                            (op.getInputs() ?? []).some(
                                (inp) => Array.isArray(inp.shape) && inp.shape.length > 0,
                            );
                        const isZeroFillExpand = (op: OperationNode.Class): boolean => {
                            if (op.type !== "Expand") return false;
                            const ins = op.getInputs() ?? [];
                            if (ins.length < 2 || !ins[0].is(ConstantNode)) return false;
                            const vals = readTensorData(ins[0].as(ConstantNode));
                            return Boolean(vals && vals.length > 0 && vals.every((v) => v === 0));
                        };

                        let changed = false;

                        const hasPostLoopTarget = g
                            .getOperationNodes()
                            .toArray()
                            .some((op) => {
                                if (POST_LOOP_SCALAR_OPS.has(op.type)) {
                                    return isFloat(op.getInputs()?.[0]?.literalType);
                                }
                                if (POST_LOOP_TENSOR_OPS.has(op.type)) {
                                    return isFloat(op.getInputs()?.[0]?.literalType) && isNon0D(op);
                                }
                                if (op.type === "Expand" && !isZeroFillExpand(op)) {
                                    return true;
                                }
                                return false;
                            });

                        if (hasPostLoopTarget) {
                            const cPass = new CanonicalizationPass(undefined, {
                                loopLowering: this.loopLowering,
                                recurse: false,
                                ...(scheduleCfg !== undefined ? { schedule: scheduleCfg } : {}),
                            });
                            if (cPass.run(g, hm)) changed = true;
                            new InferShapesPass().run(g);

                            const registry = OpRegistry.getInstance();
                            const opset = g.getTargetOpset();
                            const needsSecondLoop = g
                                .getOperationNodes()
                                .toArray()
                                .some((op) => {
                                    if (
                                        op.type === "Loop" ||
                                        op.type === "If" ||
                                        op.type === "Scan"
                                    ) {
                                        return false;
                                    }
                                    if (!isNon0D(op)) return false;
                                    if (op.type === "Pad") return true;
                                    const cat = registry.get(op.type, opset)?.category;
                                    return (
                                        cat === OpCategory.ElementWise &&
                                        isFloat(op.getInputs()?.[0]?.literalType)
                                    );
                                });
                            if (needsSecondLoop) {
                                const lPass = new LoopLoweringPass({
                                    coalesce: this.coalesce,
                                    fuse: this.fuse,
                                    indexingMode: this.indexingMode,
                                    useScanOutputs: this.useScanOutputs,
                                    fuseMultiOutput: this.fuseMultiOutput,
                                    ...(scheduleCfg !== undefined ? { schedule: scheduleCfg } : {}),
                                });
                                if (lPass.run(g, hm)) changed = true;
                            }
                        }

                        // Lower zero-fill Expand(zeros, shape) -> ConstantOfShape (+Cast) last
                        // so both Phase 2 and second-pass carry buffers use native primitives.
                        for (const op of g.getOperationNodes().toArray()) {
                            if (!isZeroFillExpand(op)) continue;
                            const ins = op.getInputs()!;
                            const outs = op.getOutputs();
                            if (outs.length === 0) continue;
                            const out = outs[0];
                            const dt =
                                (ins[0].literalType as DataType | undefined) ??
                                (out.literalType as DataType | undefined) ??
                                DataType.FLOAT;
                            const outShape = toStaticShape(out.shape) as KnownShape;
                            const tb = new TrackedGraphBuilder(
                                g,
                                `PostLoopExpand_${op.id}`,
                                `Lower zero-fill Expand ${op.id} to ConstantOfShape`,
                                "post_loop",
                            );
                            let rep: ConcreteValueNode = tb.createOp(
                                "ConstantOfShape",
                                [ins[1] as ConcreteValueNode],
                                {},
                                [{ type: DataType.FLOAT, shape: outShape }],
                            )[0];
                            if (dt !== DataType.FLOAT) {
                                [rep] = tb.createOp("Cast", [rep], { to: dt }, [
                                    { type: dt, shape: outShape },
                                ]);
                            }
                            tb.replaceAllUsesWith(out, rep);
                            tb.removeNode(op);
                            hm.pushPatch(tb.commitPatch());
                            changed = true;
                        }
                        return changed;
                    },
                });
                // Infer 0D shapes inside newly emitted Loop bodies, then recursively canonicalize
                // any remaining non-native scalar compute ops (Exp, Div, Sqrt, Log, etc.).
                pm.addPass(new InferShapesPass());
                pm.addPass(
                    new CanonicalizationPass(undefined, {
                        loopLowering: false,
                        recurse: true,
                        regionOnly: true,
                        scalar0DOnly: true,
                        ...(scheduleCfg !== undefined ? { schedule: scheduleCfg } : {}),
                    }),
                );
            }

            if (this.tileSize !== undefined && this.tileSize > 1) {
                pm.addPass(new LoopTilingPass({ tileSize: this.tileSize }));
            }
        }

        if (scheduleCfg) {
            pm.addPass({
                name: "ScheduleLoopTransforms",
                run: (g, hm) => Schedule.applyLoopTransforms(g, scheduleCfg, hm),
            });
        }

        // =========================================================================
        // PHASE 3: Final Cleanup
        // =========================================================================
        // The LoopLoweringPass leaves the original, un-looped operations (like the
        // outer Transpose and Add nodes) dangling in the graph. A final DCE sweeps
        // them away, leaving only the beautifully fused Loop!
        if (this.optimize) {
            pm.addPass(new DeadCodeEliminationPass());
            pm.addPass(new ShapeOptimizationPass());
        }

        pm.addPass(new InferShapesPass());

        // Execute the pipeline!
        pm.run(graph, historyManager);
        if (printSummary) {
            historyManager.printSummary();
        }

        return graph;
    }
}
