import type OnnxGraph from "../OnnxGraph.js";
import type { GraphPass } from "../PassManager.js";
import type OperationNode from "../OperationNode.js";
import ConstantNode from "../ConstantNode.js";
import { DataType, type ConcreteValueNode, type KnownShape } from "../OnnxTypes.js";
import { OpRegistry } from "../schema/OpRegistry.js";
import { OpCategory } from "../schema/OpSchema.js";
import type { IndexingMode } from "../../DecompositionOptions.js";
import { readTensorData, toStaticShape } from "../utils/index.js";
import { TrackedGraphBuilder } from "./tracking/TrackedGraphBuilder.js";
import type { HistoryManager } from "./tracking/HistoryManager.js";
import { CanonicalizationPass } from "./canonicalization/CanonicalizationPass.js";
import { InferShapesPass } from "./InferShapesPass.js";
import { LoopLoweringPass } from "./loop-lowering/LoopLoweringPass.js";
import type { ScheduleConfig } from "./Schedule.js";

const POST_LOOP_SCALAR_OPS = new Set(["Div", "Reciprocal", "Sqrt", "Log", "Ceil"]);
const POST_LOOP_TENSOR_OPS = new Set(["Mul", "Add", "Sub", "Pad"]);

export interface PostLoopLoweringOptions {
    loopLowering: boolean;
    coalesce: boolean;
    fuse: boolean;
    indexingMode: IndexingMode;
    useScanOutputs: boolean;
    fuseMultiOutput: boolean;
    schedule?: ScheduleConfig;
}

/**
 * Post-loop lowering pass executed when `canonicalize && loopLowering && recurse`
 * are enabled.
 *
 * Loop lowering can synthesize depth-0 helper ops (such as float `Div`/`Sqrt`,
 * tensor `Add`/`Mul`/`Pad`, or zero-fill `Expand` carry buffers). This pass
 * canonicalizes and re-lowers any remaining depth-0 non-native ops, then lowers
 * zero-fill `Expand(0, shape)` carry allocations into `ConstantOfShape (+ Cast)`.
 */
export class PostLoopLoweringPass implements GraphPass {
    public readonly name = "PostLoopDepth0Lower";

    constructor(private readonly options: PostLoopLoweringOptions) {}

    public run(graph: OnnxGraph.Class, historyManager: HistoryManager): boolean {
        let changed = false;

        if (this.hasPostLoopTarget(graph)) {
            const canonPass = new CanonicalizationPass(undefined, {
                loopLowering: this.options.loopLowering,
                recurse: false,
                ...(this.options.schedule !== undefined ? { schedule: this.options.schedule } : {}),
            });
            if (canonPass.run(graph, historyManager)) {
                changed = true;
            }
            new InferShapesPass().run(graph);

            if (this.needsSecondLoopPass(graph)) {
                const loopPass = new LoopLoweringPass({
                    coalesce: this.options.coalesce,
                    fuse: this.options.fuse,
                    indexingMode: this.options.indexingMode,
                    useScanOutputs: this.options.useScanOutputs,
                    fuseMultiOutput: this.options.fuseMultiOutput,
                    ...(this.options.schedule !== undefined
                        ? { schedule: this.options.schedule }
                        : {}),
                });
                if (loopPass.run(graph, historyManager)) {
                    changed = true;
                }
            }
        }

        if (this.lowerZeroFillExpands(graph, historyManager)) {
            changed = true;
        }

        return changed;
    }

    private isFloatType(dt?: DataType): boolean {
        return (
            dt === undefined ||
            dt === DataType.UNDEFINED ||
            dt === DataType.FLOAT ||
            dt === DataType.DOUBLE ||
            dt === DataType.FLOAT16 ||
            dt === DataType.BFLOAT16
        );
    }

    private isNon0DOp(op: OperationNode.Class): boolean {
        return (op.getInputs() ?? []).some(
            (inp) => Array.isArray(inp.shape) && inp.shape.length > 0,
        );
    }

    private isZeroFillExpand(op: OperationNode.Class): boolean {
        if (op.type !== "Expand") return false;
        const ins = op.getInputs() ?? [];
        if (ins.length < 2 || !ins[0].is(ConstantNode)) return false;
        const vals = readTensorData(ins[0].as(ConstantNode));
        return vals !== undefined && vals.length > 0 && vals.every((v) => v === 0);
    }

    private hasPostLoopTarget(graph: OnnxGraph.Class): boolean {
        return graph
            .getOperationNodes()
            .toArray()
            .some((op) => {
                if (POST_LOOP_SCALAR_OPS.has(op.type)) {
                    return this.isFloatType(op.getInputs()?.[0]?.literalType);
                }
                if (POST_LOOP_TENSOR_OPS.has(op.type)) {
                    return this.isFloatType(op.getInputs()?.[0]?.literalType) && this.isNon0DOp(op);
                }
                return op.type === "Expand" && !this.isZeroFillExpand(op);
            });
    }

    private needsSecondLoopPass(graph: OnnxGraph.Class): boolean {
        const registry = OpRegistry.getInstance();
        const opset = graph.getTargetOpset();
        return graph
            .getOperationNodes()
            .toArray()
            .some((op) => {
                if (op.type === "Loop" || op.type === "If" || op.type === "Scan") {
                    return false;
                }
                if (!this.isNon0DOp(op)) return false;
                if (op.type === "Pad") return true;
                const cat = registry.get(op.type, opset)?.category;
                return (
                    cat === OpCategory.ElementWise &&
                    this.isFloatType(op.getInputs()?.[0]?.literalType)
                );
            });
    }

    /**
     * Replaces zero-fill `Expand(zeros, shape)` carry-buffer allocations with
     * native `ConstantOfShape` (+ optional `Cast` for non-float element types).
     */
    private lowerZeroFillExpands(graph: OnnxGraph.Class, historyManager: HistoryManager): boolean {
        let changed = false;

        for (const op of graph.getOperationNodes().toArray()) {
            if (!this.isZeroFillExpand(op)) continue;
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
                graph,
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
                [rep] = tb.createOp("Cast", [rep], { to: dt }, [{ type: dt, shape: outShape }]);
            }
            tb.replaceAllUsesWith(out, rep);
            tb.removeNode(op);
            historyManager.pushPatch(tb.commitPatch());
            changed = true;
        }

        return changed;
    }
}
