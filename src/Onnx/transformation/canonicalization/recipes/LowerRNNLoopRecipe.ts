import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { getAttr, getStringAttr, makeTensorProto } from "../../../Utils.js";
import { createCapturedInput } from "../../loop-lowering/RecipeUtils.js";
import {
    applyClipIfNeeded,
    buildInitialState,
    buildTimestepGather,
    createLoopBodyShell,
    emitBodyOutput,
    emitDirectionLoop,
    finalizeRecurrentOutputs,
    isPresent,
    matchRecurrentLoopOp,
    prepareRecurrentInputs,
    sliceAxis,
} from "../RnnLoopUtils.js";

export class LowerRNNLoopRecipe implements DecompositionRecipe {
    public readonly name = "LowerRNNLoop";
    public readonly targetOp = "RNN";
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 10;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["loop", "variable-length"] as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = [
        "Loop",
        "Slice",
        "Squeeze",
        "Unsqueeze",
        "Transpose",
        "Gather",
        "MatMul",
        "Add",
        "Sub",
        "Concat",
        "Less",
        "Where",
        "Clip",
        "Identity",
        "Tanh",
    ];

    match(op: OperationNode.Class): boolean {
        return matchRecurrentLoopOp(op, "RNN", 1);
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const W = ins[1];
        const R = ins[2];
        const B = isPresent(ins[3]) ? ins[3] : undefined;
        const initialH = isPresent(ins[5]) ? ins[5] : undefined;

        const prep = prepareRecurrentInputs(op, builder, 1, "rnnl");
        const direction = getStringAttr(op, "direction", "forward");
        const rawActs = getAttr(op, "activations");
        const activations = Array.isArray(rawActs) ? (rawActs as string[]) : [];
        const bhShape: KnownShape = [prep.batchSize, prep.hiddenSize];
        const bhMeta = [{ type: prep.dtype, shape: bhShape }];

        const dirYs: ConcreteValueNode[] = [];
        const dirYhs: ConcreteValueNode[] = [];

        for (let d = 0; d < prep.numDirs; d++) {
            const isReverse = direction === "reverse" || (direction === "bidirectional" && d === 1);
            const actOp = activations[d] ?? "Tanh";

            const wd3 = sliceAxis(
                builder,
                W,
                prep.ax0,
                prep.step1,
                d,
                d + 1,
                [1, prep.hiddenSize, prep.inputSize],
                `rnnl_w_${d}_${op.id}`,
            );
            const [wd] = builder.createOp("Squeeze", [wd3, prep.ax0], {}, [
                { type: prep.dtype, shape: [prep.hiddenSize, prep.inputSize] },
            ]);
            const [wdT] = builder.createOp("Transpose", [wd], { perm: [1, 0] }, [
                { type: prep.dtype, shape: [prep.inputSize, prep.hiddenSize] },
            ]);

            const rd3 = sliceAxis(
                builder,
                R,
                prep.ax0,
                prep.step1,
                d,
                d + 1,
                [1, prep.hiddenSize, prep.hiddenSize],
                `rnnl_r_${d}_${op.id}`,
            );
            const [rd] = builder.createOp("Squeeze", [rd3, prep.ax0], {}, [
                { type: prep.dtype, shape: [prep.hiddenSize, prep.hiddenSize] },
            ]);
            const [rdT] = builder.createOp("Transpose", [rd], { perm: [1, 0] }, [
                { type: prep.dtype, shape: [prep.hiddenSize, prep.hiddenSize] },
            ]);

            let biasD: ConcreteValueNode | undefined;
            if (B !== undefined) {
                const bd = sliceAxis(
                    builder,
                    B,
                    prep.ax0,
                    prep.step1,
                    d,
                    d + 1,
                    [1, 2 * prep.hiddenSize],
                    `rnnl_b_${d}_${op.id}`,
                );
                const wb = sliceAxis(
                    builder,
                    bd,
                    prep.ax1,
                    prep.step1,
                    0,
                    prep.hiddenSize,
                    [1, prep.hiddenSize],
                    `rnnl_wb_${d}_${op.id}`,
                );
                const rb = sliceAxis(
                    builder,
                    bd,
                    prep.ax1,
                    prep.step1,
                    prep.hiddenSize,
                    2 * prep.hiddenSize,
                    [1, prep.hiddenSize],
                    `rnnl_rb_${d}_${op.id}`,
                );
                [biasD] = builder.createOp("Add", [wb, rb], {}, [
                    { type: prep.dtype, shape: [1, prep.hiddenSize] },
                ]);
            }

            const h0 = buildInitialState(builder, prep, initialH, d, `rnnl_h0_${op.id}`);

            const { loopBody, innerBuilder, iterNode, condIn, stateIns } = createLoopBodyShell(
                prep,
                1,
                `rnnl_body_${d}_${op.id}`,
            );
            const HIn = stateIns[0];
            const { xt, activeMask } = buildTimestepGather(
                loopBody,
                innerBuilder,
                prep,
                iterNode,
                isReverse,
                `rnnl_t_${d}_${op.id}`,
            );

            const wdTCap = createCapturedInput(loopBody, wdT);
            const rdTCap = createCapturedInput(loopBody, rdT);

            const [xw] = innerBuilder.createOp("MatMul", [xt, wdTCap], {}, bhMeta);
            const [hr] = innerBuilder.createOp("MatMul", [HIn, rdTCap], {}, bhMeta);
            let [preAct] = innerBuilder.createOp("Add", [xw, hr], {}, bhMeta);
            if (biasD !== undefined) {
                const bCap = createCapturedInput(loopBody, biasD);
                [preAct] = innerBuilder.createOp("Add", [preAct, bCap], {}, bhMeta);
            }
            preAct = applyClipIfNeeded(
                innerBuilder,
                preAct,
                prep.dtype,
                bhShape,
                prep.clipVal,
                `rnnl_clip_${d}_${op.id}`,
            );
            const [hCand] = innerBuilder.createOp(actOp, [preAct], {}, bhMeta);

            let hNext: ConcreteValueNode = hCand;
            let yStep: ConcreteValueNode = hCand;
            if (activeMask !== undefined) {
                const zeroC = innerBuilder.createConstant(
                    `rnnl_z_${d}_${op.id}`,
                    makeTensorProto(prep.dtype, [], [0]),
                );
                [hNext] = innerBuilder.createOp("Where", [activeMask, hCand, HIn], {}, bhMeta);
                [yStep] = innerBuilder.createOp("Where", [activeMask, hCand, zeroC], {}, bhMeta);
            }

            emitBodyOutput(loopBody, condIn, "cond_out");
            emitBodyOutput(loopBody, hNext, "h_out");
            emitBodyOutput(loopBody, yStep, "y_scan_out");

            const { finalStates, yDir4D } = emitDirectionLoop(
                builder,
                prep,
                loopBody,
                [h0],
                isReverse,
                `rnnl_d${d}_${op.id}`,
            );
            dirYs.push(yDir4D);
            const [yhDir] = builder.createOp("Unsqueeze", [finalStates[0], prep.ax0], {}, [
                { type: prep.dtype, shape: [1, prep.batchSize, prep.hiddenSize] as KnownShape },
            ]);
            dirYhs.push(yhDir);
        }

        finalizeRecurrentOutputs(op, builder, prep, dirYs, dirYhs);
    }
}
