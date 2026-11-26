import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import {
    getAttr,
    getIntAttr,
    getStringAttr,
    int64Vec,
    makeTensorProto,
} from "../../../utils/index.js";
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

export class LowerLSTMLoopRecipe implements DecompositionRecipe {
    public readonly name = "LowerLSTMLoop";
    public readonly targetOp = "LSTM";
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
        "Mul",
        "Concat",
        "Less",
        "Where",
        "Clip",
        "Identity",
        "Sigmoid",
        "Tanh",
    ];

    match(op: OperationNode.Class): boolean {
        return matchRecurrentLoopOp(op, "LSTM", 4);
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const W = ins[1];
        const R = ins[2];
        const B = isPresent(ins[3]) ? ins[3] : undefined;
        const initialH = isPresent(ins[5]) ? ins[5] : undefined;
        const initialC = isPresent(ins[6]) ? ins[6] : undefined;
        const P = isPresent(ins[7]) ? ins[7] : undefined;

        const prep = prepareRecurrentInputs(op, builder, 4, "lstml");
        const direction = getStringAttr(op, "direction", "forward");
        const inputForget = getIntAttr(op, "input_forget", 0);
        const rawActs = getAttr(op, "activations");
        const activations = Array.isArray(rawActs) ? (rawActs as string[]) : [];
        const H = prep.hiddenSize;
        const bhShape: KnownShape = [prep.batchSize, H];
        const bhMeta = [{ type: prep.dtype, shape: bhShape }];
        const b4hMeta = [{ type: prep.dtype, shape: [prep.batchSize, 4 * H] as KnownShape }];

        const dirYs: ConcreteValueNode[] = [];
        const dirYhs: ConcreteValueNode[] = [];
        const dirYcs: ConcreteValueNode[] = [];

        for (let d = 0; d < prep.numDirs; d++) {
            const isReverse = direction === "reverse" || (direction === "bidirectional" && d === 1);
            const fAct = activations[3 * d] ?? "Sigmoid";
            const gAct = activations[3 * d + 1] ?? "Tanh";
            const hAct = activations[3 * d + 2] ?? "Tanh";

            const wd3 = sliceAxis(
                builder,
                W,
                prep.ax0,
                prep.step1,
                d,
                d + 1,
                [1, 4 * H, prep.inputSize],
                `lstml_w_${d}_${op.id}`,
            );
            const [wd] = builder.createOp("Squeeze", [wd3, prep.ax0], {}, [
                { type: prep.dtype, shape: [4 * H, prep.inputSize] },
            ]);
            const [wdT] = builder.createOp("Transpose", [wd], { perm: [1, 0] }, [
                { type: prep.dtype, shape: [prep.inputSize, 4 * H] },
            ]);

            const rd3 = sliceAxis(
                builder,
                R,
                prep.ax0,
                prep.step1,
                d,
                d + 1,
                [1, 4 * H, H],
                `lstml_r_${d}_${op.id}`,
            );
            const [rd] = builder.createOp("Squeeze", [rd3, prep.ax0], {}, [
                { type: prep.dtype, shape: [4 * H, H] },
            ]);
            const [rdT] = builder.createOp("Transpose", [rd], { perm: [1, 0] }, [
                { type: prep.dtype, shape: [H, 4 * H] },
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
                    [1, 8 * H],
                    `lstml_b_${d}_${op.id}`,
                );
                const wb = sliceAxis(
                    builder,
                    bd,
                    prep.ax1,
                    prep.step1,
                    0,
                    4 * H,
                    [1, 4 * H],
                    `lstml_wb_${d}_${op.id}`,
                );
                const rb = sliceAxis(
                    builder,
                    bd,
                    prep.ax1,
                    prep.step1,
                    4 * H,
                    8 * H,
                    [1, 4 * H],
                    `lstml_rb_${d}_${op.id}`,
                );
                [biasD] = builder.createOp("Add", [wb, rb], {}, [
                    { type: prep.dtype, shape: [1, 4 * H] },
                ]);
            }

            let pi: ConcreteValueNode | undefined;
            let po: ConcreteValueNode | undefined;
            let pf: ConcreteValueNode | undefined;
            if (P !== undefined) {
                const pd = sliceAxis(
                    builder,
                    P,
                    prep.ax0,
                    prep.step1,
                    d,
                    d + 1,
                    [1, 3 * H],
                    `lstml_p_${d}_${op.id}`,
                );
                pi = sliceAxis(
                    builder,
                    pd,
                    prep.ax1,
                    prep.step1,
                    0,
                    H,
                    [1, H],
                    `lstml_pi_${d}_${op.id}`,
                );
                po = sliceAxis(
                    builder,
                    pd,
                    prep.ax1,
                    prep.step1,
                    H,
                    2 * H,
                    [1, H],
                    `lstml_po_${d}_${op.id}`,
                );
                pf = sliceAxis(
                    builder,
                    pd,
                    prep.ax1,
                    prep.step1,
                    2 * H,
                    3 * H,
                    [1, H],
                    `lstml_pf_${d}_${op.id}`,
                );
            }

            const h0 = buildInitialState(builder, prep, initialH, d, `lstml_h0_${op.id}`);
            const c0 = buildInitialState(builder, prep, initialC, d, `lstml_c0_${op.id}`);

            const { loopBody, innerBuilder, iterNode, condIn, stateIns } = createLoopBodyShell(
                prep,
                2,
                `lstml_body_${d}_${op.id}`,
            );
            const HIn = stateIns[0];
            const CIn = stateIns[1];
            const { xt, activeMask } = buildTimestepGather(
                loopBody,
                innerBuilder,
                prep,
                iterNode,
                isReverse,
                `lstml_t_${d}_${op.id}`,
            );

            const innerAx1 = innerBuilder.createConstant(`lstml_iax1_${d}_${op.id}`, int64Vec([1]));
            const innerStep1 = innerBuilder.createConstant(
                `lstml_ist1_${d}_${op.id}`,
                int64Vec([1]),
            );
            const oneC = innerBuilder.createConstant(
                `lstml_one_${d}_${op.id}`,
                makeTensorProto(prep.dtype, [], [1]),
            );

            // Combined gates [batchSize, 4*H] in ONNX order: [i, o, f, c]
            const [xw] = innerBuilder.createOp(
                "MatMul",
                [xt, createCapturedInput(loopBody, wdT)],
                {},
                b4hMeta,
            );
            const [hr] = innerBuilder.createOp(
                "MatMul",
                [HIn, createCapturedInput(loopBody, rdT)],
                {},
                b4hMeta,
            );
            let [gates] = innerBuilder.createOp("Add", [xw, hr], {}, b4hMeta);
            if (biasD !== undefined) {
                [gates] = innerBuilder.createOp(
                    "Add",
                    [gates, createCapturedInput(loopBody, biasD)],
                    {},
                    b4hMeta,
                );
            }

            let gi = sliceAxis(
                innerBuilder,
                gates,
                innerAx1,
                innerStep1,
                0,
                H,
                [prep.batchSize, H],
                `lstml_gi_${d}_${op.id}`,
            );
            let go = sliceAxis(
                innerBuilder,
                gates,
                innerAx1,
                innerStep1,
                H,
                2 * H,
                [prep.batchSize, H],
                `lstml_go_${d}_${op.id}`,
            );
            let gf = sliceAxis(
                innerBuilder,
                gates,
                innerAx1,
                innerStep1,
                2 * H,
                3 * H,
                [prep.batchSize, H],
                `lstml_gf_${d}_${op.id}`,
            );
            let gc = sliceAxis(
                innerBuilder,
                gates,
                innerAx1,
                innerStep1,
                3 * H,
                4 * H,
                [prep.batchSize, H],
                `lstml_gc_${d}_${op.id}`,
            );

            if (pi !== undefined) {
                const [piC] = innerBuilder.createOp(
                    "Mul",
                    [createCapturedInput(loopBody, pi), CIn],
                    {},
                    bhMeta,
                );
                [gi] = innerBuilder.createOp("Add", [gi, piC], {}, bhMeta);
            }
            gi = applyClipIfNeeded(
                innerBuilder,
                gi,
                prep.dtype,
                bhShape,
                prep.clipVal,
                `lstml_ci_${d}_${op.id}`,
            );
            const [it] = innerBuilder.createOp(fAct, [gi], {}, bhMeta);

            let ft: ConcreteValueNode;
            if (inputForget !== 0) {
                [ft] = innerBuilder.createOp("Sub", [oneC, it], {}, bhMeta);
            } else {
                if (pf !== undefined) {
                    const [pfC] = innerBuilder.createOp(
                        "Mul",
                        [createCapturedInput(loopBody, pf), CIn],
                        {},
                        bhMeta,
                    );
                    [gf] = innerBuilder.createOp("Add", [gf, pfC], {}, bhMeta);
                }
                gf = applyClipIfNeeded(
                    innerBuilder,
                    gf,
                    prep.dtype,
                    bhShape,
                    prep.clipVal,
                    `lstml_cf_${d}_${op.id}`,
                );
                [ft] = innerBuilder.createOp(fAct, [gf], {}, bhMeta);
            }

            gc = applyClipIfNeeded(
                innerBuilder,
                gc,
                prep.dtype,
                bhShape,
                prep.clipVal,
                `lstml_cc_${d}_${op.id}`,
            );
            const [ctCand] = innerBuilder.createOp(gAct, [gc], {}, bhMeta);

            const [fC] = innerBuilder.createOp("Mul", [ft, CIn], {}, bhMeta);
            const [iC] = innerBuilder.createOp("Mul", [it, ctCand], {}, bhMeta);
            const [cCand] = innerBuilder.createOp("Add", [fC, iC], {}, bhMeta);

            if (po !== undefined) {
                const [poC] = innerBuilder.createOp(
                    "Mul",
                    [createCapturedInput(loopBody, po), cCand],
                    {},
                    bhMeta,
                );
                [go] = innerBuilder.createOp("Add", [go, poC], {}, bhMeta);
            }
            go = applyClipIfNeeded(
                innerBuilder,
                go,
                prep.dtype,
                bhShape,
                prep.clipVal,
                `lstml_co_${d}_${op.id}`,
            );
            const [ot] = innerBuilder.createOp(fAct, [go], {}, bhMeta);

            const [hC] = innerBuilder.createOp(hAct, [cCand], {}, bhMeta);
            const [hCand] = innerBuilder.createOp("Mul", [ot, hC], {}, bhMeta);

            let hNext: ConcreteValueNode = hCand;
            let cNext: ConcreteValueNode = cCand;
            let yStep: ConcreteValueNode = hCand;
            if (activeMask !== undefined) {
                const zeroC = innerBuilder.createConstant(
                    `lstml_z_${d}_${op.id}`,
                    makeTensorProto(prep.dtype, [], [0]),
                );
                [hNext] = innerBuilder.createOp("Where", [activeMask, hCand, HIn], {}, bhMeta);
                [cNext] = innerBuilder.createOp("Where", [activeMask, cCand, CIn], {}, bhMeta);
                [yStep] = innerBuilder.createOp("Where", [activeMask, hCand, zeroC], {}, bhMeta);
            }

            emitBodyOutput(loopBody, condIn, "cond_out");
            emitBodyOutput(loopBody, hNext, "h_out");
            emitBodyOutput(loopBody, cNext, "c_out");
            emitBodyOutput(loopBody, yStep, "y_scan_out");

            const { finalStates, yDir4D } = emitDirectionLoop(
                builder,
                prep,
                loopBody,
                [h0, c0],
                isReverse,
                `lstml_d${d}_${op.id}`,
            );
            dirYs.push(yDir4D);
            dirYhs.push(
                builder.createOp("Unsqueeze", [finalStates[0], prep.ax0], {}, [
                    { type: prep.dtype, shape: [1, prep.batchSize, H] as KnownShape },
                ])[0],
            );
            dirYcs.push(
                builder.createOp("Unsqueeze", [finalStates[1], prep.ax0], {}, [
                    { type: prep.dtype, shape: [1, prep.batchSize, H] as KnownShape },
                ])[0],
            );
        }

        finalizeRecurrentOutputs(op, builder, prep, dirYs, dirYhs, dirYcs);
    }
}
