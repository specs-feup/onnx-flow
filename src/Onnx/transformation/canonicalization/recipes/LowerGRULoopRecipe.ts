import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { getAttr, getIntAttr, getStringAttr, makeTensorProto } from "../../../Utils.js";
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

export class LowerGRULoopRecipe implements DecompositionRecipe {
    public readonly name = "LowerGRULoop";
    public readonly targetOp = "GRU";
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
        return matchRecurrentLoopOp(op, "GRU", 3);
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const W = ins[1];
        const R = ins[2];
        const B = isPresent(ins[3]) ? ins[3] : undefined;
        const initialH = isPresent(ins[5]) ? ins[5] : undefined;

        const prep = prepareRecurrentInputs(op, builder, 3, "grul");
        const direction = getStringAttr(op, "direction", "forward");
        const linearBeforeReset = getIntAttr(op, "linear_before_reset", 0);
        const rawActs = getAttr(op, "activations");
        const activations = Array.isArray(rawActs) ? (rawActs as string[]) : [];
        const bhShape: KnownShape = [prep.batchSize, prep.hiddenSize];
        const bhMeta = [{ type: prep.dtype, shape: bhShape }];

        const dirYs: ConcreteValueNode[] = [];
        const dirYhs: ConcreteValueNode[] = [];

        for (let d = 0; d < prep.numDirs; d++) {
            const isReverse = direction === "reverse" || (direction === "bidirectional" && d === 1);
            const fAct = activations[2 * d] ?? "Sigmoid";
            const gAct = activations[2 * d + 1] ?? "Tanh";
            const H = prep.hiddenSize;

            const wd3 = sliceAxis(
                builder,
                W,
                prep.ax0,
                prep.step1,
                d,
                d + 1,
                [1, 3 * H, prep.inputSize],
                `grul_w_${d}_${op.id}`,
            );
            const [wd] = builder.createOp("Squeeze", [wd3, prep.ax0], {}, [
                { type: prep.dtype, shape: [3 * H, prep.inputSize] },
            ]);
            const [wdT] = builder.createOp("Transpose", [wd], { perm: [1, 0] }, [
                { type: prep.dtype, shape: [prep.inputSize, 3 * H] },
            ]);
            const WzT = sliceAxis(
                builder,
                wdT,
                prep.ax1,
                prep.step1,
                0,
                H,
                [prep.inputSize, H],
                `grul_wz_${d}_${op.id}`,
            );
            const WrT = sliceAxis(
                builder,
                wdT,
                prep.ax1,
                prep.step1,
                H,
                2 * H,
                [prep.inputSize, H],
                `grul_wr_${d}_${op.id}`,
            );
            const WhT = sliceAxis(
                builder,
                wdT,
                prep.ax1,
                prep.step1,
                2 * H,
                3 * H,
                [prep.inputSize, H],
                `grul_wh_${d}_${op.id}`,
            );

            const rd3 = sliceAxis(
                builder,
                R,
                prep.ax0,
                prep.step1,
                d,
                d + 1,
                [1, 3 * H, H],
                `grul_r_${d}_${op.id}`,
            );
            const [rd] = builder.createOp("Squeeze", [rd3, prep.ax0], {}, [
                { type: prep.dtype, shape: [3 * H, H] },
            ]);
            const [rdT] = builder.createOp("Transpose", [rd], { perm: [1, 0] }, [
                { type: prep.dtype, shape: [H, 3 * H] },
            ]);
            const RzT = sliceAxis(
                builder,
                rdT,
                prep.ax1,
                prep.step1,
                0,
                H,
                [H, H],
                `grul_rz_${d}_${op.id}`,
            );
            const RrT = sliceAxis(
                builder,
                rdT,
                prep.ax1,
                prep.step1,
                H,
                2 * H,
                [H, H],
                `grul_rr_${d}_${op.id}`,
            );
            const RhT = sliceAxis(
                builder,
                rdT,
                prep.ax1,
                prep.step1,
                2 * H,
                3 * H,
                [H, H],
                `grul_rh_${d}_${op.id}`,
            );

            let bz: ConcreteValueNode | undefined;
            let br: ConcreteValueNode | undefined;
            let wbh: ConcreteValueNode | undefined;
            let rbh: ConcreteValueNode | undefined;
            if (B !== undefined) {
                const bd = sliceAxis(
                    builder,
                    B,
                    prep.ax0,
                    prep.step1,
                    d,
                    d + 1,
                    [1, 6 * H],
                    `grul_b_${d}_${op.id}`,
                );
                const wbz = sliceAxis(
                    builder,
                    bd,
                    prep.ax1,
                    prep.step1,
                    0,
                    H,
                    [1, H],
                    `grul_wbz_${d}_${op.id}`,
                );
                const wbr = sliceAxis(
                    builder,
                    bd,
                    prep.ax1,
                    prep.step1,
                    H,
                    2 * H,
                    [1, H],
                    `grul_wbr_${d}_${op.id}`,
                );
                wbh = sliceAxis(
                    builder,
                    bd,
                    prep.ax1,
                    prep.step1,
                    2 * H,
                    3 * H,
                    [1, H],
                    `grul_wbh_${d}_${op.id}`,
                );
                const rbz = sliceAxis(
                    builder,
                    bd,
                    prep.ax1,
                    prep.step1,
                    3 * H,
                    4 * H,
                    [1, H],
                    `grul_rbz_${d}_${op.id}`,
                );
                const rbr = sliceAxis(
                    builder,
                    bd,
                    prep.ax1,
                    prep.step1,
                    4 * H,
                    5 * H,
                    [1, H],
                    `grul_rbr_${d}_${op.id}`,
                );
                rbh = sliceAxis(
                    builder,
                    bd,
                    prep.ax1,
                    prep.step1,
                    5 * H,
                    6 * H,
                    [1, H],
                    `grul_rbh_${d}_${op.id}`,
                );
                [bz] = builder.createOp("Add", [wbz, rbz], {}, [
                    { type: prep.dtype, shape: [1, H] },
                ]);
                [br] = builder.createOp("Add", [wbr, rbr], {}, [
                    { type: prep.dtype, shape: [1, H] },
                ]);
            }

            const h0 = buildInitialState(builder, prep, initialH, d, `grul_h0_${op.id}`);

            const { loopBody, innerBuilder, iterNode, condIn, stateIns } = createLoopBodyShell(
                prep,
                1,
                `grul_body_${d}_${op.id}`,
            );
            const HIn = stateIns[0];
            const { xt, activeMask } = buildTimestepGather(
                loopBody,
                innerBuilder,
                prep,
                iterNode,
                isReverse,
                `grul_t_${d}_${op.id}`,
            );

            const oneC = innerBuilder.createConstant(
                `grul_one_${d}_${op.id}`,
                makeTensorProto(prep.dtype, [], [1]),
            );

            // z_t = f(Xt*Wz + H*Rz + bz)
            const [xwz] = innerBuilder.createOp(
                "MatMul",
                [xt, createCapturedInput(loopBody, WzT)],
                {},
                bhMeta,
            );
            const [hrz] = innerBuilder.createOp(
                "MatMul",
                [HIn, createCapturedInput(loopBody, RzT)],
                {},
                bhMeta,
            );
            let [zPre] = innerBuilder.createOp("Add", [xwz, hrz], {}, bhMeta);
            if (bz !== undefined) {
                [zPre] = innerBuilder.createOp(
                    "Add",
                    [zPre, createCapturedInput(loopBody, bz)],
                    {},
                    bhMeta,
                );
            }
            zPre = applyClipIfNeeded(
                innerBuilder,
                zPre,
                prep.dtype,
                bhShape,
                prep.clipVal,
                `grul_cz_${d}_${op.id}`,
            );
            const [zt] = innerBuilder.createOp(fAct, [zPre], {}, bhMeta);

            // r_t = f(Xt*Wr + H*Rr + br)
            const [xwr] = innerBuilder.createOp(
                "MatMul",
                [xt, createCapturedInput(loopBody, WrT)],
                {},
                bhMeta,
            );
            const [hrr] = innerBuilder.createOp(
                "MatMul",
                [HIn, createCapturedInput(loopBody, RrT)],
                {},
                bhMeta,
            );
            let [rPre] = innerBuilder.createOp("Add", [xwr, hrr], {}, bhMeta);
            if (br !== undefined) {
                [rPre] = innerBuilder.createOp(
                    "Add",
                    [rPre, createCapturedInput(loopBody, br)],
                    {},
                    bhMeta,
                );
            }
            rPre = applyClipIfNeeded(
                innerBuilder,
                rPre,
                prep.dtype,
                bhShape,
                prep.clipVal,
                `grul_cr_${d}_${op.id}`,
            );
            const [rt] = innerBuilder.createOp(fAct, [rPre], {}, bhMeta);

            // h_t
            let [xwh] = innerBuilder.createOp(
                "MatMul",
                [xt, createCapturedInput(loopBody, WhT)],
                {},
                bhMeta,
            );
            if (wbh !== undefined) {
                [xwh] = innerBuilder.createOp(
                    "Add",
                    [xwh, createCapturedInput(loopBody, wbh)],
                    {},
                    bhMeta,
                );
            }

            let hRec: ConcreteValueNode;
            if (linearBeforeReset !== 0) {
                let [hrh] = innerBuilder.createOp(
                    "MatMul",
                    [HIn, createCapturedInput(loopBody, RhT)],
                    {},
                    bhMeta,
                );
                if (rbh !== undefined) {
                    [hrh] = innerBuilder.createOp(
                        "Add",
                        [hrh, createCapturedInput(loopBody, rbh)],
                        {},
                        bhMeta,
                    );
                }
                [hRec] = innerBuilder.createOp("Mul", [rt, hrh], {}, bhMeta);
            } else {
                const [rH] = innerBuilder.createOp("Mul", [rt, HIn], {}, bhMeta);
                [hRec] = innerBuilder.createOp(
                    "MatMul",
                    [rH, createCapturedInput(loopBody, RhT)],
                    {},
                    bhMeta,
                );
                if (rbh !== undefined) {
                    [hRec] = innerBuilder.createOp(
                        "Add",
                        [hRec, createCapturedInput(loopBody, rbh)],
                        {},
                        bhMeta,
                    );
                }
            }

            let [hPre] = innerBuilder.createOp("Add", [xwh, hRec], {}, bhMeta);
            hPre = applyClipIfNeeded(
                innerBuilder,
                hPre,
                prep.dtype,
                bhShape,
                prep.clipVal,
                `grul_ch_${d}_${op.id}`,
            );
            const [htCand] = innerBuilder.createOp(gAct, [hPre], {}, bhMeta);

            // H_t = (1 - z_t) * h_t + z_t * H_{t-1}
            const [oneMinusZ] = innerBuilder.createOp("Sub", [oneC, zt], {}, bhMeta);
            const [term1] = innerBuilder.createOp("Mul", [oneMinusZ, htCand], {}, bhMeta);
            const [term2] = innerBuilder.createOp("Mul", [zt, HIn], {}, bhMeta);
            const [hCand] = innerBuilder.createOp("Add", [term1, term2], {}, bhMeta);

            let hNext: ConcreteValueNode = hCand;
            let yStep: ConcreteValueNode = hCand;
            if (activeMask !== undefined) {
                const zeroC = innerBuilder.createConstant(
                    `grul_z_${d}_${op.id}`,
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
                `grul_d${d}_${op.id}`,
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
