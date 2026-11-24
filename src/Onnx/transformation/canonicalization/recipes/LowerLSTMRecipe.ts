import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape, ValueNode } from "../../../OnnxTypes.js";
import type { DataType } from "../../../OnnxTypes.js";
import {
    getAttr,
    getIntAttr,
    getStringAttr,
    int64Vec,
    makeTensorProto,
    toStaticShape,
} from "../../../Utils.js";

const isPresent = (n?: ValueNode): n is ValueNode => !!n && n.id !== "";

export class LowerLSTMRecipe implements DecompositionRecipe {
    public readonly name = "LowerLSTM";
    public readonly targetOp = "LSTM";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 100;
    public readonly shapeSupport = "static" as const;
    public readonly tags = ["unrolled"];
    public readonly producedOps = [
        "Slice",
        "Squeeze",
        "Unsqueeze",
        "Transpose",
        "MatMul",
        "Add",
        "Sub",
        "Mul",
        "Concat",
        "Sigmoid",
        "Tanh",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "LSTM") return false;
        if (getAttr(op, "clip") !== undefined) return false;
        if (getAttr(op, "activation_alpha") !== undefined) return false;
        if (getAttr(op, "activation_beta") !== undefined) return false;

        const ins = op.getInputs() as ConcreteValueNode[] | undefined;
        if (!ins || ins.length < 3) return false;
        if (isPresent(ins[4])) return false; // sequence_lens
        if (getIntAttr(op, "layout", 0) !== 0) return false;

        const xShape = toStaticShape(ins[0].shape);
        const wShape = toStaticShape(ins[1].shape);
        return (
            xShape.length === 3 &&
            wShape.length === 3 &&
            xShape[0] > 0 &&
            xShape[1] > 0 &&
            wShape[1] > 0 &&
            wShape[1] % 4 === 0
        );
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const W = ins[1];
        const R = ins[2];
        const B = isPresent(ins[3]) ? ins[3] : undefined;
        const initialH = isPresent(ins[5]) ? ins[5] : undefined;
        const initialC = isPresent(ins[6]) ? ins[6] : undefined;
        const P = isPresent(ins[7]) ? ins[7] : undefined;

        const outputs = op.getOutputs();
        const YOut = outputs[0];
        const YhOut = outputs[1];
        const YcOut = outputs[2];

        const dtype = X.literalType as DataType;
        const [seqLen, batchSize, inputSize] = toStaticShape(X.shape);
        const wShape = toStaticShape(W.shape);
        const numDirs = wShape[0];
        const hiddenSize = wShape[1] / 4;

        const direction = getStringAttr(op, "direction", "forward");
        const inputForget = getIntAttr(op, "input_forget", 0);
        const rawActs = getAttr(op, "activations");
        const activations = Array.isArray(rawActs) ? (rawActs as string[]) : [];

        const ax0 = builder.createConstant(`lstm_ax0_${op.id}`, int64Vec([0]));
        const ax1 = builder.createConstant(`lstm_ax1_${op.id}`, int64Vec([1]));
        const step1 = builder.createConstant(`lstm_step1_${op.id}`, int64Vec([1]));
        const oneConst = builder.createConstant(
            `lstm_one_${op.id}`,
            makeTensorProto(dtype, [], [1]),
        );

        const slice1 = (
            src: ConcreteValueNode,
            axisConst: ConcreteValueNode,
            start: number,
            end: number,
            shape: number[],
            tag: string,
        ) => {
            const s = builder.createConstant(`lstm_s_${tag}_${op.id}`, int64Vec([start]));
            const e = builder.createConstant(`lstm_e_${tag}_${op.id}`, int64Vec([end]));
            return builder.createOp("Slice", [src, s, e, axisConst, step1], {}, [
                { type: dtype, shape: shape as KnownShape },
            ])[0];
        };

        const xSteps: ConcreteValueNode[] = [];
        for (let t = 0; t < seqLen; t++) {
            const xSlice = slice1(X, ax0, t, t + 1, [1, batchSize, inputSize], `x_${t}`);
            const [xSq] = builder.createOp("Squeeze", [xSlice, ax0], {}, [
                { type: dtype, shape: [batchSize, inputSize] },
            ]);
            xSteps.push(xSq);
        }

        const dirYs: ConcreteValueNode[] = [];
        const dirYhs: ConcreteValueNode[] = [];
        const dirYcs: ConcreteValueNode[] = [];
        const bhMeta = [{ type: dtype, shape: [batchSize, hiddenSize] as KnownShape }];
        const b4hMeta = [{ type: dtype, shape: [batchSize, 4 * hiddenSize] as KnownShape }];

        for (let d = 0; d < numDirs; d++) {
            const isReverse = direction === "reverse" || (direction === "bidirectional" && d === 1);
            const fAct = activations[3 * d] ?? "Sigmoid";
            const gAct = activations[3 * d + 1] ?? "Tanh";
            const hAct = activations[3 * d + 2] ?? "Tanh";

            const wd3 = slice1(W, ax0, d, d + 1, [1, 4 * hiddenSize, inputSize], `w_${d}`);
            const [wd] = builder.createOp("Squeeze", [wd3, ax0], {}, [
                { type: dtype, shape: [4 * hiddenSize, inputSize] },
            ]);
            const [wdT] = builder.createOp("Transpose", [wd], { perm: [1, 0] }, [
                { type: dtype, shape: [inputSize, 4 * hiddenSize] },
            ]);

            const rd3 = slice1(R, ax0, d, d + 1, [1, 4 * hiddenSize, hiddenSize], `r_${d}`);
            const [rd] = builder.createOp("Squeeze", [rd3, ax0], {}, [
                { type: dtype, shape: [4 * hiddenSize, hiddenSize] },
            ]);
            const [rdT] = builder.createOp("Transpose", [rd], { perm: [1, 0] }, [
                { type: dtype, shape: [hiddenSize, 4 * hiddenSize] },
            ]);

            let biasD: ConcreteValueNode | undefined;
            if (B) {
                const bd = slice1(B, ax0, d, d + 1, [1, 8 * hiddenSize], `b_${d}`);
                const wb = slice1(bd, ax1, 0, 4 * hiddenSize, [1, 4 * hiddenSize], `wb_${d}`);
                const rb = slice1(
                    bd,
                    ax1,
                    4 * hiddenSize,
                    8 * hiddenSize,
                    [1, 4 * hiddenSize],
                    `rb_${d}`,
                );
                [biasD] = builder.createOp("Add", [wb, rb], {}, [
                    { type: dtype, shape: [1, 4 * hiddenSize] },
                ]);
            }

            let pi: ConcreteValueNode | undefined;
            let po: ConcreteValueNode | undefined;
            let pf: ConcreteValueNode | undefined;
            if (P) {
                const pd = slice1(P, ax0, d, d + 1, [1, 3 * hiddenSize], `p_${d}`);
                pi = slice1(pd, ax1, 0, hiddenSize, [1, hiddenSize], `pi_${d}`);
                po = slice1(pd, ax1, hiddenSize, 2 * hiddenSize, [1, hiddenSize], `po_${d}`);
                pf = slice1(pd, ax1, 2 * hiddenSize, 3 * hiddenSize, [1, hiddenSize], `pf_${d}`);
            }

            const makeZeroState = (tag: string) =>
                builder.createConstant(
                    `lstm_${tag}_zero_${d}_${op.id}`,
                    makeTensorProto(
                        dtype,
                        [batchSize, hiddenSize],
                        new Array<number>(batchSize * hiddenSize).fill(0),
                    ),
                );

            let H: ConcreteValueNode = initialH
                ? builder.createOp(
                      "Squeeze",
                      [slice1(initialH, ax0, d, d + 1, [1, batchSize, hiddenSize], `h0_${d}`), ax0],
                      {},
                      bhMeta,
                  )[0]
                : makeZeroState("h0");

            let C: ConcreteValueNode = initialC
                ? builder.createOp(
                      "Squeeze",
                      [slice1(initialC, ax0, d, d + 1, [1, batchSize, hiddenSize], `c0_${d}`), ax0],
                      {},
                      bhMeta,
                  )[0]
                : makeZeroState("c0");

            const stepOutputs: ConcreteValueNode[] = new Array(seqLen);

            for (let step = 0; step < seqLen; step++) {
                const t = isReverse ? seqLen - 1 - step : step;

                // Combined gates [batchSize, 4*hiddenSize] in ONNX order: [i, o, f, c]
                const [xw] = builder.createOp("MatMul", [xSteps[t], wdT], {}, b4hMeta);
                const [hr] = builder.createOp("MatMul", [H, rdT], {}, b4hMeta);
                let [gates] = builder.createOp("Add", [xw, hr], {}, b4hMeta);
                if (biasD) [gates] = builder.createOp("Add", [gates, biasD], {}, b4hMeta);

                let gi = slice1(gates, ax1, 0, hiddenSize, [batchSize, hiddenSize], `gi_${d}_${t}`);
                let go = slice1(
                    gates,
                    ax1,
                    hiddenSize,
                    2 * hiddenSize,
                    [batchSize, hiddenSize],
                    `go_${d}_${t}`,
                );
                let gf = slice1(
                    gates,
                    ax1,
                    2 * hiddenSize,
                    3 * hiddenSize,
                    [batchSize, hiddenSize],
                    `gf_${d}_${t}`,
                );
                const gc = slice1(
                    gates,
                    ax1,
                    3 * hiddenSize,
                    4 * hiddenSize,
                    [batchSize, hiddenSize],
                    `gc_${d}_${t}`,
                );

                if (pi) {
                    const [piC] = builder.createOp("Mul", [pi, C], {}, bhMeta);
                    [gi] = builder.createOp("Add", [gi, piC], {}, bhMeta);
                }
                const [it] = builder.createOp(fAct, [gi], {}, bhMeta);

                let ft: ConcreteValueNode;
                if (inputForget !== 0) {
                    [ft] = builder.createOp("Sub", [oneConst, it], {}, bhMeta);
                } else {
                    if (pf) {
                        const [pfC] = builder.createOp("Mul", [pf, C], {}, bhMeta);
                        [gf] = builder.createOp("Add", [gf, pfC], {}, bhMeta);
                    }
                    [ft] = builder.createOp(fAct, [gf], {}, bhMeta);
                }

                const [ctCand] = builder.createOp(gAct, [gc], {}, bhMeta);

                // C_t = f_t * C_{t-1} + i_t * c_t
                const [fC] = builder.createOp("Mul", [ft, C], {}, bhMeta);
                const [iC] = builder.createOp("Mul", [it, ctCand], {}, bhMeta);
                [C] = builder.createOp("Add", [fC, iC], {}, bhMeta);

                if (po) {
                    const [poC] = builder.createOp("Mul", [po, C], {}, bhMeta);
                    [go] = builder.createOp("Add", [go, poC], {}, bhMeta);
                }
                const [ot] = builder.createOp(fAct, [go], {}, bhMeta);

                const [hC] = builder.createOp(hAct, [C], {}, bhMeta);
                [H] = builder.createOp("Mul", [ot, hC], {}, bhMeta);

                const [hUnsq] = builder.createOp("Unsqueeze", [H, ax0], {}, [
                    { type: dtype, shape: [1, batchSize, hiddenSize] },
                ]);
                stepOutputs[t] = hUnsq;
            }

            const yDirSeq =
                seqLen === 1
                    ? stepOutputs[0]
                    : builder.createOp("Concat", stepOutputs, { axis: 0 }, [
                          { type: dtype, shape: [seqLen, batchSize, hiddenSize] },
                      ])[0];
            const [yDir4D] = builder.createOp("Unsqueeze", [yDirSeq, ax1], {}, [
                { type: dtype, shape: [seqLen, 1, batchSize, hiddenSize] },
            ]);
            dirYs.push(yDir4D);

            dirYhs.push(
                builder.createOp("Unsqueeze", [H, ax0], {}, [
                    { type: dtype, shape: [1, batchSize, hiddenSize] },
                ])[0],
            );
            dirYcs.push(
                builder.createOp("Unsqueeze", [C, ax0], {}, [
                    { type: dtype, shape: [1, batchSize, hiddenSize] },
                ])[0],
            );
        }

        if (isPresent(YOut)) {
            const finalY =
                numDirs === 1
                    ? dirYs[0]
                    : builder.createOp("Concat", dirYs, { axis: 1 }, [
                          { type: dtype, shape: [seqLen, numDirs, batchSize, hiddenSize] },
                      ])[0];
            builder.replaceAllUsesWith(YOut, finalY);
        }
        if (isPresent(YhOut)) {
            const finalYh =
                numDirs === 1
                    ? dirYhs[0]
                    : builder.createOp("Concat", dirYhs, { axis: 0 }, [
                          { type: dtype, shape: [numDirs, batchSize, hiddenSize] },
                      ])[0];
            builder.replaceAllUsesWith(YhOut, finalYh);
        }
        if (isPresent(YcOut)) {
            const finalYc =
                numDirs === 1
                    ? dirYcs[0]
                    : builder.createOp("Concat", dirYcs, { axis: 0 }, [
                          { type: dtype, shape: [numDirs, batchSize, hiddenSize] },
                      ])[0];
            builder.replaceAllUsesWith(YcOut, finalYc);
        }

        builder.removeNode(op);
    }
}
