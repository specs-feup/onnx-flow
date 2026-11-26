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
} from "../../../utils/index.js";

const isPresent = (n?: ValueNode): n is ValueNode => !!n && n.id !== "";

export class LowerGRURecipe implements DecompositionRecipe {
    public readonly name = "LowerGRU";
    public readonly targetOp = "GRU";
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
        if (op.type !== "GRU") return false;
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
            wShape[1] % 3 === 0
        );
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const W = ins[1];
        const R = ins[2];
        const B = isPresent(ins[3]) ? ins[3] : undefined;
        const initialH = isPresent(ins[5]) ? ins[5] : undefined;

        const outputs = op.getOutputs();
        const YOut = outputs[0];
        const YhOut = outputs[1];

        const dtype = X.literalType as DataType;
        const [seqLen, batchSize, inputSize] = toStaticShape(X.shape);
        const wShape = toStaticShape(W.shape);
        const numDirs = wShape[0];
        const hiddenSize = wShape[1] / 3;

        const direction = getStringAttr(op, "direction", "forward");
        const linearBeforeReset = getIntAttr(op, "linear_before_reset", 0);
        const rawActs = getAttr(op, "activations");
        const activations = Array.isArray(rawActs) ? (rawActs as string[]) : [];

        const ax0 = builder.createConstant(`gru_ax0_${op.id}`, int64Vec([0]));
        const ax1 = builder.createConstant(`gru_ax1_${op.id}`, int64Vec([1]));
        const step1 = builder.createConstant(`gru_step1_${op.id}`, int64Vec([1]));
        const oneConst = builder.createConstant(
            `gru_one_${op.id}`,
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
            const s = builder.createConstant(`gru_s_${tag}_${op.id}`, int64Vec([start]));
            const e = builder.createConstant(`gru_e_${tag}_${op.id}`, int64Vec([end]));
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
        const bhMeta = [{ type: dtype, shape: [batchSize, hiddenSize] as KnownShape }];

        for (let d = 0; d < numDirs; d++) {
            const isReverse = direction === "reverse" || (direction === "bidirectional" && d === 1);
            const fAct = activations[2 * d] ?? "Sigmoid";
            const gAct = activations[2 * d + 1] ?? "Tanh";

            // Extract W[d]^T [inputSize, 3H] and split into Wz, Wr, Wh [inputSize, H]
            const wd3 = slice1(W, ax0, d, d + 1, [1, 3 * hiddenSize, inputSize], `w_${d}`);
            const [wd] = builder.createOp("Squeeze", [wd3, ax0], {}, [
                { type: dtype, shape: [3 * hiddenSize, inputSize] },
            ]);
            const [wdT] = builder.createOp("Transpose", [wd], { perm: [1, 0] }, [
                { type: dtype, shape: [inputSize, 3 * hiddenSize] },
            ]);
            const WzT = slice1(wdT, ax1, 0, hiddenSize, [inputSize, hiddenSize], `wz_${d}`);
            const WrT = slice1(
                wdT,
                ax1,
                hiddenSize,
                2 * hiddenSize,
                [inputSize, hiddenSize],
                `wr_${d}`,
            );
            const WhT = slice1(
                wdT,
                ax1,
                2 * hiddenSize,
                3 * hiddenSize,
                [inputSize, hiddenSize],
                `wh_${d}`,
            );

            // Extract R[d]^T [hiddenSize, 3H] and split into Rz, Rr, Rh [hiddenSize, H]
            const rd3 = slice1(R, ax0, d, d + 1, [1, 3 * hiddenSize, hiddenSize], `r_${d}`);
            const [rd] = builder.createOp("Squeeze", [rd3, ax0], {}, [
                { type: dtype, shape: [3 * hiddenSize, hiddenSize] },
            ]);
            const [rdT] = builder.createOp("Transpose", [rd], { perm: [1, 0] }, [
                { type: dtype, shape: [hiddenSize, 3 * hiddenSize] },
            ]);
            const RzT = slice1(rdT, ax1, 0, hiddenSize, [hiddenSize, hiddenSize], `rz_${d}`);
            const RrT = slice1(
                rdT,
                ax1,
                hiddenSize,
                2 * hiddenSize,
                [hiddenSize, hiddenSize],
                `rr_${d}`,
            );
            const RhT = slice1(
                rdT,
                ax1,
                2 * hiddenSize,
                3 * hiddenSize,
                [hiddenSize, hiddenSize],
                `rh_${d}`,
            );

            // Optional biases: [Wbz, Wbr, Wbh, Rbz, Rbr, Rbh]
            let bz: ConcreteValueNode | undefined;
            let br: ConcreteValueNode | undefined;
            let wbh: ConcreteValueNode | undefined;
            let rbh: ConcreteValueNode | undefined;
            if (B) {
                const bd = slice1(B, ax0, d, d + 1, [1, 6 * hiddenSize], `b_${d}`);
                const wbz = slice1(bd, ax1, 0, hiddenSize, [1, hiddenSize], `wbz_${d}`);
                const wbr = slice1(
                    bd,
                    ax1,
                    hiddenSize,
                    2 * hiddenSize,
                    [1, hiddenSize],
                    `wbr_${d}`,
                );
                wbh = slice1(bd, ax1, 2 * hiddenSize, 3 * hiddenSize, [1, hiddenSize], `wbh_${d}`);
                const rbz = slice1(
                    bd,
                    ax1,
                    3 * hiddenSize,
                    4 * hiddenSize,
                    [1, hiddenSize],
                    `rbz_${d}`,
                );
                const rbr = slice1(
                    bd,
                    ax1,
                    4 * hiddenSize,
                    5 * hiddenSize,
                    [1, hiddenSize],
                    `rbr_${d}`,
                );
                rbh = slice1(bd, ax1, 5 * hiddenSize, 6 * hiddenSize, [1, hiddenSize], `rbh_${d}`);
                [bz] = builder.createOp("Add", [wbz, rbz], {}, [
                    { type: dtype, shape: [1, hiddenSize] },
                ]);
                [br] = builder.createOp("Add", [wbr, rbr], {}, [
                    { type: dtype, shape: [1, hiddenSize] },
                ]);
            }

            let H: ConcreteValueNode;
            if (initialH) {
                const h3 = slice1(initialH, ax0, d, d + 1, [1, batchSize, hiddenSize], `h0_${d}`);
                [H] = builder.createOp("Squeeze", [h3, ax0], {}, bhMeta);
            } else {
                H = builder.createConstant(
                    `gru_h0_zero_${d}_${op.id}`,
                    makeTensorProto(
                        dtype,
                        [batchSize, hiddenSize],
                        new Array<number>(batchSize * hiddenSize).fill(0),
                    ),
                );
            }

            const stepOutputs: ConcreteValueNode[] = new Array(seqLen);

            for (let step = 0; step < seqLen; step++) {
                const t = isReverse ? seqLen - 1 - step : step;
                const xt = xSteps[t];

                // z_t = f(Xt*Wz + H*Rz + bz)
                const [xwz] = builder.createOp("MatMul", [xt, WzT], {}, bhMeta);
                const [hrz] = builder.createOp("MatMul", [H, RzT], {}, bhMeta);
                let [zPre] = builder.createOp("Add", [xwz, hrz], {}, bhMeta);
                if (bz) [zPre] = builder.createOp("Add", [zPre, bz], {}, bhMeta);
                const [zt] = builder.createOp(fAct, [zPre], {}, bhMeta);

                // r_t = f(Xt*Wr + H*Rr + br)
                const [xwr] = builder.createOp("MatMul", [xt, WrT], {}, bhMeta);
                const [hrr] = builder.createOp("MatMul", [H, RrT], {}, bhMeta);
                let [rPre] = builder.createOp("Add", [xwr, hrr], {}, bhMeta);
                if (br) [rPre] = builder.createOp("Add", [rPre, br], {}, bhMeta);
                const [rt] = builder.createOp(fAct, [rPre], {}, bhMeta);

                // h_t
                let [xwh] = builder.createOp("MatMul", [xt, WhT], {}, bhMeta);
                if (wbh) [xwh] = builder.createOp("Add", [xwh, wbh], {}, bhMeta);

                let hRec: ConcreteValueNode;
                if (linearBeforeReset !== 0) {
                    let [hrh] = builder.createOp("MatMul", [H, RhT], {}, bhMeta);
                    if (rbh) [hrh] = builder.createOp("Add", [hrh, rbh], {}, bhMeta);
                    [hRec] = builder.createOp("Mul", [rt, hrh], {}, bhMeta);
                } else {
                    const [rH] = builder.createOp("Mul", [rt, H], {}, bhMeta);
                    [hRec] = builder.createOp("MatMul", [rH, RhT], {}, bhMeta);
                    if (rbh) [hRec] = builder.createOp("Add", [hRec, rbh], {}, bhMeta);
                }

                const [hPre] = builder.createOp("Add", [xwh, hRec], {}, bhMeta);
                const [htCand] = builder.createOp(gAct, [hPre], {}, bhMeta);

                // H_t = (1 - z_t) * h_t + z_t * H_{t-1}
                const [oneMinusZ] = builder.createOp("Sub", [oneConst, zt], {}, bhMeta);
                const [term1] = builder.createOp("Mul", [oneMinusZ, htCand], {}, bhMeta);
                const [term2] = builder.createOp("Mul", [zt, H], {}, bhMeta);
                [H] = builder.createOp("Add", [term1, term2], {}, bhMeta);

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

            const [yhDir] = builder.createOp("Unsqueeze", [H, ax0], {}, [
                { type: dtype, shape: [1, batchSize, hiddenSize] },
            ]);
            dirYhs.push(yhDir);
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

        builder.removeNode(op);
    }
}
