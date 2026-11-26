import type OperationNode from "../../../ir/OperationNode.js";
import type { GraphBuilder } from "../../../ir/GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape, ValueNode } from "../../../ir/OnnxTypes.js";
import type { DataType } from "../../../ir/OnnxTypes.js";
import {
    getAttr,
    getIntAttr,
    getStringAttr,
    int64Vec,
    makeTensorProto,
    toStaticShape,
} from "../../../utils/index.js";

const isPresent = (n?: ValueNode): n is ValueNode => !!n && n.id !== "";

export class LowerRNNRecipe implements DecompositionRecipe {
    public readonly name = "LowerRNN";
    public readonly targetOp = "RNN";
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
        "Concat",
        "Tanh",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "RNN") return false;
        if (getAttr(op, "clip") !== undefined) return false;
        if (getAttr(op, "activation_alpha") !== undefined) return false;
        if (getAttr(op, "activation_beta") !== undefined) return false;

        const ins = op.getInputs() as ConcreteValueNode[] | undefined;
        if (!ins || ins.length < 3) return false;
        if (isPresent(ins[4])) return false; // sequence_lens not supported in static unroll
        if (getIntAttr(op, "layout", 0) !== 0) return false;

        const xShape = toStaticShape(ins[0].shape);
        const wShape = toStaticShape(ins[1].shape);
        return (
            xShape.length === 3 &&
            wShape.length === 3 &&
            xShape[0] > 0 &&
            xShape[1] > 0 &&
            wShape[1] > 0
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
        const hiddenSize = wShape[1];

        const direction = getStringAttr(op, "direction", "forward");
        const rawActs = getAttr(op, "activations");
        const activations = Array.isArray(rawActs) ? (rawActs as string[]) : [];

        const ax0 = builder.createConstant(`rnn_ax0_${op.id}`, int64Vec([0]));
        const ax1 = builder.createConstant(`rnn_ax1_${op.id}`, int64Vec([1]));
        const step1 = builder.createConstant(`rnn_step1_${op.id}`, int64Vec([1]));

        const slice1 = (
            src: ConcreteValueNode,
            axisConst: ConcreteValueNode,
            start: number,
            end: number,
            shape: number[],
            tag: string,
        ) => {
            const s = builder.createConstant(`rnn_s_${tag}_${op.id}`, int64Vec([start]));
            const e = builder.createConstant(`rnn_e_${tag}_${op.id}`, int64Vec([end]));
            return builder.createOp("Slice", [src, s, e, axisConst, step1], {}, [
                { type: dtype, shape: shape as KnownShape },
            ])[0];
        };

        // Pre-slice X into per-timestep [batchSize, inputSize]
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

        for (let d = 0; d < numDirs; d++) {
            const isReverse = direction === "reverse" || (direction === "bidirectional" && d === 1);
            const actOp = activations[d] ?? "Tanh";

            // Extract W[d]^T: [inputSize, hiddenSize] and R[d]^T: [hiddenSize, hiddenSize]
            const wd3 = slice1(W, ax0, d, d + 1, [1, hiddenSize, inputSize], `w_${d}`);
            const [wd] = builder.createOp("Squeeze", [wd3, ax0], {}, [
                { type: dtype, shape: [hiddenSize, inputSize] },
            ]);
            const [wdT] = builder.createOp("Transpose", [wd], { perm: [1, 0] }, [
                { type: dtype, shape: [inputSize, hiddenSize] },
            ]);

            const rd3 = slice1(R, ax0, d, d + 1, [1, hiddenSize, hiddenSize], `r_${d}`);
            const [rd] = builder.createOp("Squeeze", [rd3, ax0], {}, [
                { type: dtype, shape: [hiddenSize, hiddenSize] },
            ]);
            const [rdT] = builder.createOp("Transpose", [rd], { perm: [1, 0] }, [
                { type: dtype, shape: [hiddenSize, hiddenSize] },
            ]);

            // Optional combined bias: Wb[d] + Rb[d] of shape [1, hiddenSize]
            let biasD: ConcreteValueNode | undefined;
            if (B) {
                const wb = slice1(B, ax0, d, d + 1, [1, 2 * hiddenSize], `b_${d}`);
                const wbPart = slice1(wb, ax1, 0, hiddenSize, [1, hiddenSize], `wb_${d}`);
                const rbPart = slice1(
                    wb,
                    ax1,
                    hiddenSize,
                    2 * hiddenSize,
                    [1, hiddenSize],
                    `rb_${d}`,
                );
                [biasD] = builder.createOp("Add", [wbPart, rbPart], {}, [
                    { type: dtype, shape: [1, hiddenSize] },
                ]);
            }

            // Initial hidden state H: [batchSize, hiddenSize]
            let H: ConcreteValueNode;
            if (initialH) {
                const h3 = slice1(initialH, ax0, d, d + 1, [1, batchSize, hiddenSize], `h0_${d}`);
                [H] = builder.createOp("Squeeze", [h3, ax0], {}, [
                    { type: dtype, shape: [batchSize, hiddenSize] },
                ]);
            } else {
                H = builder.createConstant(
                    `rnn_h0_zero_${d}_${op.id}`,
                    makeTensorProto(
                        dtype,
                        [batchSize, hiddenSize],
                        new Array<number>(batchSize * hiddenSize).fill(0),
                    ),
                );
            }

            const stepOutputs: ConcreteValueNode[] = new Array(seqLen);
            const bhMeta = [{ type: dtype, shape: [batchSize, hiddenSize] as KnownShape }];

            for (let step = 0; step < seqLen; step++) {
                const t = isReverse ? seqLen - 1 - step : step;
                const [xw] = builder.createOp("MatMul", [xSteps[t], wdT], {}, bhMeta);
                const [hr] = builder.createOp("MatMul", [H, rdT], {}, bhMeta);
                let [preAct] = builder.createOp("Add", [xw, hr], {}, bhMeta);
                if (biasD) {
                    [preAct] = builder.createOp("Add", [preAct, biasD], {}, bhMeta);
                }
                [H] = builder.createOp(actOp, [preAct], {}, bhMeta);

                const [hUnsq] = builder.createOp("Unsqueeze", [H, ax0], {}, [
                    { type: dtype, shape: [1, batchSize, hiddenSize] },
                ]);
                stepOutputs[t] = hUnsq;
            }

            // Stack direction Y: [seqLen, 1, batchSize, hiddenSize]
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
