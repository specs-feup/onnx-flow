import type OperationNode from "../../../OperationNode.js";
import { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { getIntAttr, getStringAttr, makeTensorProto, toStaticShape } from "../../../Utils.js";
import OnnxGraph from "../../../OnnxGraph.js";
import TensorNode from "../../../TensorNode.js";
import Graph from "@specs-feup/flow/graph/Graph";

export class LowerLSTMRecipe implements DecompositionRecipe {
    public readonly name = "LowerLSTM";
    public readonly targetOp = "LSTM";
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = false;
    public readonly producedOps = [
        "Scan",
        "Squeeze",
        "Unsqueeze",
        "MatMul",
        "Add",
        "Split",
        "Sigmoid",
        "Tanh",
        "Mul",
        "ConstantOfShape",
        "Shape",
        "Identity",
        "Cast",
        "Gather",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "LSTM") return false;
        if (getStringAttr(op, "direction", "forward") !== "forward") return false;
        return true;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const outs = op.getOutputs();

        const X = ins[0];
        const W = ins[1];
        const R = ins[2];

        let B: ConcreteValueNode | undefined;
        let sequence_lens: ConcreteValueNode | undefined;
        let initial_h: ConcreteValueNode | undefined;
        let initial_c: ConcreteValueNode | undefined;

        for (let i = 3; i < ins.length; i++) {
            if (!(i in ins)) continue;
            const input = ins[i];
            const rank = input.shape.length;

            if (rank === 2 && !B) B = input;
            else if (rank === 1 && !sequence_lens) sequence_lens = input;
            else if (rank === 3) {
                if (!initial_h) initial_h = input;
                else if (!initial_c) initial_c = input;
            }
        }

        const hidden_size = getIntAttr(op, "hidden_size", -1);
        const dtype = (X.literalType as DataType | undefined) ?? DataType.FLOAT;
        const zeroAxis = builder.createConstant(
            `lstm_zero_axis_${op.id}`,
            makeTensorProto(DataType.INT64, [1], [0]),
        );

        // OVERRIDE BUG IN INFERSHAPES SQUEEZE BY PASSING EXPLICIT SHAPES
        const w_hidden = hidden_size * 4;
        const w_input = toStaticShape(W.shape)[2];

        const W_sq = builder.createOp("Squeeze", [W, zeroAxis], {}, [
            { type: dtype, shape: [w_hidden, w_input] },
        ])[0];
        const R_sq = builder.createOp("Squeeze", [R, zeroAxis], {}, [
            { type: dtype, shape: [w_hidden, hidden_size] },
        ])[0];

        const W_T = builder.createOp("Transpose", [W_sq], { perm: [1, 0] }, [
            { type: dtype, shape: [w_input, w_hidden] },
        ])[0];
        const R_T = builder.createOp("Transpose", [R_sq], { perm: [1, 0] }, [
            { type: dtype, shape: [hidden_size, w_hidden] },
        ])[0];

        let B_combined: ConcreteValueNode | undefined;
        if (B) {
            const B_sq = builder.createOp("Squeeze", [B, zeroAxis], {}, [
                { type: dtype, shape: [w_hidden * 2] },
            ])[0];
            const splitB = builder.createOp("Split", [B_sq], { axis: 0, num_outputs: 2 }, [
                { type: dtype, shape: [w_hidden] },
                { type: dtype, shape: [w_hidden] },
            ]);
            B_combined = builder.createOp("Add", [splitB[0], splitB[1]], {}, [
                { type: dtype, shape: [w_hidden] },
            ])[0];
        }

        const shapeX = builder.createOp("Shape", [X])[0];
        const batchIdxConst = builder.createConstant(
            `lstm_b_idx_${op.id}`,
            makeTensorProto(DataType.INT64, [1], [1]),
        );
        const batchSize = builder.createOp("Gather", [shapeX, batchIdxConst], { axis: 0 })[0];

        let seqLens = sequence_lens;
        if (!seqLens) {
            const seqLenIdxConst = builder.createConstant(
                `lstm_s_idx_${op.id}`,
                makeTensorProto(DataType.INT64, [1], [0]),
            );
            const seqLen = builder.createOp("Gather", [shapeX, seqLenIdxConst], { axis: 0 })[0];

            let zeros = builder.createOp("ConstantOfShape", [batchSize], {}, [
                { type: DataType.FLOAT, shape: [-1] },
            ])[0];
            zeros = builder.createOp("Cast", [zeros], { to: DataType.INT64 }, [
                { type: DataType.INT64, shape: [-1] },
            ])[0];
            seqLens = builder.createOp("Add", [seqLen, zeros], {}, [
                { type: DataType.INT64, shape: [-1] },
            ])[0];
        }

        if (!initial_h || !initial_c) {
            const hiddenSizeConst = builder.createConstant(
                `lstm_hidden_size_${op.id}`,
                makeTensorProto(DataType.INT64, [1], [hidden_size]),
            );
            const stateShape = builder.createOp("Concat", [batchSize, hiddenSizeConst], {
                axis: 0,
            })[0];
            const zeros = builder.createOp(
                "ConstantOfShape",
                [stateShape],
                {
                    value: makeTensorProto(dtype, [1], [0]),
                },
                [{ type: dtype, shape: [-1, hidden_size] }],
            )[0];

            if (!initial_h) initial_h = zeros;
            else
                initial_h = builder.createOp("Squeeze", [initial_h, zeroAxis], {}, [
                    { type: dtype, shape: [-1, hidden_size] },
                ])[0];

            if (!initial_c) initial_c = zeros;
            else
                initial_c = builder.createOp("Squeeze", [initial_c, zeroAxis], {}, [
                    { type: dtype, shape: [-1, hidden_size] },
                ])[0];
        } else {
            initial_h = builder.createOp("Squeeze", [initial_h, zeroAxis], {}, [
                { type: dtype, shape: [-1, hidden_size] },
            ])[0];
            initial_c = builder.createOp("Squeeze", [initial_c, zeroAxis], {}, [
                { type: dtype, shape: [-1, hidden_size] },
            ])[0];
        }

        const bodyGraph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const innerBuilder = new GraphBuilder(bodyGraph, `cell_${op.id}`);

        const H_prev = bodyGraph
            .addNode("H_prev")
            .init(new TensorNode.Builder(dtype, [-1, hidden_size], "input"))
            .as(TensorNode);
        const C_prev = bodyGraph
            .addNode("C_prev")
            .init(new TensorNode.Builder(dtype, [-1, hidden_size], "input"))
            .as(TensorNode);
        const X_t = bodyGraph
            .addNode("X_t")
            .init(new TensorNode.Builder(dtype, [-1, w_input], "input"))
            .as(TensorNode);

        const W_T_inner = innerBuilder.createOp("Identity", [W_T])[0];
        const R_T_inner = innerBuilder.createOp("Identity", [R_T])[0];

        const X_gates = innerBuilder.createOp("MatMul", [X_t, W_T_inner])[0];
        const H_gates = innerBuilder.createOp("MatMul", [H_prev, R_T_inner])[0];

        let Gates = innerBuilder.createOp("Add", [X_gates, H_gates])[0];
        if (B_combined) {
            const B_inner = innerBuilder.createOp("Identity", [B_combined])[0];
            Gates = innerBuilder.createOp("Add", [Gates, B_inner])[0];
        }

        const splitGates = innerBuilder.createOp("Split", [Gates], { axis: -1, num_outputs: 4 }, [
            { type: dtype, shape: [-1, hidden_size] },
            { type: dtype, shape: [-1, hidden_size] },
            { type: dtype, shape: [-1, hidden_size] },
            { type: dtype, shape: [-1, hidden_size] },
        ]);

        const i_gate = innerBuilder.createOp("Sigmoid", [splitGates[0]])[0];
        const o_gate = innerBuilder.createOp("Sigmoid", [splitGates[1]])[0];
        const f_gate = innerBuilder.createOp("Sigmoid", [splitGates[2]])[0];
        const c_cand = innerBuilder.createOp("Tanh", [splitGates[3]])[0];

        const f_C = innerBuilder.createOp("Mul", [f_gate, C_prev])[0];
        const i_c = innerBuilder.createOp("Mul", [i_gate, c_cand])[0];
        const C_t_math = innerBuilder.createOp("Add", [f_C, i_c])[0];

        const tanh_Ct = innerBuilder.createOp("Tanh", [C_t_math])[0];
        const H_t_math = innerBuilder.createOp("Mul", [o_gate, tanh_Ct])[0];

        const H_t = innerBuilder.createOp("Identity", [H_t_math], {}, [
            { type: dtype, shape: [-1, hidden_size] },
        ])[0] as TensorNode.Class;
        H_t.setType("output");

        const C_t = innerBuilder.createOp("Identity", [C_t_math], {}, [
            { type: dtype, shape: [-1, hidden_size] },
        ])[0] as TensorNode.Class;
        C_t.setType("output");

        const Y_t = innerBuilder.createOp("Identity", [H_t_math], {}, [
            { type: dtype, shape: [-1, hidden_size] },
        ])[0] as TensorNode.Class;
        Y_t.setType("output");

        const scanOuts = builder.createOp(
            "Scan",
            [seqLens, initial_h, initial_c, X],
            { num_scan_inputs: 1 },
            [
                { type: dtype, shape: [-1, hidden_size] },
                { type: dtype, shape: [-1, hidden_size] },
                { type: dtype, shape: [-1, -1, hidden_size] },
            ],
            [bodyGraph],
        );

        const Y_final = builder.createOp(
            "Unsqueeze",
            [
                scanOuts[2],
                builder.createConstant(
                    `unsq_Y_${op.id}`,
                    makeTensorProto(DataType.INT64, [1], [1]),
                ),
            ],
            {},
            [{ type: dtype, shape: [-1, 1, -1, hidden_size] }],
        )[0];
        const Y_h_final = builder.createOp("Unsqueeze", [scanOuts[0], zeroAxis], {}, [
            { type: dtype, shape: [1, -1, hidden_size] },
        ])[0];
        const Y_c_final = builder.createOp("Unsqueeze", [scanOuts[1], zeroAxis], {}, [
            { type: dtype, shape: [1, -1, hidden_size] },
        ])[0];

        if (0 in outs) builder.replaceAllUsesWith(outs[0], Y_final);
        if (outs.length > 1) builder.replaceAllUsesWith(outs[1], Y_h_final);
        if (outs.length > 2) builder.replaceAllUsesWith(outs[2], Y_c_final);

        builder.removeNode(op);
    }
}
