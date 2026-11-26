import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import {
    getFloatAttr,
    getIntAttr,
    int64Vec,
    makeTensorProto,
    scalarInt64,
} from "../../../utils/index.js";

/**
 * Canonicalizes Attention (Opset 23/24) supporting:
 * - 3D ([B, S, num_heads * head_size]) and 4D ([B, num_heads, S, head_size]) Q/K/V
 * - MHA, GQA, and MQA head counts
 * - Optional past_key/past_value concatenation -> present_key/present_value
 * - Optional scale (default 1/sqrt(head_dim)), softcap, boolean or float attn_mask, is_causal
 * - Optional qk_matmul_output at modes 0, 1, 2, 3
 */
export class LowerAttentionRecipe implements DecompositionRecipe {
    public readonly name = "LowerAttention";
    public readonly targetOp = "Attention";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly producedOps = [
        "Shape",
        "Gather",
        "Reshape",
        "Transpose",
        "Concat",
        "Unsqueeze",
        "Expand",
        "Mul",
        "MatMul",
        "Add",
        "Sub",
        "Range",
        "LessOrEqual",
        "Where",
        "Div",
        "Tanh",
        "Softmax",
        "Cast",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Attention") return false;
        const ins = op.getInputs();
        if (!ins || ins.length < 3) return false;
        const qRank = ins[0].shape.length;
        const kRank = ins[1].shape.length;
        const vRank = ins[2].shape.length;
        if (
            (qRank !== 3 && qRank !== 4) ||
            (kRank !== 3 && kRank !== 4) ||
            (vRank !== 3 && vRank !== 4)
        ) {
            return false;
        }
        if (qRank === 3 && getIntAttr(op, "q_num_heads", 0) <= 0) return false;
        if ((kRank === 3 || vRank === 3) && getIntAttr(op, "kv_num_heads", 0) <= 0) return false;
        return true;
    }

    private to4D(
        builder: GraphBuilder,
        tensor: ConcreteValueNode,
        numHeads: number,
        dtype: DataType,
        tag: string,
    ): ConcreteValueNode {
        if (tensor.shape.length === 4) return tensor;
        const b = tensor.shape[0] ?? -1;
        const s = tensor.shape[1] ?? -1;
        const hidden = tensor.shape[2];
        const d = typeof hidden === "number" && hidden > 0 ? Math.floor(hidden / numHeads) : -1;
        const r4Const = builder.createConstant(`${tag}_r4`, int64Vec([0, 0, numHeads, -1]));
        const [reshaped] = builder.createOp("Reshape", [tensor, r4Const], {}, [
            { type: dtype, shape: [b, s, numHeads, d] },
        ]);
        const [trans] = builder.createOp("Transpose", [reshaped], { perm: [0, 2, 1, 3] }, [
            { type: dtype, shape: [b, numHeads, s, d] },
        ]);
        return trans;
    }

    private repeatKvHeads(
        builder: GraphBuilder,
        kv: ConcreteValueNode,
        qHeads: number,
        kvHeads: number,
        dtype: DataType,
        tag: string,
    ): ConcreteValueNode {
        const rep = Math.floor(qHeads / kvHeads);
        const [kvShape] = builder.createOp("Shape", [kv], {}, [
            { type: DataType.INT64, shape: [4] },
        ]);
        const i0 = builder.createConstant(`${tag}_i0`, int64Vec([0]));
        const i2 = builder.createConstant(`${tag}_i2`, int64Vec([2]));
        const i3 = builder.createConstant(`${tag}_i3`, int64Vec([3]));
        const [bDim1D] = builder.createOp("Gather", [kvShape, i0], { axis: 0 }, [
            { type: DataType.INT64, shape: [1] },
        ]);
        const [sDim1D] = builder.createOp("Gather", [kvShape, i2], { axis: 0 }, [
            { type: DataType.INT64, shape: [1] },
        ]);
        const [dDim1D] = builder.createOp("Gather", [kvShape, i3], { axis: 0 }, [
            { type: DataType.INT64, shape: [1] },
        ]);

        const ax2 = builder.createConstant(`${tag}_ax2`, int64Vec([2]));
        const u5Shape: KnownShape = [
            kv.shape[0] ?? -1,
            kvHeads,
            1,
            kv.shape[2] ?? -1,
            kv.shape[3] ?? -1,
        ];
        const [kvU] = builder.createOp("Unsqueeze", [kv, ax2], {}, [
            { type: dtype, shape: u5Shape },
        ]);

        const exp5ShapeConst = builder.createConstant(`${tag}_exp`, int64Vec([1, 1, rep, 1, 1]));
        const exp5Shape: KnownShape = [
            kv.shape[0] ?? -1,
            kvHeads,
            rep,
            kv.shape[2] ?? -1,
            kv.shape[3] ?? -1,
        ];
        const [kvExp] = builder.createOp("Expand", [kvU, exp5ShapeConst], {}, [
            { type: dtype, shape: exp5Shape },
        ]);

        const qH1D = builder.createConstant(`${tag}_qh`, int64Vec([qHeads]));
        const [target4Shape] = builder.createOp(
            "Concat",
            [bDim1D, qH1D, sDim1D, dDim1D],
            { axis: 0 },
            [{ type: DataType.INT64, shape: [4] }],
        );
        const [out4] = builder.createOp("Reshape", [kvExp, target4Shape], {}, [
            {
                type: dtype,
                shape: [kv.shape[0] ?? -1, qHeads, kv.shape[2] ?? -1, kv.shape[3] ?? -1],
            },
        ]);
        return out4;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const Q = ins[0];
        const K = ins[1];
        const V = ins[2];
        const attnMask = ins.length >= 4 && ins[3].id !== "" ? ins[3] : undefined;
        const pastKey = ins.length >= 5 && ins[4].id !== "" ? ins[4] : undefined;
        const pastValue = ins.length >= 6 && ins[5].id !== "" ? ins[5] : undefined;
        const outputs = op.getOutputs();
        const Y = outputs[0];

        const dtype =
            (Q.literalType as DataType | undefined) ??
            (Y.literalType as DataType | undefined) ??
            DataType.FLOAT;
        const isCausal = getIntAttr(op, "is_causal", 0) === 1;
        const softcap = getFloatAttr(op, "softcap", 0.0);
        const qkOutMode = getIntAttr(op, "qk_matmul_output_mode", 0);
        const qWas3D = Q.shape.length === 3;

        const qHeadsAttr = getIntAttr(op, "q_num_heads", 0);
        const kvHeadsAttr = getIntAttr(op, "kv_num_heads", 0);
        const qHeads =
            qHeadsAttr > 0 ? qHeadsAttr : typeof Q.shape[1] === "number" ? Q.shape[1] : 0;
        const kvHeads =
            kvHeadsAttr > 0 ? kvHeadsAttr : typeof K.shape[1] === "number" ? K.shape[1] : qHeads;

        // 1. Normalize Q, K, V to 4D [B, H, S, D]
        const q4 = this.to4D(builder, Q, qHeads, dtype, `attn_q_${op.id}`);
        const k4 = this.to4D(builder, K, kvHeads, dtype, `attn_k_${op.id}`);
        const v4 = this.to4D(builder, V, kvHeads, dtype, `attn_v_${op.id}`);

        // 2. Append incoming K/V to optional past_key/past_value along sequence axis 2
        let presentK = k4;
        if (pastKey) {
            const pKSeq =
                typeof pastKey.shape[2] === "number" && typeof k4.shape[2] === "number"
                    ? pastKey.shape[2] + k4.shape[2]
                    : -1;
            [presentK] = builder.createOp("Concat", [pastKey, k4], { axis: 2 }, [
                {
                    type: dtype,
                    shape: [
                        k4.shape[0] ?? -1,
                        kvHeads || (k4.shape[1] ?? -1),
                        pKSeq,
                        k4.shape[3] ?? -1,
                    ],
                },
            ]);
        }

        let presentV = v4;
        if (pastValue) {
            const pVSeq =
                typeof pastValue.shape[2] === "number" && typeof v4.shape[2] === "number"
                    ? pastValue.shape[2] + v4.shape[2]
                    : -1;
            [presentV] = builder.createOp("Concat", [pastValue, v4], { axis: 2 }, [
                {
                    type: dtype,
                    shape: [
                        v4.shape[0] ?? -1,
                        kvHeads || (v4.shape[1] ?? -1),
                        pVSeq,
                        v4.shape[3] ?? -1,
                    ],
                },
            ]);
        }

        // 3. Expand KV heads for GQA/MQA if needed
        const needsKvRepeat =
            qHeads > 0 && kvHeads > 0 && qHeads > kvHeads && qHeads % kvHeads === 0;
        const kFull = needsKvRepeat
            ? this.repeatKvHeads(builder, presentK, qHeads, kvHeads, dtype, `attn_rk_${op.id}`)
            : presentK;
        const vFull = needsKvRepeat
            ? this.repeatKvHeads(builder, presentV, qHeads, kvHeads, dtype, `attn_rv_${op.id}`)
            : presentV;

        const batchDim = q4.shape[0] ?? -1;
        const qSeqDim = q4.shape[2] ?? -1;
        const kvSeqDim = kFull.shape[2] ?? -1;
        const headDimK = q4.shape[3] ?? -1;
        const headDimV = vFull.shape[3] ?? -1;
        const hEff = qHeads > 0 ? qHeads : (q4.shape[1] ?? -1);

        // 4. Scale Q and K by sqrt(scale) where default scale = 1 / sqrt(head_dim)
        const scaleAttr = op.attributes["scale"];
        let sqrtScaleNode: ConcreteValueNode;
        if (typeof scaleAttr === "number" && scaleAttr > 0) {
            sqrtScaleNode = builder.createConstant(
                `attn_ssc_${op.id}`,
                makeTensorProto(dtype, [], [Math.sqrt(scaleAttr)]),
            );
        } else if (typeof headDimK === "number" && headDimK > 0) {
            const scaleVal = 1.0 / Math.sqrt(headDimK);
            sqrtScaleNode = builder.createConstant(
                `attn_ssc_${op.id}`,
                makeTensorProto(dtype, [], [Math.sqrt(scaleVal)]),
            );
        } else {
            const [qShapeNode] = builder.createOp("Shape", [q4], {}, [
                { type: DataType.INT64, shape: [4] },
            ]);
            const idx3 = builder.createConstant(`attn_i3_${op.id}`, scalarInt64(3));
            const [dKNode] = builder.createOp("Gather", [qShapeNode, idx3], { axis: 0 }, [
                { type: DataType.INT64, shape: [] },
            ]);
            const [dKFloat] = builder.createOp("Cast", [dKNode], { to: dtype }, [
                { type: dtype, shape: [] },
            ]);
            const [sqrtD] = builder.createOp("Sqrt", [dKFloat], {}, [{ type: dtype, shape: [] }]);
            const oneC = builder.createConstant(
                `attn_one_${op.id}`,
                makeTensorProto(dtype, [], [1.0]),
            );
            const [scaleDyn] = builder.createOp("Div", [oneC, sqrtD], {}, [
                { type: dtype, shape: [] },
            ]);
            [sqrtScaleNode] = builder.createOp("Sqrt", [scaleDyn], {}, [
                { type: dtype, shape: [] },
            ]);
        }

        const [qScaled] = builder.createOp("Mul", [q4, sqrtScaleNode], {}, [
            { type: dtype, shape: [batchDim, hEff, qSeqDim, headDimK] },
        ]);
        const [kScaled] = builder.createOp("Mul", [kFull, sqrtScaleNode], {}, [
            { type: dtype, shape: [batchDim, hEff, kvSeqDim, headDimK] },
        ]);
        const [kTrans] = builder.createOp("Transpose", [kScaled], { perm: [0, 1, 3, 2] }, [
            { type: dtype, shape: [batchDim, hEff, headDimK, kvSeqDim] },
        ]);

        const scoresShape: KnownShape = [batchDim, hEff, qSeqDim, kvSeqDim];
        const expectedScores = [{ type: dtype, shape: scoresShape }];
        let [scores] = builder.createOp("MatMul", [qScaled, kTrans], {}, expectedScores);

        let qkCaptured: ConcreteValueNode | undefined = qkOutMode === 0 ? scores : undefined;

        // 5. Apply optional softcap before masking so masked (-Inf) positions are never compressed to -softcap
        if (softcap > 0) {
            const scConst = builder.createConstant(
                `attn_scap_${op.id}`,
                makeTensorProto(dtype, [], [softcap]),
            );
            const [divCap] = builder.createOp("Div", [scores, scConst], {}, expectedScores);
            const [tanhCap] = builder.createOp("Tanh", [divCap], {}, expectedScores);
            [scores] = builder.createOp("Mul", [tanhCap, scConst], {}, expectedScores);
        }

        if (qkOutMode === 2) qkCaptured = scores;

        // 6. Apply attn_mask and optional causal upper-triangular -Inf mask
        if (attnMask) {
            let maskAligned = attnMask;
            if (attnMask.shape.length === 3) {
                const ax1 = builder.createConstant(`attn_max1_${op.id}`, int64Vec([1]));
                [maskAligned] = builder.createOp("Unsqueeze", [attnMask, ax1], {}, [
                    {
                        type: attnMask.literalType as DataType,
                        shape: [
                            attnMask.shape[0] ?? -1,
                            1,
                            attnMask.shape[1] ?? -1,
                            attnMask.shape[2] ?? -1,
                        ],
                    },
                ]);
            }
            if (attnMask.literalType === DataType.BOOL) {
                const negInf = builder.createConstant(
                    `attn_ninf_m_${op.id}`,
                    makeTensorProto(dtype, [], [-Infinity]),
                );
                [scores] = builder.createOp(
                    "Where",
                    [maskAligned, scores, negInf],
                    {},
                    expectedScores,
                );
            } else {
                const maskFloat =
                    maskAligned.literalType === dtype
                        ? maskAligned
                        : builder.createOp("Cast", [maskAligned], { to: dtype }, [
                              { type: dtype, shape: maskAligned.shape as KnownShape },
                          ])[0];
                [scores] = builder.createOp("Add", [scores, maskFloat], {}, expectedScores);
            }
        }

        if (isCausal) {
            const [qS] = builder.createOp("Shape", [q4], {}, [
                { type: DataType.INT64, shape: [4] },
            ]);
            const [kS] = builder.createOp("Shape", [kFull], {}, [
                { type: DataType.INT64, shape: [4] },
            ]);
            const idx2 = builder.createConstant(`attn_c_i2_${op.id}`, scalarInt64(2));
            const [qSeqLen] = builder.createOp("Gather", [qS, idx2], { axis: 0 }, [
                { type: DataType.INT64, shape: [] },
            ]);
            const [kvSeqLen] = builder.createOp("Gather", [kS, idx2], { axis: 0 }, [
                { type: DataType.INT64, shape: [] },
            ]);
            const [pastLen] = builder.createOp("Sub", [kvSeqLen, qSeqLen], {}, [
                { type: DataType.INT64, shape: [] },
            ]);

            const z0 = builder.createConstant(`attn_cz0_${op.id}`, scalarInt64(0));
            const o1 = builder.createConstant(`attn_co1_${op.id}`, scalarInt64(1));
            const [rowRange] = builder.createOp("Range", [z0, qSeqLen, o1], {}, [
                { type: DataType.INT64, shape: [qSeqDim] },
            ]);
            const [colRange] = builder.createOp("Range", [z0, kvSeqLen, o1], {}, [
                { type: DataType.INT64, shape: [kvSeqDim] },
            ]);

            const rowColConst = builder.createConstant(`attn_rc_${op.id}`, int64Vec([-1, 1]));
            const colRowConst = builder.createConstant(`attn_cr_${op.id}`, int64Vec([1, -1]));
            const [row2D] = builder.createOp("Reshape", [rowRange, rowColConst], {}, [
                { type: DataType.INT64, shape: [qSeqDim, 1] },
            ]);
            const [col2D] = builder.createOp("Reshape", [colRange, colRowConst], {}, [
                { type: DataType.INT64, shape: [1, kvSeqDim] },
            ]);
            const [rowLimit] = builder.createOp("Add", [row2D, pastLen], {}, [
                { type: DataType.INT64, shape: [qSeqDim, 1] },
            ]);
            const [causalBool] = builder.createOp("LessOrEqual", [col2D, rowLimit], {}, [
                { type: DataType.BOOL, shape: [qSeqDim, kvSeqDim] },
            ]);

            const negInfC = builder.createConstant(
                `attn_ninf_c_${op.id}`,
                makeTensorProto(dtype, [], [-Infinity]),
            );
            [scores] = builder.createOp("Where", [causalBool, scores, negInfC], {}, expectedScores);
        }

        if (qkOutMode === 1) qkCaptured = scores;

        // 7. Softmax along last dim -> MatMul with V
        const [probs] = builder.createOp("Softmax", [scores], { axis: -1 }, expectedScores);
        if (qkOutMode === 3) qkCaptured = probs;

        const out4Shape: KnownShape = [batchDim, hEff, qSeqDim, headDimV];
        const [attn4] = builder.createOp("MatMul", [probs, vFull], {}, [
            { type: dtype, shape: out4Shape },
        ]);

        // 8. Reshape back to 3D if Q was 3D
        let finalY = attn4;
        if (qWas3D) {
            const [transBack] = builder.createOp("Transpose", [attn4], { perm: [0, 2, 1, 3] }, [
                { type: dtype, shape: [batchDim, qSeqDim, hEff, headDimV] },
            ]);
            const flat3Const = builder.createConstant(`attn_flat3_${op.id}`, int64Vec([0, 0, -1]));
            [finalY] = builder.createOp("Reshape", [transBack, flat3Const], {}, [
                { type: dtype, shape: Y.shape as KnownShape },
            ]);
        }

        builder.replaceAllUsesWith(Y, finalY);
        if (outputs.length > 1 && outputs[1].id !== "") {
            builder.replaceAllUsesWith(outputs[1], presentK);
        }
        if (outputs.length > 2 && outputs[2].id !== "") {
            builder.replaceAllUsesWith(outputs[2], presentV);
        }
        if (outputs.length > 3 && outputs[3].id !== "" && qkCaptured !== undefined) {
            builder.replaceAllUsesWith(outputs[3], qkCaptured);
        }

        op.remove();
    }
}
