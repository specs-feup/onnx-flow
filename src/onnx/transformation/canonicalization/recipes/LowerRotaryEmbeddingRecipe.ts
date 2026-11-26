import type OperationNode from "../../../ir/OperationNode.js";
import type { GraphBuilder } from "../../../ir/GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../ir/OnnxTypes.js";
import { DataType } from "../../../ir/OnnxTypes.js";
import { getIntAttr, int64Vec, scalarInt64 } from "../../../utils/index.js";

/**
 * Canonicalizes RotaryEmbedding (Opset 23) for both 3D ([B, S, num_heads * head_size])
 * and 4D ([B, num_heads, S, head_size]) layouts, full or partial rotary_embedding_dim,
 * and split-half (`interleaved=0`) or interleaved (`interleaved=1`) complex pairs.
 */
export class LowerRotaryEmbeddingRecipe implements DecompositionRecipe {
    public readonly name = "LowerRotaryEmbedding";
    public readonly targetOp = "RotaryEmbedding";
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
        "Slice",
        "Unsqueeze",
        "Mul",
        "Sub",
        "Add",
        "Concat",
        "Split",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "RotaryEmbedding") return false;
        const ins = op.getInputs();
        if (!ins || ins.length < 3) return false;
        const xRank = ins[0].shape.length;
        if (xRank !== 3 && xRank !== 4) return false;
        if (xRank === 3 && getIntAttr(op, "num_heads", 0) <= 0) return false;
        return true;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const cosCacheRaw = ins[1];
        const sinCacheRaw = ins[2];
        const positionIds = ins.length >= 4 && ins[3].id !== "" ? ins[3] : undefined;
        const Y = op.getOutputs()[0];

        const dtype =
            (X.literalType as DataType | undefined) ??
            (Y.literalType as DataType | undefined) ??
            DataType.FLOAT;
        const xRank = X.shape.length;
        const interleaved = getIntAttr(op, "interleaved", 0) !== 0;
        const numHeadsAttr = getIntAttr(op, "num_heads", 0);
        const rotaryDimAttr = getIntAttr(op, "rotary_embedding_dim", 0);

        const [origShapeX] = builder.createOp("Shape", [X], {}, [
            { type: DataType.INT64, shape: [xRank] },
        ]);

        // 1. Normalize input to 4D [B, S, H, D]
        let x4: ConcreteValueNode;
        let bDim: number | string;
        let sDim: number | string;
        let hDim: number | string;
        let dDim: number | string;

        if (xRank === 4) {
            bDim = X.shape[0] ?? -1;
            hDim = X.shape[1] ?? -1;
            sDim = X.shape[2] ?? -1;
            dDim = X.shape[3] ?? -1;
            [x4] = builder.createOp("Transpose", [X], { perm: [0, 2, 1, 3] }, [
                { type: dtype, shape: [bDim, sDim, hDim, dDim] },
            ]);
        } else {
            bDim = X.shape[0] ?? -1;
            sDim = X.shape[1] ?? -1;
            hDim = numHeadsAttr;
            const hidden = X.shape[2];
            dDim =
                typeof hidden === "number" && hidden > 0 ? Math.floor(hidden / numHeadsAttr) : -1;
            const target4D = builder.createConstant(
                `rope_r4_${op.id}`,
                int64Vec([0, 0, numHeadsAttr, -1]),
            );
            [x4] = builder.createOp("Reshape", [X, target4D], {}, [
                { type: dtype, shape: [bDim, sDim, hDim, dDim] },
            ]);
        }

        // Resolve half dimension cleanly from cache last dim or head_size
        const cosLastDim = cosCacheRaw.shape[cosCacheRaw.shape.length - 1];
        const cacheHalf = typeof cosLastDim === "number" && cosLastDim > 0 ? cosLastDim : -1;
        const headStatic = typeof dDim === "number" && dDim > 0 ? dDim : -1;
        const rotaryDim =
            rotaryDimAttr > 0
                ? rotaryDimAttr
                : headStatic > 0
                  ? headStatic
                  : cacheHalf > 0
                    ? cacheHalf * 2
                    : -1;
        const halfDim = rotaryDim > 0 ? Math.floor(rotaryDim / 2) : cacheHalf;

        const [shapeX4] = builder.createOp("Shape", [x4], {}, [
            { type: DataType.INT64, shape: [4] },
        ]);
        const idx3 = builder.createConstant(`rope_i3_${op.id}`, scalarInt64(3));
        const [dRuntime] = builder.createOp("Gather", [shapeX4, idx3], { axis: 0 }, [
            { type: DataType.INT64, shape: [] },
        ]);

        // 2. Partition partial rotary_embedding_dim if smaller than head_size
        const needsPartialSlice = rotaryDim > 0 && (headStatic < 0 || rotaryDim < headStatic);
        let xRotate = x4;
        let xNotRotate: ConcreteValueNode | undefined;

        if (needsPartialSlice) {
            const ax3 = builder.createConstant(`rope_ax3_${op.id}`, int64Vec([3]));
            const zeroVec = builder.createConstant(`rope_z1_${op.id}`, int64Vec([0]));
            const rotVec = builder.createConstant(`rope_rot_${op.id}`, int64Vec([rotaryDim]));
            const [dRuntime1D] = builder.createOp(
                "Unsqueeze",
                [dRuntime, builder.createConstant(`rope_u0_${op.id}`, int64Vec([0]))],
                {},
                [{ type: DataType.INT64, shape: [1] }],
            );

            [xRotate] = builder.createOp("Slice", [x4, zeroVec, rotVec, ax3], {}, [
                { type: dtype, shape: [bDim, sDim, hDim, rotaryDim] },
            ]);
            const remDim = headStatic > 0 ? headStatic - rotaryDim : -1;
            [xNotRotate] = builder.createOp("Slice", [x4, rotVec, dRuntime1D, ax3], {}, [
                { type: dtype, shape: [bDim, sDim, hDim, remDim] },
            ]);
        }

        // 3. Index cos_cache & sin_cache by position_ids if provided and expand head axis
        let cosVal = cosCacheRaw;
        let sinVal = sinCacheRaw;
        if (positionIds) {
            const posShape = positionIds.shape.map((d) => d ?? -1);
            const gatherShape: KnownShape = [...posShape, halfDim];
            [cosVal] = builder.createOp("Gather", [cosCacheRaw, positionIds], { axis: 0 }, [
                { type: dtype, shape: gatherShape },
            ]);
            [sinVal] = builder.createOp("Gather", [sinCacheRaw, positionIds], { axis: 0 }, [
                { type: dtype, shape: gatherShape },
            ]);
        }

        // cosVal / sinVal are either [B, S, half] (3D) or [S, half] (2D) -> insert H=1 axis
        const cacheRank = cosVal.shape.length;
        const headAxisPos = Math.max(0, cacheRank - 1);
        const headAxVec = builder.createConstant(`rope_hax_${op.id}`, int64Vec([headAxisPos]));
        const expandedCacheShape: KnownShape = [
            ...cosVal.shape.slice(0, headAxisPos).map((d) => d ?? -1),
            1,
            cosVal.shape[cacheRank - 1] ?? -1,
        ];
        const [cosU] = builder.createOp("Unsqueeze", [cosVal, headAxVec], {}, [
            { type: dtype, shape: expandedCacheShape },
        ]);
        const [sinU] = builder.createOp("Unsqueeze", [sinVal, headAxVec], {}, [
            { type: dtype, shape: expandedCacheShape },
        ]);

        // 4. Split rotated embedding into (x1, x2)
        const halfShape: KnownShape = [bDim, sDim, hDim, halfDim];
        let x1: ConcreteValueNode;
        let x2: ConcreteValueNode;

        if (interleaved) {
            const pairsShapeConst = builder.createConstant(
                `rope_pairs_${op.id}`,
                int64Vec([0, 0, 0, -1, 2]),
            );
            const [xPairs] = builder.createOp("Reshape", [xRotate, pairsShapeConst], {}, [
                { type: dtype, shape: [bDim, sDim, hDim, halfDim, 2] },
            ]);
            const idxZero = builder.createConstant(`rope_g0_${op.id}`, scalarInt64(0));
            const idxOne = builder.createConstant(`rope_g1_${op.id}`, scalarInt64(1));
            [x1] = builder.createOp("Gather", [xPairs, idxZero], { axis: 4 }, [
                { type: dtype, shape: halfShape },
            ]);
            [x2] = builder.createOp("Gather", [xPairs, idxOne], { axis: 4 }, [
                { type: dtype, shape: halfShape },
            ]);
        } else {
            [x1, x2] = builder.createOp("Split", [xRotate], { axis: 3, num_outputs: 2 }, [
                { type: dtype, shape: halfShape },
                { type: dtype, shape: halfShape },
            ]);
        }

        // 5. Apply complex rotation: real = cos*x1 - sin*x2, imag = sin*x1 + cos*x2
        const halfExpected = [{ type: dtype, shape: halfShape }];
        const [cx1] = builder.createOp("Mul", [cosU, x1], {}, halfExpected);
        const [sx2] = builder.createOp("Mul", [sinU, x2], {}, halfExpected);
        const [real] = builder.createOp("Sub", [cx1, sx2], {}, halfExpected);

        const [sx1] = builder.createOp("Mul", [sinU, x1], {}, halfExpected);
        const [cx2] = builder.createOp("Mul", [cosU, x2], {}, halfExpected);
        const [imag] = builder.createOp("Add", [sx1, cx2], {}, halfExpected);

        let xRotatedOut: ConcreteValueNode;
        const rotShape: KnownShape = [bDim, sDim, hDim, rotaryDim > 0 ? rotaryDim : dDim];

        if (interleaved) {
            const ax4Vec = builder.createConstant(`rope_ax4_${op.id}`, int64Vec([4]));
            const u5Shape: KnownShape = [bDim, sDim, hDim, halfDim, 1];
            const [realU] = builder.createOp("Unsqueeze", [real, ax4Vec], {}, [
                { type: dtype, shape: u5Shape },
            ]);
            const [imagU] = builder.createOp("Unsqueeze", [imag, ax4Vec], {}, [
                { type: dtype, shape: u5Shape },
            ]);
            const [pairsOut] = builder.createOp("Concat", [realU, imagU], { axis: 4 }, [
                { type: dtype, shape: [bDim, sDim, hDim, halfDim, 2] },
            ]);
            const [shapeXRot] = builder.createOp("Shape", [xRotate], {}, [
                { type: DataType.INT64, shape: [4] },
            ]);
            [xRotatedOut] = builder.createOp("Reshape", [pairsOut, shapeXRot], {}, [
                { type: dtype, shape: rotShape },
            ]);
        } else {
            [xRotatedOut] = builder.createOp("Concat", [real, imag], { axis: 3 }, [
                { type: dtype, shape: rotShape },
            ]);
        }

        let combined4D = xRotatedOut;
        if (xNotRotate) {
            [combined4D] = builder.createOp("Concat", [xRotatedOut, xNotRotate], { axis: 3 }, [
                { type: dtype, shape: [bDim, sDim, hDim, dDim] },
            ]);
        }

        // 6. Restore original 3D/4D layout
        let finalOut: ConcreteValueNode;
        if (xRank === 4) {
            [finalOut] = builder.createOp("Transpose", [combined4D], { perm: [0, 2, 1, 3] }, [
                { type: dtype, shape: Y.shape as KnownShape },
            ]);
        } else {
            [finalOut] = builder.createOp("Reshape", [combined4D, origShapeX], {}, [
                { type: dtype, shape: Y.shape as KnownShape },
            ]);
        }

        builder.replaceAllUsesWith(Y, finalOut);
        op.remove();
    }
}
