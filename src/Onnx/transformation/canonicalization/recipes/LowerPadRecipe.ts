import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import {
    decodeIntegerVectorFromTensorProto,
    getStringAttr,
    makeTensorProto,
    readScalarFromTensorNode,
    toStaticShape,
} from "../../../Utils.js";
import ConstantNode from "../../../ConstantNode.js";

export class LowerPadRecipe implements DecompositionRecipe {
    public readonly name = "LowerPad";
    public readonly targetOp = "Pad";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = true;
    public readonly producedOps = ["Slice", "Concat", "Gather", "Expand", "Range"];

    canApply(op: OperationNode.Class): boolean {
        if (op.type !== "Pad") return false;
        const ins = op.getInputs() ?? [];
        if (ins.length < 2 || !ins[1]?.is(ConstantNode)) return false;

        // We can only safely canonicalize static pad dimensions
        const inShape = toStaticShape(ins[0].shape);
        return inShape.length > 0 && inShape.every((d) => d > 0);
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const Xin = ins[0];
        const padsNode = ins[1] as ConstantNode.Class;
        const Y = op.getOutputs()[0];

        const rank = Xin.shape.length;

        // Safely parse pads as Numbers to avoid BigInt Math/JSON crashes
        const rawPads = (decodeIntegerVectorFromTensorProto(padsNode.constantValue) ?? []).map(
            Number,
        );
        const mode = getStringAttr(op, "mode", "constant").toLowerCase();

        // Correctly map pads to the proper axes if the `axes` input is present
        let axes = Array.from({ length: rank }, (_, i) => i);
        if (ins.length > 3 && ins[3]?.is(ConstantNode)) {
            const decodedAxes = decodeIntegerVectorFromTensorProto(
                (ins[3] as ConstantNode.Class).constantValue,
            );
            if (decodedAxes) axes = decodedAxes.map(Number);
        }

        const numAxes = axes.length;
        const beg = new Array(rank).fill(0);
        const end = new Array(rank).fill(0);
        for (let i = 0; i < numAxes; i++) {
            const ax = axes[i];
            const realAx = ax < 0 ? ax + rank : ax; // Handle negative axes
            beg[realAx] = rawPads[i];
            end[realAx] = rawPads[i + numAxes];
        }

        const dtype = Xin.literalType as DataType;

        // Keep valNode as a TensorNode so it supports dynamic graphs and all dtypes natively
        let valNode: ConcreteValueNode;
        if (ins.length > 2 && ins[2]) {
            valNode = ins[2];
        } else {
            valNode = builder.createConstant(
                `pad_val_default_${op.id}`,
                makeTensorProto(dtype, [], [0]),
            );
        }

        let cur = Xin;
        const curShape = toStaticShape(Xin.shape);

        // 1. Negative pads (Crop)
        for (let ax = 0; ax < rank; ax++) {
            const negB = Math.max(0, -beg[ax]);
            const negE = Math.max(0, -end[ax]);
            if (negB === 0 && negE === 0) continue;

            const start1 = builder.createConstant(
                `pad_crop_s_${op.id}_${ax}`,
                makeTensorProto(DataType.INT64, [1], [negB]),
            );
            const end1 = builder.createConstant(
                `pad_crop_e_${op.id}_${ax}`,
                makeTensorProto(DataType.INT64, [1], [curShape[ax] - negE]),
            );
            const axVec = builder.createConstant(
                `pad_crop_ax_${op.id}_${ax}`,
                makeTensorProto(DataType.INT64, [1], [ax]),
            );

            curShape[ax] -= negB + negE;
            const expectedSliceOut = [{ type: dtype, shape: [...curShape] }];
            cur = builder.createOp("Slice", [cur, start1, end1, axVec], {}, expectedSliceOut)[0];

            beg[ax] = Math.max(0, beg[ax]);
            end[ax] = Math.max(0, end[ax]);
        }

        // 2. Positive pads
        for (let ax = 0; ax < rank; ax++) {
            const pBeg = beg[ax];
            const pEnd = end[ax];
            if (pBeg === 0 && pEnd === 0) continue;

            let left: ConcreteValueNode | undefined;
            let right: ConcreteValueNode | undefined;

            if (mode === "constant") {
                if (pBeg > 0)
                    left = this.ensurePadSlabConst(
                        builder,
                        curShape,
                        ax,
                        pBeg,
                        dtype,
                        valNode, 
                        `${op.id}_${ax}_L`,
                    );
                if (pEnd > 0)
                    right = this.ensurePadSlabConst(
                        builder,
                        curShape,
                        ax,
                        pEnd,
                        dtype,
                        valNode, 
                        `${op.id}_${ax}_R`,
                    );
            } else if (mode === "edge") {
                if (pBeg > 0)
                    left = this.ensureEdgeSlab(
                        builder,
                        curShape,
                        cur,
                        ax,
                        pBeg,
                        `${op.id}_${ax}_L`,
                    );
                if (pEnd > 0)
                    right = this.ensureEdgeSlab(
                        builder,
                        curShape,
                        cur,
                        ax,
                        pEnd,
                        `${op.id}_${ax}_R`,
                    );
            } else {
                if (pBeg > 0)
                    left = this.ensureReflectSlab(
                        builder,
                        curShape,
                        cur,
                        ax,
                        pBeg,
                        `${op.id}_${ax}_L`,
                    );
                if (pEnd > 0)
                    right = this.ensureReflectSlab(
                        builder,
                        curShape,
                        cur,
                        ax,
                        pEnd,
                        `${op.id}_${ax}_R`,
                    );
            }

            const parts: ConcreteValueNode[] = [];
            if (left) parts.push(left);
            parts.push(cur);
            if (right) parts.push(right);

            curShape[ax] += pBeg + pEnd;
            const expectedConcatOut = [{ type: dtype, shape: [...curShape] }];
            cur = builder.createOp("Concat", parts, { axis: ax }, expectedConcatOut)[0];
        }

        if (cur === Xin) cur = builder.createOp("Identity", [Xin], {}, [{type: Xin.literalType, shape: Xin.shape as KnownShape}] )[0];
        builder.replaceAllUsesWith(Y, cur);
        op.remove();
    }

    private ensurePadSlabConst(
        builder: GraphBuilder,
        curShape: number[],
        axis: number,
        size: number,
        dtype: DataType,
        valNode: ConcreteValueNode,
        tag: string,
    ) {
        const slabShape = [...curShape];
        slabShape[axis] = size;
        const newShape = builder.createConstant(
            `pad_sh_${tag}`,
            makeTensorProto(DataType.INT64, [slabShape.length], slabShape),
        );

        // Statically typed!
        const expectedOut = [{ type: dtype, shape: slabShape }];
        return builder.createOp("Expand", [valNode, newShape], {}, expectedOut)[0];
    }

    private ensureEdgeSlab(
        builder: GraphBuilder,
        curShape: number[],
        cur: ConcreteValueNode,
        axis: number,
        size: number,
        tag: string,
    ) {
        const axVec = builder.createConstant(
            `edge_ax_${tag}`,
            makeTensorProto(DataType.INT64, [1], [axis]),
        );
        const zero1 = builder.createConstant(
            `edge_0_${tag}`,
            makeTensorProto(DataType.INT64, [1], [0]),
        );
        const one1 = builder.createConstant(
            `edge_1_${tag}`,
            makeTensorProto(DataType.INT64, [1], [1]),
        );

        let starts: ConcreteValueNode, ends: ConcreteValueNode;
        if (tag.endsWith("L")) {
            starts = zero1;
            ends = one1;
        } else {
            starts = builder.createConstant(
                `edge_s_${tag}`,
                makeTensorProto(DataType.INT64, [1], [curShape[axis] - 1]),
            );
            ends = builder.createConstant(
                `edge_e_${tag}`,
                makeTensorProto(DataType.INT64, [1], [curShape[axis]]),
            );
        }

        const expectedSlice = [
            {
                type: cur.literalType as DataType,
                shape: [...curShape].map((d, i) => (i === axis ? 1 : d)),
            },
        ];
        const oneSlice = builder.createOp(
            "Slice",
            [cur, starts, ends, axVec],
            {},
            expectedSlice,
        )[0];

        const slabShape = [...curShape];
        slabShape[axis] = size;
        const newShape = builder.createConstant(
            `edge_sh_${tag}`,
            makeTensorProto(DataType.INT64, [slabShape.length], slabShape),
        );

        const expectedOut = [{ type: cur.literalType as DataType, shape: slabShape }];
        return builder.createOp("Expand", [oneSlice, newShape], {}, expectedOut)[0];
    }

    private ensureReflectSlab(
        builder: GraphBuilder,
        curShape: number[],
        cur: ConcreteValueNode,
        axis: number,
        size: number,
        tag: string,
    ) {
        const dim = curShape[axis];
        const sizeClamped = Math.min(size, dim - 1);

        let startSc: number, endSc: number;
        if (tag.endsWith("L")) {
            startSc = sizeClamped;
            endSc = 0;
        } else {
            startSc = dim - 2;
            endSc = startSc - sizeClamped;
        }

        const startC = builder.createConstant(
            `refl_s_${tag}`,
            makeTensorProto(DataType.INT64, [], [startSc]),
        );
        const endC = builder.createConstant(
            `refl_e_${tag}`,
            makeTensorProto(DataType.INT64, [], [endSc]),
        );
        const stepC = builder.createConstant(
            `refl_step_${tag}`,
            makeTensorProto(DataType.INT64, [], [-1]),
        );

        const expectedGather = [
            {
                type: cur.literalType as DataType,
                shape: [...curShape].map((d, i) => (i === axis ? sizeClamped : d)),
            },
        ];
        const idx = builder.createOp("Range", [startC, endC, stepC], {}, [
            { type: DataType.INT64, shape: [sizeClamped] },
        ])[0];
        return builder.createOp("Gather", [cur, idx], { axis }, expectedGather)[0];
    }
}
