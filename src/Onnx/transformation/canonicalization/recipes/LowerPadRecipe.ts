import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode } from "../../../OnnxTypes.js";
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
        const pads = decodeIntegerVectorFromTensorProto(padsNode.constantValue) ?? [];
        const mode = getStringAttr(op, "mode", "constant").toLowerCase();

        let padValue = 0;
        if (ins.length > 2) {
            const rawPadValue = ins[2];
            const s = readScalarFromTensorNode(rawPadValue);
            if (typeof s === "number" && Number.isFinite(s)) padValue = s;
        }

        const dtype = Xin.literalType as DataType;
        let cur = Xin;
        const curShape = toStaticShape(Xin.shape);

        const beg = pads.slice(0, rank);
        const end = pads.slice(rank);

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

            cur = builder.createOp("Slice", [cur, start1, end1, axVec])[0];
            curShape[ax] -= negB + negE;

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
                        padValue,
                        `${op.id}_${ax}_L`,
                    );
                if (pEnd > 0)
                    right = this.ensurePadSlabConst(
                        builder,
                        curShape,
                        ax,
                        pEnd,
                        dtype,
                        padValue,
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

            cur = builder.createOp("Concat", parts, { axis: ax })[0];
            curShape[ax] += pBeg + pEnd;
        }

        if (cur === Xin) cur = builder.createOp("Identity", [Xin])[0];
        builder.replaceAllUsesWith(Y, cur);
        op.remove();
    }

    private ensurePadSlabConst(
        builder: GraphBuilder,
        curShape: number[],
        axis: number,
        size: number,
        dtype: DataType,
        padValue: number,
        tag: string,
    ) {
        const slabShape = [...curShape];
        slabShape[axis] = size;
        const newShape = builder.createConstant(
            `pad_sh_${tag}`,
            makeTensorProto(DataType.INT64, [slabShape.length], slabShape),
        );
        const kT = builder.createConstant(`pad_val_${tag}`, makeTensorProto(dtype, [], [padValue]));

        // Statically typed!
        const expectedOut = [{ type: dtype, shape: slabShape }];
        return builder.createOp("Expand", [kT, newShape], {}, expectedOut)[0];
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
