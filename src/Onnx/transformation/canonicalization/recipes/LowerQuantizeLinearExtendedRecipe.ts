import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import { DataType, type ConcreteValueNode, type KnownShape } from "../../../OnnxTypes.js";
import {
    getIntAttr,
    int64Vec,
    makeTensorProto,
    normalizeAxis,
    toStaticShape,
} from "../../../Utils.js";

const QUANT_BOUNDS: Partial<Record<DataType, [number, number]>> = {
    [DataType.UINT8]: [0, 255],
    [DataType.INT8]: [-128, 127],
    [DataType.UINT16]: [0, 65535],
    [DataType.INT16]: [-32768, 32767],
    [DataType.UINT4]: [0, 15],
    [DataType.INT4]: [-8, 7],
    [DataType.UINT2]: [0, 3],
    [DataType.INT2]: [-2, 1],
};

export class LowerQuantizeLinearExtendedRecipe implements DecompositionRecipe {
    public readonly name = "LowerQuantizeLinearExtended";
    public readonly targetOp = "QuantizeLinear";
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 10;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["quantization", "opset21", "blocked", "int16"] as const;
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = [
        "Shape",
        "Gather",
        "Concat",
        "Reshape",
        "Unsqueeze",
        "Cast",
        "Div",
        "Round",
        "Add",
        "Clip",
    ];

    private resolveTargetDtype(op: OperationNode.Class): DataType {
        const ins = op.getInputs()!;
        const zp = ins.length >= 3 && ins[2].id !== "" ? ins[2] : undefined;
        if (zp?.literalType !== undefined) return zp.literalType as DataType;
        const attrDtype = getIntAttr(op, "output_dtype", 0);
        if (attrDtype !== 0) return attrDtype as DataType;
        const outDtype = op.getOutputs()[0]?.literalType;
        return outDtype as DataType;
    }

    match(op: OperationNode.Class): boolean {
        if (op.type !== "QuantizeLinear") return false;
        const ins = op.getInputs();
        if (!ins || ins.length < 2) return false;

        const targetDtype = this.resolveTargetDtype(op);
        if (QUANT_BOUNDS[targetDtype] === undefined) return false;

        const blockSize = getIntAttr(op, "block_size", 0);
        const X = ins[0];
        const scale = ins[1];
        if (blockSize > 0) {
            if (!Array.isArray(X.shape) || X.shape.length === 0) return false;
            if (!Array.isArray(scale.shape) || scale.shape.length !== X.shape.length) return false;
            const dims = toStaticShape(X.shape);
            const axis = normalizeAxis(getIntAttr(op, "axis", 1), dims.length);
            if (dims[axis] > 0 && dims[axis] % blockSize !== 0) return false;
            return true;
        }

        // Per-tensor / per-axis extended dtypes (INT16 / UINT16 / INT4 / UINT4 / INT2 / UINT2)
        return (
            targetDtype === DataType.INT16 ||
            targetDtype === DataType.UINT16 ||
            targetDtype === DataType.INT4 ||
            targetDtype === DataType.UINT4 ||
            targetDtype === DataType.INT2 ||
            targetDtype === DataType.UINT2
        );
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs()! as ConcreteValueNode[];
        const X = ins[0];
        const scale = ins[1];
        const zp = ins.length >= 3 && ins[2].id !== "" ? ins[2] : undefined;
        const Y = op.getOutputs()[0];

        const targetDtype = this.resolveTargetDtype(op);
        const [qMin, qMax] = QUANT_BOUNDS[targetDtype]!;
        const calcDtype = (scale.literalType || DataType.FLOAT) as DataType;
        const outShape = Y.shape as KnownShape;
        const rank = X.shape.length;
        const axis = rank > 0 ? normalizeAxis(getIntAttr(op, "axis", 1), rank) : 0;
        const blockSize = getIntAttr(op, "block_size", 0);

        let xCalc: ConcreteValueNode =
            X.literalType === calcDtype
                ? X
                : builder.createOp("Cast", [X], { to: calcDtype }, [
                      { type: calcDtype, shape: X.shape as KnownShape },
                  ])[0];

        let scaleCalc: ConcreteValueNode = scale;
        let zpCalc: ConcreteValueNode | undefined = zp
            ? builder.createOp("Cast", [zp], { to: calcDtype }, [
                  { type: calcDtype, shape: zp.shape as KnownShape },
              ])[0]
            : undefined;

        let shapeXNode: ConcreteValueNode | undefined;
        let workShape: KnownShape = outShape;

        if (blockSize > 0) {
            // Reshape X: [..., numBlocks * blockSize, ...] -> [..., numBlocks, blockSize, ...]
            [shapeXNode] = builder.createOp("Shape", [X], {}, [
                { type: DataType.INT64, shape: [rank] },
            ]);
            const xStatic = toStaticShape(X.shape);
            const blockedStatic = [
                ...xStatic.slice(0, axis),
                xStatic[axis] > 0 ? Math.floor(xStatic[axis] / blockSize) : -1,
                blockSize,
                ...xStatic.slice(axis + 1),
            ] as KnownShape;
            workShape = blockedStatic;

            let blockedShapeNode: ConcreteValueNode;
            if (!blockedStatic.includes(-1)) {
                blockedShapeNode = builder.createConstant(
                    `qle_bshape_${op.id}`,
                    int64Vec(blockedStatic as number[]),
                );
            } else {
                const [shapeScaleNode] = builder.createOp("Shape", [scale], {}, [
                    { type: DataType.INT64, shape: [rank] },
                ]);

                const parts: ConcreteValueNode[] = [];
                if (axis > 0) {
                    const leftIdx = builder.createConstant(
                        `qle_l_${op.id}`,
                        int64Vec(Array.from({ length: axis }, (_, i) => i)),
                    );
                    parts.push(
                        builder.createOp("Gather", [shapeXNode, leftIdx], { axis: 0 }, [
                            { type: DataType.INT64, shape: [axis] },
                        ])[0],
                    );
                }
                const axIdx = builder.createConstant(`qle_ax_${op.id}`, int64Vec([axis]));
                const [numBlocks1D] = builder.createOp(
                    "Gather",
                    [shapeScaleNode, axIdx],
                    { axis: 0 },
                    [{ type: DataType.INT64, shape: [1] }],
                );
                parts.push(numBlocks1D);
                parts.push(builder.createConstant(`qle_bs_${op.id}`, int64Vec([blockSize])));
                if (axis < rank - 1) {
                    const rightIdx = builder.createConstant(
                        `qle_r_${op.id}`,
                        int64Vec(Array.from({ length: rank - 1 - axis }, (_, i) => axis + 1 + i)),
                    );
                    parts.push(
                        builder.createOp("Gather", [shapeXNode, rightIdx], { axis: 0 }, [
                            { type: DataType.INT64, shape: [rank - 1 - axis] },
                        ])[0],
                    );
                }

                [blockedShapeNode] = builder.createOp("Concat", parts, { axis: 0 }, [
                    { type: DataType.INT64, shape: [rank + 1] },
                ]);
            }

            [xCalc] = builder.createOp("Reshape", [xCalc, blockedShapeNode], {}, [
                { type: calcDtype, shape: workShape },
            ]);

            const unsqAxis = builder.createConstant(`qle_ua_${op.id}`, int64Vec([axis + 1]));
            [scaleCalc] = builder.createOp("Unsqueeze", [scaleCalc, unsqAxis]);
            if (zpCalc) {
                [zpCalc] = builder.createOp("Unsqueeze", [zpCalc, unsqAxis]);
            }
        } else if (scale.shape.length === 1 && rank > 1) {
            // Per-axis broadcast: unsqueeze all axes != axis
            const unsqAxes: number[] = [];
            for (let d = 0; d < rank; d++) {
                if (d !== axis) unsqAxes.push(d);
            }
            const unsqConst = builder.createConstant(`qle_pa_${op.id}`, int64Vec(unsqAxes));
            [scaleCalc] = builder.createOp("Unsqueeze", [scaleCalc, unsqConst]);
            if (zpCalc && zp!.shape.length === 1) {
                [zpCalc] = builder.createOp("Unsqueeze", [zpCalc, unsqConst]);
            }
        }

        const [scaled] = builder.createOp("Div", [xCalc, scaleCalc], {}, [
            { type: calcDtype, shape: workShape },
        ]);
        const [rounded] = builder.createOp("Round", [scaled], {}, [
            { type: calcDtype, shape: workShape },
        ]);
        const shifted = zpCalc
            ? builder.createOp("Add", [rounded, zpCalc], {}, [
                  { type: calcDtype, shape: workShape },
              ])[0]
            : rounded;

        const minC = builder.createConstant(
            `qle_min_${op.id}`,
            makeTensorProto(calcDtype, [], [qMin]),
        );
        const maxC = builder.createConstant(
            `qle_max_${op.id}`,
            makeTensorProto(calcDtype, [], [qMax]),
        );
        const [clipped] = builder.createOp("Clip", [shifted, minC, maxC], {}, [
            { type: calcDtype, shape: workShape },
        ]);

        let out: ConcreteValueNode = builder.createOp("Cast", [clipped], { to: targetDtype }, [
            { type: targetDtype, shape: workShape },
        ])[0];

        if (blockSize > 0 && shapeXNode) {
            [out] = builder.createOp("Reshape", [out, shapeXNode], {}, [
                { type: targetDtype, shape: outShape },
            ]);
        }

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
