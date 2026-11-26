import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import { DataType, type ConcreteValueNode, type KnownShape } from "../../../OnnxTypes.js";
import { getIntAttr, int64Vec, normalizeAxis, toStaticShape } from "../../../utils/index.js";

export class LowerDequantizeLinearBlockedRecipe implements DecompositionRecipe {
    public readonly name = "LowerDequantizeLinearBlocked";
    public readonly targetOp = "DequantizeLinear";
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 10;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["quantization", "opset21", "blocked"] as const;
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = [
        "Shape",
        "Gather",
        "Concat",
        "Reshape",
        "Unsqueeze",
        "Cast",
        "Sub",
        "Mul",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "DequantizeLinear") return false;
        const blockSize = getIntAttr(op, "block_size", 0);
        if (blockSize <= 0) return false;
        const ins = op.getInputs();
        if (!ins || ins.length < 2) return false;
        const X = ins[0];
        const scale = ins[1];
        if (!Array.isArray(X.shape) || X.shape.length === 0) return false;
        if (!Array.isArray(scale.shape) || scale.shape.length !== X.shape.length) return false;
        const dims = toStaticShape(X.shape);
        const axis = normalizeAxis(getIntAttr(op, "axis", 1), dims.length);
        return dims[axis] < 0 || dims[axis] % blockSize === 0;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs()! as ConcreteValueNode[];
        const X = ins[0];
        const scale = ins[1];
        const zp = ins.length >= 3 && ins[2].id !== "" ? ins[2] : undefined;
        const Y = op.getOutputs()[0];

        const outDtype = (scale.literalType || DataType.FLOAT) as DataType;
        const outShape = Y.shape as KnownShape;
        const rank = X.shape.length;
        const axis = normalizeAxis(getIntAttr(op, "axis", 1), rank);
        const blockSize = getIntAttr(op, "block_size", 0);

        const [shapeXNode] = builder.createOp("Shape", [X], {}, [
            { type: DataType.INT64, shape: [rank] },
        ]);
        const xStatic = toStaticShape(X.shape);
        const workShape = [
            ...xStatic.slice(0, axis),
            xStatic[axis] > 0 ? Math.floor(xStatic[axis] / blockSize) : -1,
            blockSize,
            ...xStatic.slice(axis + 1),
        ] as KnownShape;

        let blockedShapeNode: ConcreteValueNode;
        if (!workShape.includes(-1)) {
            blockedShapeNode = builder.createConstant(
                `dqlb_bshape_${op.id}`,
                int64Vec(workShape as number[]),
            );
        } else {
            const [shapeScaleNode] = builder.createOp("Shape", [scale], {}, [
                { type: DataType.INT64, shape: [rank] },
            ]);

            const parts: ConcreteValueNode[] = [];
            if (axis > 0) {
                const leftIdx = builder.createConstant(
                    `dqlb_l_${op.id}`,
                    int64Vec(Array.from({ length: axis }, (_, i) => i)),
                );
                parts.push(
                    builder.createOp("Gather", [shapeXNode, leftIdx], { axis: 0 }, [
                        { type: DataType.INT64, shape: [axis] },
                    ])[0],
                );
            }
            const axIdx = builder.createConstant(`dqlb_ax_${op.id}`, int64Vec([axis]));
            const [numBlocks1D] = builder.createOp("Gather", [shapeScaleNode, axIdx], { axis: 0 }, [
                { type: DataType.INT64, shape: [1] },
            ]);
            parts.push(numBlocks1D);
            parts.push(builder.createConstant(`dqlb_bs_${op.id}`, int64Vec([blockSize])));
            if (axis < rank - 1) {
                const rightIdx = builder.createConstant(
                    `dqlb_r_${op.id}`,
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

        const [xCast] = builder.createOp("Cast", [X], { to: outDtype }, [
            { type: outDtype, shape: outShape },
        ]);
        const [xBlocked] = builder.createOp("Reshape", [xCast, blockedShapeNode], {}, [
            { type: outDtype, shape: workShape },
        ]);

        const unsqAxis = builder.createConstant(`dqlb_ua_${op.id}`, int64Vec([axis + 1]));
        const [scaleUnsq] = builder.createOp("Unsqueeze", [scale, unsqAxis]);

        let diff: ConcreteValueNode = xBlocked;
        if (zp) {
            const [zpCast] = builder.createOp("Cast", [zp], { to: outDtype }, [
                { type: outDtype, shape: zp.shape as KnownShape },
            ]);
            const [zpUnsq] = builder.createOp("Unsqueeze", [zpCast, unsqAxis]);
            [diff] = builder.createOp("Sub", [xBlocked, zpUnsq], {}, [
                { type: outDtype, shape: workShape },
            ]);
        }

        const [yBlocked] = builder.createOp("Mul", [diff, scaleUnsq], {}, [
            { type: outDtype, shape: workShape },
        ]);
        const [out] = builder.createOp("Reshape", [yBlocked, shapeXNode], {}, [
            { type: outDtype, shape: outShape },
        ]);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
