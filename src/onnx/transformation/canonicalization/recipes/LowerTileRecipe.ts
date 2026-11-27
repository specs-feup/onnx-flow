import type OperationNode from "../../../ir/OperationNode.js";
import type { GraphBuilder } from "../../../ir/GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../ir/OnnxTypes.js";
import type { DataType } from "../../../ir/OnnxTypes.js";
import {
    int64Vec,
    readConstIntegerVectorFromTensorNode,
    toStaticShape,
} from "../../../utils/index.js";

export class LowerTileRecipe implements DecompositionRecipe {
    public readonly name = "LowerTile";
    public readonly targetOp = "Tile";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 100;
    public readonly shapeSupport = "static" as const;
    public readonly tags = ["constant-repeats"];
    public readonly producedOps = ["Reshape", "Expand", "Identity"];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Tile") return false;
        const ins = op.getInputs();
        if (!ins || ins.length < 2) return false;
        const inShape = toStaticShape(ins[0].shape);
        const repeats = readConstIntegerVectorFromTensorNode(ins[1]);
        return !!repeats && inShape.length === repeats.length && !inShape.includes(-1);
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const [X, repeatsNode] = op.getInputs() as ConcreteValueNode[];
        const Y = op.getOutputs()[0];

        const dtype = X.literalType as DataType;
        const inShape = toStaticShape(X.shape);
        const repeats = readConstIntegerVectorFromTensorNode(repeatsNode)!;

        if (repeats.length === 0 || repeats.every((r) => r === 1)) {
            const [idOut] = builder.createOp("Identity", [X], {}, [
                { type: dtype, shape: inShape as KnownShape },
            ]);
            builder.replaceAllUsesWith(Y, idOut);
            builder.removeNode(op);
            return;
        }

        const interleavedShape: number[] = [];
        const expandedShape: number[] = [];
        const finalShape: number[] = [];

        for (let i = 0; i < inShape.length; i++) {
            interleavedShape.push(1, inShape[i]);
            expandedShape.push(repeats[i], inShape[i]);
            finalShape.push(inShape[i] * repeats[i]);
        }

        const shape1Const = builder.createConstant(`tile_s1_${op.id}`, int64Vec(interleavedShape));
        const expShapeConst = builder.createConstant(`tile_exp_${op.id}`, int64Vec(expandedShape));
        const finalShapeConst = builder.createConstant(`tile_sf_${op.id}`, int64Vec(finalShape));

        const [reshapedIn] = builder.createOp("Reshape", [X, shape1Const], {}, [
            { type: dtype, shape: interleavedShape as KnownShape },
        ]);
        const [expanded] = builder.createOp("Expand", [reshapedIn, expShapeConst], {}, [
            { type: dtype, shape: expandedShape as KnownShape },
        ]);
        const [out] = builder.createOp("Reshape", [expanded, finalShapeConst], {}, [
            { type: dtype, shape: finalShape as KnownShape },
        ]);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
