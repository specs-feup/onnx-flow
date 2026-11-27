import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../ir/OnnxGraph.js";
import OperationNode from "../../../ir/OperationNode.js";
import TensorNode from "../../../ir/TensorNode.js";
import { GraphBuilder } from "../../../ir/GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape, ValueNode } from "../../../ir/OnnxTypes.js";
import { DataType } from "../../../ir/OnnxTypes.js";
import {
    bool,
    getIntAttr,
    int64Vec,
    normalizeAxis,
    scalarInt64,
    toStaticShape,
    uniq,
} from "../../../utils/index.js";
import { createCapturedInput } from "../../loop-lowering/RecipeUtils.js";
import { emitBodyOutput } from "../RnnLoopUtils.js";

abstract class BaseLowerArgMinMaxLoopRecipe implements DecompositionRecipe {
    public abstract readonly name: string;
    public abstract readonly targetOp: "ArgMax" | "ArgMin";
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 10;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["loop", "dynamic-axis"] as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "Gather",
        "ConstantOfShape",
        "Cast",
        "Greater",
        "GreaterOrEqual",
        "Less",
        "LessOrEqual",
        "Where",
        "Unsqueeze",
        "Identity",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== this.targetOp) return false;
        const X = op.getInputs()?.[0];
        return X !== undefined && X.shape.length > 0;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const outerGraph = builder.graph;
        const X = op.getInputs()![0] as ConcreteValueNode;
        const Y = op.getOutputs()[0];

        const inDims = toStaticShape(X.shape);
        const rank = inDims.length;
        const dtype = X.literalType as DataType;

        const axis = normalizeAxis(getIntAttr(op, "axis", 0), rank);
        const keepdims = getIntAttr(op, "keepdims", 1);
        const selectLast = getIntAttr(op, "select_last_index", 0) !== 0;

        const reducedShape = inDims.filter((_, i) => i !== axis) as KnownShape;
        const keepdimsShape = [...inDims] as KnownShape;
        keepdimsShape[axis] = 1;

        // 1. Trip count = dims[axis] (static or dynamic)
        let tripCountNode: ConcreteValueNode;
        if (inDims[axis] > 0) {
            tripCountNode = builder.createConstant(`argl_trip_${op.id}`, scalarInt64(inDims[axis]));
        } else {
            const [shapeX] = builder.createOp("Shape", [X], {}, [
                { type: DataType.INT64, shape: [rank] },
            ]);
            const axSc = builder.createConstant(`argl_axsc_${op.id}`, scalarInt64(axis));
            [tripCountNode] = builder.createOp("Gather", [shapeX, axSc], { axis: 0 }, [
                { type: DataType.INT64, shape: [] },
            ]);
        }
        const condInit = builder.createConstant(`argl_cond_${op.id}`, bool(true));

        // 2. Initial carries: bestVal = X[0 along axis], bestIdx = zeros(reducedShape)
        const zeroI64 = builder.createConstant(`argl_z64_${op.id}`, scalarInt64(0));
        const [bestVal0] = builder.createOp("Gather", [X, zeroI64], { axis }, [
            { type: dtype, shape: reducedShape },
        ]);

        let bestIdx0: ConcreteValueNode;
        if (rank === 1) {
            bestIdx0 = zeroI64;
        } else {
            const [redShapeNode] = builder.createOp("Shape", [bestVal0], {}, [
                { type: DataType.INT64, shape: [rank - 1] },
            ]);
            const [zerosF] = builder.createOp("ConstantOfShape", [redShapeNode], {}, [
                { type: DataType.FLOAT, shape: reducedShape },
            ]);
            [bestIdx0] = builder.createOp("Cast", [zerosF], { to: DataType.INT64 }, [
                { type: DataType.INT64, shape: reducedShape },
            ]);
        }

        // 3. Build Loop body: scan iter = 0..dims[axis]-1
        const loopBody = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const innerBuilder = new GraphBuilder(loopBody, `argl_body_${op.id}`);

        const iterNode = loopBody
            .addNode(uniq(loopBody, "iter"))
            .init(new TensorNode.Builder(DataType.INT64, [], "input"))
            .as(TensorNode);
        const condIn = loopBody
            .addNode(uniq(loopBody, "cond_in"))
            .init(new TensorNode.Builder(DataType.BOOL, [], "input"))
            .as(TensorNode);
        const bestValIn = loopBody
            .addNode(uniq(loopBody, "best_val_in"))
            .init(new TensorNode.Builder(dtype, reducedShape, "input"))
            .as(TensorNode);
        const bestIdxIn = loopBody
            .addNode(uniq(loopBody, "best_idx_in"))
            .init(new TensorNode.Builder(DataType.INT64, reducedShape, "input"))
            .as(TensorNode);

        const xCap = createCapturedInput(loopBody, X);
        const [curVal] = innerBuilder.createOp("Gather", [xCap, iterNode], { axis }, [
            { type: dtype, shape: reducedShape },
        ]);

        const cmpOp =
            this.targetOp === "ArgMax"
                ? selectLast
                    ? "GreaterOrEqual"
                    : "Greater"
                : selectLast
                  ? "LessOrEqual"
                  : "Less";

        const [isBetter] = innerBuilder.createOp(cmpOp, [curVal, bestValIn], {}, [
            { type: DataType.BOOL, shape: reducedShape },
        ]);
        const [nextVal] = innerBuilder.createOp("Where", [isBetter, curVal, bestValIn], {}, [
            { type: dtype, shape: reducedShape },
        ]);
        const [nextIdx] = innerBuilder.createOp("Where", [isBetter, iterNode, bestIdxIn], {}, [
            { type: DataType.INT64, shape: reducedShape },
        ]);

        emitBodyOutput(loopBody, condIn, "cond_out");
        emitBodyOutput(loopBody, nextVal, "best_val_out");
        emitBodyOutput(loopBody, nextIdx, "best_idx_out");

        // 4. Emit outer Loop node
        const loopInputs: ValueNode[] = [tripCountNode, condInit, bestVal0, bestIdx0];
        const loopOp = builder
            .addNode(
                uniq(outerGraph, `ArgLoop_${op.id}`),
                new OperationNode.Builder("Loop", loopInputs, {}, [loopBody]),
            )
            .as(OperationNode);

        for (const inp of loopInputs) {
            if (outerGraph.hasNode(inp.id)) {
                builder.addEdge(inp, loopOp, inp.literalType, inp.shape);
            }
        }

        const finalVal = builder
            .addNode(
                uniq(outerGraph, `argl_final_val_${op.id}`),
                new TensorNode.Builder(dtype, reducedShape, "intermediate"),
            )
            .as(TensorNode);
        const finalIdx = builder
            .addNode(
                uniq(outerGraph, `argl_final_idx_${op.id}`),
                new TensorNode.Builder(DataType.INT64, reducedShape, "intermediate"),
            )
            .as(TensorNode);

        builder.addEdge(loopOp, finalVal, dtype, reducedShape);
        builder.addEdge(loopOp, finalIdx, DataType.INT64, reducedShape);

        let out: ConcreteValueNode = finalIdx;
        if (keepdims === 1) {
            const axVec = builder.createConstant(`argl_axvec_${op.id}`, int64Vec([axis]));
            [out] = builder.createOp("Unsqueeze", [finalIdx, axVec], {}, [
                { type: DataType.INT64, shape: keepdimsShape },
            ]);
        }

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}

export class LowerArgMaxLoopRecipe extends BaseLowerArgMinMaxLoopRecipe {
    public readonly name = "LowerArgMaxLoop";
    public readonly targetOp = "ArgMax" as const;
}

export class LowerArgMinLoopRecipe extends BaseLowerArgMinMaxLoopRecipe {
    public readonly name = "LowerArgMinLoop";
    public readonly targetOp = "ArgMin" as const;
}
