import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../ir/OnnxGraph.js";
import OperationNode from "../../../ir/OperationNode.js";
import TensorNode from "../../../ir/TensorNode.js";
import { GraphBuilder } from "../../../ir/GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape, ValueNode } from "../../../ir/OnnxTypes.js";
import { DataType } from "../../../ir/OnnxTypes.js";
import {
    asConcreteValueNode,
    bool,
    getIntAttr,
    int64Vec,
    makeTensorProto,
    normalizeAxis,
    scalarInt64,
    toStaticShape,
    uniq,
} from "../../../utils/index.js";
import { createCapturedInput } from "../../loop-lowering/RecipeUtils.js";
import { emitBodyOutput } from "../RnnLoopUtils.js";

export class LowerTopKLoopRecipe implements DecompositionRecipe {
    public readonly name = "LowerTopKLoop";
    public readonly targetOp = "TopK";
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 10;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["loop", "dynamic-k"] as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "Gather",
        "Concat",
        "Range",
        "Reshape",
        "Squeeze",
        "Transpose",
        "ArgMax",
        "ArgMin",
        "ReduceMax",
        "ReduceMin",
        "Equal",
        "Where",
        "Cast",
        "Identity",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "TopK") return false;
        const ins = op.getInputs();
        return (
            ins !== undefined &&
            ins.length >= 2 &&
            ins[0].shape.length > 0 &&
            op.getOutputs().length >= 2
        );
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const outerGraph = builder.graph;
        const ins = op.getInputs()!;
        const X = asConcreteValueNode(ins[0]);
        const K = asConcreteValueNode(ins[1]);
        const outputs = op.getOutputs();
        const YVals = outputs[0];
        const YIdxs = outputs[1];

        const inDims = toStaticShape(X.shape);
        const rank = inDims.length;
        const dtype = X.literalType as DataType;
        const axis = normalizeAxis(getIntAttr(op, "axis", -1), rank);
        const largest = getIntAttr(op, "largest", 1) !== 0;

        const stepShape = [...inDims] as KnownShape;
        stepShape[axis] = 1;
        const squeezedShape = inDims.filter((_, i) => i !== axis) as KnownShape;
        const scanOutShape = [-1, ...squeezedShape] as KnownShape;
        const finalOutShape = [...inDims] as KnownShape;
        finalOutShape[axis] = -1;

        // 1. Trip count = scalar INT64 from K (shape [1])
        const k64: ConcreteValueNode =
            K.literalType === DataType.INT64
                ? K
                : builder.createOp("Cast", [K], { to: DataType.INT64 })[0];
        const scalarShape = builder.createConstant(`topkl_sc_${op.id}`, int64Vec([]));
        const [tripCountNode] = builder.createOp("Reshape", [k64, scalarShape], {}, [
            { type: DataType.INT64, shape: [] },
        ]);
        const condInit = builder.createConstant(`topkl_cond_${op.id}`, bool(true));

        // 2. Build broadcastable index vector 0..dims[axis]-1 along `axis` (handles dynamic dims[axis])
        let dimAxis1D: ConcreteValueNode;
        let dimAxisSc: ConcreteValueNode;
        if (inDims[axis] > 0) {
            dimAxis1D = builder.createConstant(`topkl_d1_${op.id}`, int64Vec([inDims[axis]]));
            dimAxisSc = builder.createConstant(`topkl_ds_${op.id}`, scalarInt64(inDims[axis]));
        } else {
            const [shapeX] = builder.createOp("Shape", [X], {}, [
                { type: DataType.INT64, shape: [rank] },
            ]);
            const ax1D = builder.createConstant(`topkl_ax1_${op.id}`, int64Vec([axis]));
            const axSc = builder.createConstant(`topkl_axs_${op.id}`, scalarInt64(axis));
            [dimAxis1D] = builder.createOp("Gather", [shapeX, ax1D], { axis: 0 }, [
                { type: DataType.INT64, shape: [1] },
            ]);
            [dimAxisSc] = builder.createOp("Gather", [shapeX, axSc], { axis: 0 }, [
                { type: DataType.INT64, shape: [] },
            ]);
        }

        const startC = builder.createConstant(`topkl_start_${op.id}`, scalarInt64(0));
        const deltaC = builder.createConstant(`topkl_delta_${op.id}`, scalarInt64(1));
        const [idxVec] = builder.createOp("Range", [startC, dimAxisSc, deltaC], {}, [
            { type: DataType.INT64, shape: [inDims[axis]] as KnownShape },
        ]);

        let idxBcast: ConcreteValueNode = idxVec;
        if (rank > 1) {
            const bcastShape = new Array<number>(rank).fill(1);
            bcastShape[axis] = inDims[axis];

            const parts: ConcreteValueNode[] = [];
            if (axis > 0) {
                parts.push(
                    builder.createConstant(
                        `topkl_ones_l_${op.id}`,
                        int64Vec(new Array<number>(axis).fill(1)),
                    ),
                );
            }
            parts.push(dimAxis1D);
            if (axis < rank - 1) {
                parts.push(
                    builder.createConstant(
                        `topkl_ones_r_${op.id}`,
                        int64Vec(new Array<number>(rank - 1 - axis).fill(1)),
                    ),
                );
            }
            const [bcastShapeNode] = builder.createOp("Concat", parts, { axis: 0 }, [
                { type: DataType.INT64, shape: [rank] },
            ]);
            [idxBcast] = builder.createOp("Reshape", [idxVec, bcastShapeNode], {}, [
                { type: DataType.INT64, shape: bcastShape as KnownShape },
            ]);
        }

        // 3. Build Loop body: extract 1 extremum per trip and mask it with sentinel
        const loopBody = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const innerBuilder = new GraphBuilder(loopBody, `topkl_body_${op.id}`);

        loopBody
            .addNode(uniq(loopBody, "iter"))
            .init(new TensorNode.Builder(DataType.INT64, [], "input"));
        const condIn = loopBody
            .addNode(uniq(loopBody, "cond_in"))
            .init(new TensorNode.Builder(DataType.BOOL, [], "input"))
            .as(TensorNode);
        const curXIn = loopBody
            .addNode(uniq(loopBody, "cur_x_in"))
            .init(new TensorNode.Builder(dtype, inDims as KnownShape, "input"))
            .as(TensorNode);

        const idxBcastCap = createCapturedInput(loopBody, idxBcast);
        const axesConst = innerBuilder.createConstant(`topkl_axes_${op.id}`, int64Vec([axis]));

        const isInt = dtype === DataType.INT32 || dtype === DataType.INT64;
        const sentinelVal = isInt
            ? largest
                ? -2147483648
                : 2147483647
            : largest
              ? -3.4028234e38
              : 3.4028234e38;
        const sentinelC = innerBuilder.createConstant(
            `topkl_sent_${op.id}`,
            makeTensorProto(dtype, [], [sentinelVal]),
        );

        const argOp = largest ? "ArgMax" : "ArgMin";
        const redOp = largest ? "ReduceMax" : "ReduceMin";

        const [idxT] = innerBuilder.createOp(
            argOp,
            [curXIn],
            { axis, keepdims: 1, select_last_index: 0 },
            [{ type: DataType.INT64, shape: stepShape }],
        );
        const [valT] = innerBuilder.createOp(redOp, [curXIn, axesConst], { keepdims: 1 }, [
            { type: dtype, shape: stepShape },
        ]);

        const [maskT] = innerBuilder.createOp("Equal", [idxBcastCap, idxT], {}, [
            { type: DataType.BOOL, shape: inDims as KnownShape },
        ]);
        const [nextX] = innerBuilder.createOp("Where", [maskT, sentinelC, curXIn], {}, [
            { type: dtype, shape: inDims as KnownShape },
        ]);

        // Squeeze the size-1 `axis` dim so Loop stacks scan outputs along axis 0
        const [valSq] = innerBuilder.createOp("Squeeze", [valT, axesConst], {}, [
            { type: dtype, shape: squeezedShape },
        ]);
        const [idxSq] = innerBuilder.createOp("Squeeze", [idxT, axesConst], {}, [
            { type: DataType.INT64, shape: squeezedShape },
        ]);

        emitBodyOutput(loopBody, condIn, "cond_out");
        emitBodyOutput(loopBody, nextX, "next_x_out");
        emitBodyOutput(loopBody, valSq, "val_scan_out");
        emitBodyOutput(loopBody, idxSq, "idx_scan_out");

        // 4. Emit outer Loop node
        const loopInputs: ValueNode[] = [tripCountNode, condInit, X];
        const loopOp = builder
            .addNode(
                uniq(outerGraph, `TopKLoop_${op.id}`),
                new OperationNode.Builder("Loop", loopInputs, {}, [loopBody]),
            )
            .as(OperationNode);

        for (const inp of loopInputs) {
            if (outerGraph.hasNode(inp.id)) {
                builder.addEdge(inp, loopOp, inp.literalType, inp.shape);
            }
        }

        const finalX = builder
            .addNode(
                uniq(outerGraph, `topkl_final_x_${op.id}`),
                new TensorNode.Builder(dtype, inDims as KnownShape, "intermediate"),
            )
            .as(TensorNode);
        const valsScan = builder
            .addNode(
                uniq(outerGraph, `topkl_vals_scan_${op.id}`),
                new TensorNode.Builder(dtype, scanOutShape, "intermediate"),
            )
            .as(TensorNode);
        const idxsScan = builder
            .addNode(
                uniq(outerGraph, `topkl_idxs_scan_${op.id}`),
                new TensorNode.Builder(DataType.INT64, scanOutShape, "intermediate"),
            )
            .as(TensorNode);

        builder.addEdge(loopOp, finalX, dtype, inDims as KnownShape);
        builder.addEdge(loopOp, valsScan, dtype, scanOutShape);
        builder.addEdge(loopOp, idxsScan, DataType.INT64, scanOutShape);

        let finalVals: ConcreteValueNode = valsScan;
        let finalIdxs: ConcreteValueNode = idxsScan;

        if (axis !== 0 && rank > 1) {
            const perm: number[] = [];
            for (let i = 1; i <= axis; i++) perm.push(i);
            perm.push(0);
            for (let i = axis + 1; i < rank; i++) perm.push(i);

            [finalVals] = builder.createOp("Transpose", [valsScan], { perm }, [
                { type: dtype, shape: finalOutShape },
            ]);
            [finalIdxs] = builder.createOp("Transpose", [idxsScan], { perm }, [
                { type: DataType.INT64, shape: finalOutShape },
            ]);
        }

        builder.replaceAllUsesWith(YVals, finalVals);
        builder.replaceAllUsesWith(YIdxs, finalIdxs);
        builder.removeNode(op);
    }
}
