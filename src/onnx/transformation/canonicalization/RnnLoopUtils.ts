import Graph from "@specs-feup/flow/graph/Graph";
import { GraphBuilder } from "../../GraphBuilder.js";
import OnnxEdge from "../../OnnxEdge.js";
import OnnxGraph from "../../OnnxGraph.js";
import type { ValueNode, ConcreteValueNode, KnownShape } from "../../OnnxTypes.js";
import { DataType } from "../../OnnxTypes.js";
import OperationNode from "../../OperationNode.js";
import TensorNode from "../../TensorNode.js";
import {
    getAttr,
    getIntAttr,
    toStaticShape,
    int64Vec,
    scalarInt64,
    bool,
    makeTensorProto,
    uniq,
} from "../../utils/index.js";
import { createCapturedInput } from "../loop-lowering/RecipeUtils.js";

export const isPresent = (n?: ValueNode): n is ConcreteValueNode => n !== undefined && n.id !== "";

export function matchRecurrentLoopOp(
    op: OperationNode.Class,
    expectedOp: "RNN" | "GRU" | "LSTM",
    gateMultiplier: number,
): boolean {
    if (op.type !== expectedOp) return false;
    const ins = op.getInputs() as ConcreteValueNode[] | undefined;
    if (!ins || ins.length < 3) return false;
    if (getAttr(op, "activation_alpha") !== undefined) return false;
    if (getAttr(op, "activation_beta") !== undefined) return false;

    const layout = getIntAttr(op, "layout", 0);
    if (layout !== 0 && layout !== 1) return false;

    const xShape = toStaticShape(ins[0].shape);
    const wShape = toStaticShape(ins[1].shape);
    return (
        xShape.length === 3 &&
        wShape.length === 3 &&
        wShape[0] > 0 &&
        wShape[1] > 0 &&
        wShape[1] % gateMultiplier === 0 &&
        wShape[2] > 0
    );
}

export interface RecurrentPrep {
    XSeq: ConcreteValueNode;
    dtype: DataType;
    layout: number;
    seqLen: number;
    batchSize: number;
    inputSize: number;
    numDirs: number;
    hiddenSize: number;
    tripCountNode: ConcreteValueNode;
    condInit: ConcreteValueNode;
    seqLens2D?: ConcreteValueNode;
    clipVal?: number;
    ax0: ConcreteValueNode;
    ax1: ConcreteValueNode;
    step1: ConcreteValueNode;
}

export function prepareRecurrentInputs(
    op: OperationNode.Class,
    builder: GraphBuilder,
    gateMultiplier: number,
    prefix: string,
): RecurrentPrep {
    const ins = op.getInputs() as ConcreteValueNode[];
    const layout = getIntAttr(op, "layout", 0);
    const dtype = ins[0].literalType as DataType;
    const rawXShape = toStaticShape(ins[0].shape);
    const wShape = toStaticShape(ins[1].shape);

    const seqLen = layout === 1 ? rawXShape[1] : rawXShape[0];
    const batchSize = layout === 1 ? rawXShape[0] : rawXShape[1];
    const inputSize = wShape[2];
    const numDirs = wShape[0];
    const hiddenSize = wShape[1] / gateMultiplier;

    const XSeq =
        layout === 1
            ? builder.createOp("Transpose", [ins[0]], { perm: [1, 0, 2] }, [
                  { type: dtype, shape: [seqLen, batchSize, inputSize] as KnownShape },
              ])[0]
            : ins[0];

    const ax0 = builder.createConstant(`${prefix}_ax0_${op.id}`, int64Vec([0]));
    const ax1 = builder.createConstant(`${prefix}_ax1_${op.id}`, int64Vec([1]));
    const step1 = builder.createConstant(`${prefix}_step1_${op.id}`, int64Vec([1]));

    let tripCountNode: ConcreteValueNode;
    if (seqLen > 0) {
        tripCountNode = builder.createConstant(`${prefix}_trip_${op.id}`, scalarInt64(seqLen));
    } else {
        const [xShapeNode] = builder.createOp("Shape", [XSeq], {}, [
            { type: DataType.INT64, shape: [3] },
        ]);
        const zeroIdx = builder.createConstant(`${prefix}_zidx_${op.id}`, scalarInt64(0));
        [tripCountNode] = builder.createOp("Gather", [xShapeNode, zeroIdx], { axis: 0 }, [
            { type: DataType.INT64, shape: [] },
        ]);
    }

    const condInit = builder.createConstant(`${prefix}_cond_${op.id}`, bool(true));

    let seqLens2D: ConcreteValueNode | undefined;
    if (isPresent(ins[4])) {
        const [sl64] = builder.createOp("Cast", [ins[4]], { to: DataType.INT64 }, [
            { type: DataType.INT64, shape: [batchSize] as KnownShape },
        ]);
        [seqLens2D] = builder.createOp("Unsqueeze", [sl64, ax1], {}, [
            { type: DataType.INT64, shape: [batchSize, 1] as KnownShape },
        ]);
    }

    const rawClip = getAttr(op, "clip");
    const clipVal = typeof rawClip === "number" ? rawClip : undefined;

    return {
        XSeq,
        dtype,
        layout,
        seqLen,
        batchSize,
        inputSize,
        numDirs,
        hiddenSize,
        tripCountNode,
        condInit,
        seqLens2D,
        clipVal,
        ax0,
        ax1,
        step1,
    } as RecurrentPrep;
}

export function sliceAxis(
    builder: GraphBuilder,
    src: ConcreteValueNode,
    axisConst: ConcreteValueNode,
    step1: ConcreteValueNode,
    start: number,
    end: number,
    shape: number[],
    tag: string,
): ConcreteValueNode {
    const s = builder.createConstant(`${tag}_s`, int64Vec([start]));
    const e = builder.createConstant(`${tag}_e`, int64Vec([end]));
    return builder.createOp("Slice", [src, s, e, axisConst, step1], {}, [
        { type: src.literalType as DataType, shape: shape as KnownShape },
    ])[0];
}

export function buildInitialState(
    builder: GraphBuilder,
    prep: RecurrentPrep,
    initialTensor: ConcreteValueNode | undefined,
    dir: number,
    tag: string,
): ConcreteValueNode {
    const bhShape: KnownShape = [prep.batchSize, prep.hiddenSize];

    if (initialTensor !== undefined) {
        const dirFirst =
            prep.layout === 1
                ? builder.createOp("Transpose", [initialTensor], { perm: [1, 0, 2] }, [
                      {
                          type: prep.dtype,
                          shape: [prep.numDirs, prep.batchSize, prep.hiddenSize] as KnownShape,
                      },
                  ])[0]
                : initialTensor;
        const sliced = sliceAxis(
            builder,
            dirFirst,
            prep.ax0,
            prep.step1,
            dir,
            dir + 1,
            [1, prep.batchSize, prep.hiddenSize],
            `${tag}_sl_${dir}`,
        );
        return builder.createOp("Squeeze", [sliced, prep.ax0], {}, [
            { type: prep.dtype, shape: bhShape },
        ])[0];
    }

    if (prep.batchSize > 0) {
        return builder.createConstant(
            `${tag}_zero_${dir}`,
            makeTensorProto(
                prep.dtype,
                [prep.batchSize, prep.hiddenSize],
                new Array<number>(prep.batchSize * prep.hiddenSize).fill(0),
            ),
        );
    }

    // Dynamic batch size: allocate zeros via ConstantOfShape([batch_size, hidden_size])
    const [xShapeNode] = builder.createOp("Shape", [prep.XSeq], {}, [
        { type: DataType.INT64, shape: [3] },
    ]);
    const [bDim1D] = builder.createOp("Gather", [xShapeNode, prep.ax1], { axis: 0 }, [
        { type: DataType.INT64, shape: [1] },
    ]);
    const hDim1D = builder.createConstant(`${tag}_hdim_${dir}`, int64Vec([prep.hiddenSize]));
    const [bhShapeNode] = builder.createOp("Concat", [bDim1D, hDim1D], { axis: 0 }, [
        { type: DataType.INT64, shape: [2] },
    ]);
    let [zeros] = builder.createOp("ConstantOfShape", [bhShapeNode], {}, [
        { type: DataType.FLOAT, shape: bhShape },
    ]);
    if (prep.dtype !== DataType.FLOAT) {
        [zeros] = builder.createOp("Cast", [zeros], { to: prep.dtype }, [
            { type: prep.dtype, shape: bhShape },
        ]);
    }
    return zeros;
}

export function applyClipIfNeeded(
    builder: GraphBuilder,
    x: ConcreteValueNode,
    dtype: DataType,
    shape: KnownShape,
    clipVal: number | undefined,
    tag: string,
): ConcreteValueNode {
    if (clipVal === undefined) return x;
    const minC = builder.createConstant(`${tag}_cmin`, makeTensorProto(dtype, [], [-clipVal]));
    const maxC = builder.createConstant(`${tag}_cmax`, makeTensorProto(dtype, [], [clipVal]));
    return builder.createOp("Clip", [x, minC, maxC], {}, [{ type: dtype, shape }])[0];
}

export function emitBodyOutput(loopBody: OnnxGraph.Class, src: ValueNode, name: string): void {
    const idOp = loopBody
        .addNode(uniq(loopBody, `id_${name}`))
        .init(new OperationNode.Builder("Identity", [src]))
        .as(OperationNode);
    const outT = loopBody
        .addNode(uniq(loopBody, name))
        .init(new TensorNode.Builder(src.literalType, src.shape as KnownShape, "output"))
        .as(TensorNode);
    loopBody.addEdge(src, idOp).init(new OnnxEdge.Builder(src.literalType, src.shape));
    loopBody.addEdge(idOp, outT).init(new OnnxEdge.Builder(outT.literalType, outT.shape));
}

export function buildTimestepGather(
    loopBody: OnnxGraph.Class,
    innerBuilder: GraphBuilder,
    prep: RecurrentPrep,
    iterNode: TensorNode.Class,
    isReverse: boolean,
    tag: string,
): { tNode: ConcreteValueNode; xt: ConcreteValueNode; activeMask?: ConcreteValueNode } {
    const xCap = createCapturedInput(loopBody, prep.XSeq);
    let tNode: ConcreteValueNode = iterNode;

    if (isReverse) {
        const tripCap = createCapturedInput(loopBody, prep.tripCountNode);
        const oneC = innerBuilder.createConstant(`${tag}_one`, scalarInt64(1));
        const [lenMinus1] = innerBuilder.createOp("Sub", [tripCap, oneC], {}, [
            { type: DataType.INT64, shape: [] },
        ]);
        [tNode] = innerBuilder.createOp("Sub", [lenMinus1, iterNode], {}, [
            { type: DataType.INT64, shape: [] },
        ]);
    }

    const [xt] = innerBuilder.createOp("Gather", [xCap, tNode], { axis: 0 }, [
        { type: prep.dtype, shape: [prep.batchSize, prep.inputSize] as KnownShape },
    ]);

    let activeMask: ConcreteValueNode | undefined;
    if (prep.seqLens2D !== undefined) {
        const slCap = createCapturedInput(loopBody, prep.seqLens2D);
        [activeMask] = innerBuilder.createOp("Less", [tNode, slCap], {}, [
            { type: DataType.BOOL, shape: [prep.batchSize, 1] as KnownShape },
        ]);
    }

    if (activeMask !== undefined) {
        return { tNode, xt, activeMask };
    } else {
        return { tNode, xt };
    }
}

export function emitDirectionLoop(
    outerBuilder: GraphBuilder,
    prep: RecurrentPrep,
    loopBody: OnnxGraph.Class,
    initialStates: ConcreteValueNode[],
    isReverse: boolean,
    tag: string,
): { finalStates: ConcreteValueNode[]; yDir4D: ConcreteValueNode } {
    const outerGraph = outerBuilder.graph;
    const bhShape: KnownShape = [prep.batchSize, prep.hiddenSize];
    const loopInputs: ValueNode[] = [prep.tripCountNode, prep.condInit, ...initialStates];

    const loopOp = outerBuilder
        .addNode(
            uniq(outerGraph, `${tag}_loop`),
            new OperationNode.Builder("Loop", loopInputs, {}, [loopBody]),
        )
        .as(OperationNode);

    for (const inp of loopInputs) {
        if (outerGraph.hasNode(inp.id)) {
            outerBuilder.addEdge(inp, loopOp, inp.literalType, inp.shape);
        }
    }

    const finalStates: ConcreteValueNode[] = initialStates.map((_, idx) => {
        const stOut = outerBuilder
            .addNode(
                uniq(outerGraph, `${tag}_state_final_${idx}`),
                new TensorNode.Builder(prep.dtype, bhShape, "intermediate"),
            )
            .as(TensorNode);
        outerBuilder.addEdge(loopOp, stOut, prep.dtype, bhShape);
        return stOut;
    });

    const scanShape: KnownShape = [prep.seqLen, prep.batchSize, prep.hiddenSize];
    let yScan: ConcreteValueNode = outerBuilder
        .addNode(
            uniq(outerGraph, `${tag}_y_scan`),
            new TensorNode.Builder(prep.dtype, scanShape, "intermediate"),
        )
        .as(TensorNode);
    outerBuilder.addEdge(loopOp, yScan, prep.dtype, scanShape);

    if (isReverse) {
        const revStart = outerBuilder.createConstant(`${tag}_revs`, int64Vec([-1]));
        const revEnd = outerBuilder.createConstant(`${tag}_reve`, int64Vec([-2147483648]));
        const revStep = outerBuilder.createConstant(`${tag}_revst`, int64Vec([-1]));
        [yScan] = outerBuilder.createOp("Slice", [yScan, revStart, revEnd, prep.ax0, revStep], {}, [
            { type: prep.dtype, shape: scanShape },
        ]);
    }

    const [yDir4D] = outerBuilder.createOp("Unsqueeze", [yScan, prep.ax1], {}, [
        {
            type: prep.dtype,
            shape: [prep.seqLen, 1, prep.batchSize, prep.hiddenSize] as KnownShape,
        },
    ]);

    return { finalStates, yDir4D };
}

export function createLoopBodyShell(
    prep: RecurrentPrep,
    numStates: number,
    tag: string,
): {
    loopBody: OnnxGraph.Class;
    innerBuilder: GraphBuilder;
    iterNode: TensorNode.Class;
    condIn: TensorNode.Class;
    stateIns: TensorNode.Class[];
} {
    const loopBody = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
    const innerBuilder = new GraphBuilder(loopBody, tag);
    const bhShape: KnownShape = [prep.batchSize, prep.hiddenSize];

    const iterNode = loopBody
        .addNode(uniq(loopBody, "iter"))
        .init(new TensorNode.Builder(DataType.INT64, [], "input"))
        .as(TensorNode);
    const condIn = loopBody
        .addNode(uniq(loopBody, "cond_in"))
        .init(new TensorNode.Builder(DataType.BOOL, [], "input"))
        .as(TensorNode);

    const stateIns: TensorNode.Class[] = [];
    for (let i = 0; i < numStates; i++) {
        stateIns.push(
            loopBody
                .addNode(uniq(loopBody, `state_in_${i}`))
                .init(new TensorNode.Builder(prep.dtype, bhShape, "input"))
                .as(TensorNode),
        );
    }

    return { loopBody, innerBuilder, iterNode, condIn, stateIns };
}

export function finalizeRecurrentOutputs(
    op: OperationNode.Class,
    builder: GraphBuilder,
    prep: RecurrentPrep,
    dirYs: ConcreteValueNode[],
    dirYhs: ConcreteValueNode[],
    dirYcs?: ConcreteValueNode[],
): void {
    const outputs = op.getOutputs();
    const YOut = outputs[0];
    const YhOut = outputs[1];
    const YcOut = outputs[2];

    if (isPresent(YOut)) {
        let finalY =
            prep.numDirs === 1
                ? dirYs[0]
                : builder.createOp("Concat", dirYs, { axis: 1 }, [
                      {
                          type: prep.dtype,
                          shape: [
                              prep.seqLen,
                              prep.numDirs,
                              prep.batchSize,
                              prep.hiddenSize,
                          ] as KnownShape,
                      },
                  ])[0];
        if (prep.layout === 1) {
            [finalY] = builder.createOp("Transpose", [finalY], { perm: [2, 0, 1, 3] }, [
                {
                    type: prep.dtype,
                    shape: [
                        prep.batchSize,
                        prep.seqLen,
                        prep.numDirs,
                        prep.hiddenSize,
                    ] as KnownShape,
                },
            ]);
        }
        builder.replaceAllUsesWith(YOut, finalY);
    }

    const wireStateOut = (outNode: ValueNode | undefined, dirStates: ConcreteValueNode[]) => {
        if (!isPresent(outNode)) return;
        let finalSt =
            prep.numDirs === 1
                ? dirStates[0]
                : builder.createOp("Concat", dirStates, { axis: 0 }, [
                      {
                          type: prep.dtype,
                          shape: [prep.numDirs, prep.batchSize, prep.hiddenSize] as KnownShape,
                      },
                  ])[0];
        if (prep.layout === 1) {
            [finalSt] = builder.createOp("Transpose", [finalSt], { perm: [1, 0, 2] }, [
                {
                    type: prep.dtype,
                    shape: [prep.batchSize, prep.numDirs, prep.hiddenSize] as KnownShape,
                },
            ]);
        }
        builder.replaceAllUsesWith(outNode, finalSt);
    };

    wireStateOut(YhOut, dirYhs);
    if (dirYcs !== undefined) {
        wireStateOut(YcOut, dirYcs);
    }

    builder.removeNode(op);
}
