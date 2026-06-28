import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../OnnxGraph.js";
import { GraphBuilder } from "../../GraphBuilder.js";
import type { ValueNode, StaticShape, KnownShape, ConcreteValueNode } from "../../OnnxTypes.js";
import { DataType } from "../../OnnxTypes.js";
import OperationNode from "../../OperationNode.js";
import TensorNode from "../../TensorNode.js";
import OnnxEdge from "../../OnnxEdge.js";
import { bool, int64Vec, scalarInt64, uniq, UNKNOWN_SHAPE, zeroTensor } from "../../Utils.js"; // Or your new GraphFactories/Constants imports

/**
 * Creates a fully wired ONNX Loop node and its internal subgraph.
 * Supports both static and dynamic trip counts and carry shapes.
 */
export function buildForLoopRegion(
    outerBuilder: GraphBuilder,
    totalIters: number | ValueNode,
    elemTy: DataType,
    carryLen: StaticShape | ValueNode,
    tag: string = "loop",
    scopeTag: string = "", // Pass the scope tag down if needed
): {
    loopOp: OperationNode.Class;
    innerBuilder: GraphBuilder;
    trip: TensorNode.Class;
    condIn: TensorNode.Class;
    vInitial: TensorNode.Class;
    loopOutput: TensorNode.Class;
    finalize: (nextStates: ConcreteValueNode[]) => void;
} {
    const outerGraph = outerBuilder.graph;

    // 1. Resolve Outer Inputs
    const tripInp =
        typeof totalIters === "number"
            ? outerBuilder.createConstant(`trip_count_${tag}`, scalarInt64(totalIters))
            : totalIters;

    const condInInp = outerBuilder.createConstant(`cond_${tag}`, bool(true));

    let vInitialInp: ValueNode;
    let internalCarryShape: KnownShape;

    if (Array.isArray(carryLen)) {
        const flatLen = carryLen.reduce((a, b) => a * b, 1);
        vInitialInp = outerBuilder.createConstant(
            `init_carry_${tag}`,
            zeroTensor(elemTy, [flatLen]),
        );
        internalCarryShape = [flatLen];
    } else {
        const zeroScalar = outerBuilder.createConstant(`zero_${tag}`, zeroTensor(elemTy, []));
        const axes0 = outerBuilder.createConstant(`ax0_fl_${tag}`, int64Vec([0]));
        const flatTrip = outerBuilder.createOp("ReduceProd", [carryLen, axes0], { keepdims: 0 })[0];
        const flatTrip1D = outerBuilder.createOp("Unsqueeze", [flatTrip, axes0])[0];

        [vInitialInp] = outerBuilder.createOp("Expand", [zeroScalar, flatTrip1D], {}, [
            { type: elemTy, shape: UNKNOWN_SHAPE },
        ]);
        internalCarryShape = UNKNOWN_SHAPE;
    }

    // 2. The Loop Body (region/inner graph)
    const innerGraph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
    const innerBuilder = new GraphBuilder(innerGraph, tag);

    const trip = innerGraph
        .addNode(uniq(innerGraph, "iter"))
        .init(new TensorNode.Builder(DataType.INT64, [], "input"))
        .as(TensorNode);
    const condIn = innerGraph
        .addNode(uniq(innerGraph, "cond_in"))
        .init(new TensorNode.Builder(DataType.BOOL, [], "input"))
        .as(TensorNode);
    const vInitial = innerGraph
        .addNode(uniq(innerGraph, "carry"))
        .init(new TensorNode.Builder(elemTy, internalCarryShape, "input"))
        .as(TensorNode);

    const loopInputs = [tripInp, condInInp, vInitialInp];

    // 3. Cond Passthrough
    const idCond = innerGraph
        .addNode(uniq(innerGraph, `id_cond_${tag}`))
        .init(new OperationNode.Builder("Identity", [condIn]))
        .as(OperationNode);
    const condOut = innerGraph
        .addNode(uniq(innerGraph, `cond_out_${tag}`))
        .init(new TensorNode.Builder(DataType.BOOL, [], "output"))
        .as(TensorNode);

    innerGraph.addEdge(condIn, idCond).init(new OnnxEdge.Builder(condIn.literalType, condIn.shape));
    innerGraph
        .addEdge(idCond, condOut)
        .init(new OnnxEdge.Builder(condOut.literalType, condOut.shape));

    // 4. The Finalizer
    const finalize = (nextStates: ConcreteValueNode[]) => {
        nextStates.forEach((state, idx) => {
            const expectedType = idx === 0 ? vInitial.literalType : state.literalType;
            const expectedShape = idx === 0 ? vInitial.shape : state.shape;

            const stateOut = innerGraph
                .addNode(uniq(innerGraph, `${scopeTag}_${idx}_state_out_${tag}`))
                .init(new TensorNode.Builder(expectedType, expectedShape, "output"))
                .as(TensorNode);

            const idOp = innerGraph
                .addNode(uniq(innerGraph, `${scopeTag}_id_state_out_${tag}_${idx}`))
                .init(new OperationNode.Builder("Identity", [state]))
                .as(OperationNode);

            innerGraph
                .addEdge(state, idOp)
                .init(new OnnxEdge.Builder(state.literalType, state.shape));
            innerGraph
                .addEdge(idOp, stateOut)
                .init(new OnnxEdge.Builder(expectedType, expectedShape));
        });
    };

    // 5. The Loop Node
    const loopOp = outerGraph
        .addNode(uniq(outerGraph, tag))
        .init(new OperationNode.Builder("Loop", loopInputs, {}, [innerGraph]))
        .as(OperationNode);

    // 6. Wire Outer Graph Edges
    loopInputs.forEach((input) => {
        if (outerGraph.hasNode(input.id))
            outerGraph
                .addEdge(input, loopOp)
                .init(new OnnxEdge.Builder(input.literalType, input.shape))
                .as(OnnxEdge);
    });

    // 7. Generate Outputs for the Outer Graph
    const loopOutput = outerGraph
        .addNode(uniq(outerGraph, `${scopeTag}_carry_out`))
        .init(new TensorNode.Builder(vInitial.literalType, vInitial.shape, "intermediate"))
        .as(TensorNode);
    outerGraph
        .addEdge(loopOp, loopOutput)
        .init(new OnnxEdge.Builder(vInitial.literalType, vInitial.shape))
        .as(OnnxEdge);

    return { loopOp, innerBuilder, trip, condIn, vInitial, loopOutput, finalize };
}
