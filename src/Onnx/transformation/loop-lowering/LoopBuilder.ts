import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../OnnxGraph.js";
import { GraphBuilder } from "../../GraphBuilder.js";
import type { ValueNode, StaticShape, KnownShape, ConcreteValueNode } from "../../OnnxTypes.js";
import { DataType } from "../../OnnxTypes.js";
import OperationNode from "../../OperationNode.js";
import TensorNode from "../../TensorNode.js";
import OnnxEdge from "../../OnnxEdge.js";
import { bool, int64Vec, scalarInt64, uniq, UNKNOWN_SHAPE, zeroTensor } from "../../Utils.js";
import { IndexingMode } from "../../../DecompositionOptions.js";
import type { LoopCarryContext } from "./LoopLoweringRecipe.js";

export interface LoopCarriedVar {
    tag?: string;
    initialVal: ValueNode;
    elemTy: DataType;
    shape: KnownShape;
}

export interface LoopRegionResult {
    innerBuilder: GraphBuilder;
    trip: TensorNode.Class;
    condIn: TensorNode.Class;
    carriedInputs: TensorNode.Class[];
    loopOutputs: TensorNode.Class[];
    vInitial: TensorNode.Class;
    loopOutput: TensorNode.Class;
    loopCtx: LoopCarryContext;
    finalize: (primaryNextStates: ConcreteValueNode[]) => OperationNode.Class;
}

export function createAccumulatorCarry(
    outerBuilder: GraphBuilder,
    elemTy: DataType,
    carryLen: StaticShape | ValueNode,
    tag: string = "acc",
): LoopCarriedVar {
    let initialVal: ValueNode;
    let internalShape: KnownShape;

    if (Array.isArray(carryLen)) {
        const flatLen = carryLen.reduce((a, b) => a * b, 1);
        initialVal = outerBuilder.createConstant(
            `init_carry_${tag}`,
            zeroTensor(elemTy, [flatLen]),
        );
        internalShape = [flatLen];
    } else {
        const zeroScalar = outerBuilder.createConstant(`zero_${tag}`, zeroTensor(elemTy, []));
        const axes0 = outerBuilder.createConstant(`ax0_fl_${tag}`, int64Vec([0]));
        const flatTrip = outerBuilder.createOp("ReduceProd", [carryLen, axes0], { keepdims: 0 })[0];
        const flatTrip1D = outerBuilder.createOp("Unsqueeze", [flatTrip, axes0])[0];
        [initialVal] = outerBuilder.createOp("Expand", [zeroScalar, flatTrip1D], {}, [
            { type: elemTy, shape: UNKNOWN_SHAPE },
        ]);
        internalShape = UNKNOWN_SHAPE;
    }

    return { tag, initialVal, elemTy, shape: internalShape };
}

export function buildForLoopRegion(
    outerBuilder: GraphBuilder,
    totalIters: number | ValueNode,
    elemTy: DataType,
    carryLen: StaticShape | ValueNode,
    tag: string = "loop",
    scopeTag: string = "",
    indexingMode: IndexingMode = IndexingMode.Pointer,
): LoopRegionResult {
    const outerGraph = outerBuilder.graph;

    // 1. Resolve Outer Trip Count & Condition Inputs
    const tripInp =
        typeof totalIters === "number"
            ? outerBuilder.createConstant(`trip_count_${tag}`, scalarInt64(totalIters))
            : totalIters;
    const condInInp = outerBuilder.createConstant(`cond_${tag}`, bool(true));

    // 2. Initialize Inner Graph
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

    // Condition Passthrough
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

    // 3. Carried State Tracking
    const carries: LoopCarriedVar[] = [];
    const carriedInputs: TensorNode.Class[] = [];
    const loopOutputs: TensorNode.Class[] = [];
    const auxNextStates = new Map<string, ConcreteValueNode>();

    const registerCarryInternal = (
        cTag: string,
        initialVal: ValueNode,
        cElemTy: DataType,
        cShape: KnownShape,
        isPrimary: boolean = false,
    ): TensorNode.Class => {
        const idx = carries.length;
        carries.push({ tag: cTag, initialVal, elemTy: cElemTy, shape: cShape });

        const inName = isPrimary ? "carry" : `carry_${cTag}_${idx}`;
        const cIn = innerGraph
            .addNode(uniq(innerGraph, inName))
            .init(new TensorNode.Builder(cElemTy, cShape, "input"))
            .as(TensorNode);
        carriedInputs.push(cIn);

        const outName = isPrimary
            ? `${scopeTag}_carry_out`
            : `${scopeTag}_carry_out_${cTag}_${idx}`;
        const cOut = outerBuilder
            .addNode(
                uniq(outerGraph, outName),
                new TensorNode.Builder(cElemTy, cShape, "intermediate"),
            )
            .as(TensorNode);
        loopOutputs.push(cOut);

        return cIn;
    };

    // Register Carry 0 (Primary Output Accumulator)
    const primaryCarryDef = createAccumulatorCarry(outerBuilder, elemTy, carryLen, tag);
    const vInitial = registerCarryInternal(
        primaryCarryDef.tag ?? "acc",
        primaryCarryDef.initialVal,
        primaryCarryDef.elemTy,
        primaryCarryDef.shape,
        true,
    );

    const loopCtx: LoopCarryContext = {
        indexingMode,
        outerBuilder,
        innerBuilder,
        addCarry: (cTag, initialVal, cElemTy, cShape) =>
            registerCarryInternal(cTag, initialVal, cElemTy, cShape, false),
        setNextState: (carryInputNode, nextState) => {
            auxNextStates.set(carryInputNode.id, nextState);
        },
    };

    // 4. Finalizer: Wires inner outputs and creates the outer Loop node with all carries
    const finalize = (primaryNextStates: ConcreteValueNode[]): OperationNode.Class => {
        const allNextStates: ConcreteValueNode[] = [...primaryNextStates];

        // Append auxiliary AGU states in exact carry registration order
        for (let i = primaryNextStates.length; i < carriedInputs.length; i++) {
            const cIn = carriedInputs[i];
            const nextState = auxNextStates.get(cIn.id);
            if (!nextState) {
                throw new Error(
                    `[LoopBuilder] Missing nextState for registered carry '${cIn.id}' in ${tag}`,
                );
            }
            allNextStates.push(nextState);
        }

        allNextStates.forEach((state, idx) => {
            const def = carries[idx];
            const stateOut = innerGraph
                .addNode(uniq(innerGraph, `${scopeTag}_${idx}_state_out_${tag}`))
                .init(new TensorNode.Builder(def.elemTy, def.shape, "output"))
                .as(TensorNode);
            const idOp = innerGraph
                .addNode(uniq(innerGraph, `${scopeTag}_id_state_out_${tag}_${idx}`))
                .init(new OperationNode.Builder("Identity", [state]))
                .as(OperationNode);

            innerGraph
                .addEdge(state, idOp)
                .init(new OnnxEdge.Builder(state.literalType, state.shape));
            innerGraph.addEdge(idOp, stateOut).init(new OnnxEdge.Builder(def.elemTy, def.shape));
        });

        const loopInputs: ValueNode[] = [tripInp, condInInp, ...carries.map((c) => c.initialVal)];
        const loopOp = outerBuilder
            .addNode(
                uniq(outerGraph, tag),
                new OperationNode.Builder("Loop", loopInputs, {}, [innerGraph]),
            )
            .as(OperationNode);

        loopInputs.forEach((input) => {
            if (outerGraph.hasNode(input.id)) {
                outerBuilder.addEdge(input, loopOp, input.literalType, input.shape);
            }
        });

        loopOutputs.forEach((outNode, idx) => {
            const def = carries[idx];
            outerBuilder.addEdge(loopOp, outNode, def.elemTy, def.shape);
        });

        return loopOp;
    };

    return {
        innerBuilder,
        trip,
        condIn,
        carriedInputs,
        loopOutputs,
        vInitial,
        loopOutput: loopOutputs[0],
        loopCtx,
        finalize,
    };
}
