import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../OnnxGraph.js";
import type { GraphPass } from "../../PassManager.js";
import OperationNode from "../../OperationNode.js";
import TensorNode from "../../TensorNode.js";
import ConstantNode from "../../ConstantNode.js";
import RegionArgumentNode from "../../RegionArgumentNode.js";
import OnnxEdge from "../../OnnxEdge.js";
import { GraphBuilder } from "../../GraphBuilder.js";
import { TrackedGraphBuilder } from "../tracking/TrackedGraphBuilder.js";
import type { HistoryManager } from "../tracking/HistoryManager.js";
import { TransformationOpportunity } from "../TransformationOpportunity.js";
import { DataType, type ValueNode } from "../../OnnxTypes.js";
import {
    bool,
    readScalarFromTensorNode,
    scalarInt64,
    topologicalSortOperationNodes,
    uniq,
} from "../../Utils.js";

export interface LoopTilingOptions {
    /** Tile size (strip-mine factor). Must be > 1 to tile. */
    tileSize: number;
}

export class LoopTilingPass implements GraphPass {
    public readonly name = "LoopTiling";
    private tileSize: number;

    constructor(options: LoopTilingOptions = { tileSize: 16 }) {
        this.tileSize = options.tileSize;
    }

    public findOpportunities(graph: OnnxGraph.Class): TransformationOpportunity[] {
        if (this.tileSize <= 1) return [];

        const opportunities: TransformationOpportunity[] = [];
        const ops = topologicalSortOperationNodes(graph);

        for (const op of ops) {
            if (this.canTile(op)) {
                opportunities.push(
                    new TransformationOpportunity(
                        "LoopTiling",
                        op.id,
                        `Tile Loop '${op.id}' with tile size ${this.tileSize}`,
                        (builder: GraphBuilder) => {
                            this.tileLoop(op, builder);
                        },
                    ),
                );
            }
        }

        return opportunities;
    }

    public run(graph: OnnxGraph.Class, historyManager: HistoryManager): boolean {
        if (this.tileSize <= 1) return false;

        let changed = false;
        const opportunities = this.findOpportunities(graph);

        for (const opp of opportunities) {
            if (graph.hasNode(opp.targetNodeId)) {
                const builder = new TrackedGraphBuilder(
                    graph,
                    opp.id,
                    opp.description,
                    "loop_tiling",
                );
                opp.apply(builder);
                const patch = builder.commitPatch();
                historyManager.pushPatch(patch);
                changed = true;
            }
        }

        return changed;
    }

    private canTile(op: OperationNode.Class): boolean {
        if (op.type !== "Loop") return false;
        if (op.getMetadata("tiled") === true) return false;
        if (op.regions.length === 0) return false;

        const body = op.regions[0];
        // Skip if already a nested loop
        const hasInnerLoop = body
            .getOperationNodes()
            .toArray()
            .some((innerOp) => innerOp.type === "Loop");
        if (hasInnerLoop) return false;

        // If static trip count is smaller than or equal to tileSize, no need to tile
        const inputs = op.getInputs() ?? [];
        if (inputs.length < 2) return false;
        const tripNode = inputs[0];
        if (tripNode.is(ConstantNode)) {
            const staticTrips = readScalarFromTensorNode(tripNode);
            if (staticTrips !== undefined && staticTrips <= this.tileSize) {
                return false;
            }
        }

        return true;
    }

    private tileLoop(loopOp: OperationNode.Class, rootBuilder: GraphBuilder): void {
        const T = this.tileSize;
        const loopInputs = loopOp.getInputs()!;
        const totalTripNode = loopInputs[0];
        const condInitNode = loopInputs[1];
        const initialCarries = loopInputs.slice(2);
        const origLoopOutputs = loopOp.getOutputs() as TensorNode.Class[];

        const innerBody = loopOp.regions[0];
        const innerInputNodes = innerBody.getInputTensorNodes().toArray();
        const innerIterNode = innerInputNodes[0];

        // 1. Compute Outer Trip Count in Root Graph
        let outerTripInput: ValueNode;
        const staticTotalTrips = totalTripNode.is(ConstantNode)
            ? readScalarFromTensorNode(totalTripNode)
            : undefined;

        if (staticTotalTrips !== undefined) {
            const outerTrips = Math.ceil(staticTotalTrips / T);
            outerTripInput = rootBuilder.createConstant(
                `outer_trips_${loopOp.id}`,
                scalarInt64(outerTrips),
            );
        } else {
            // Dynamic ceiling division: (M + T - 1) / T
            const tMinus1 = rootBuilder.createConstant(`tile_m1_${loopOp.id}`, scalarInt64(T - 1));
            const tConst = rootBuilder.createConstant(`tile_sz_${loopOp.id}`, scalarInt64(T));
            const [num] = rootBuilder.createOp("Add", [totalTripNode, tMinus1]);
            [outerTripInput] = rootBuilder.createOp("Div", [num, tConst]);
        }

        // 2. Build Outer Tile Loop Body Graph
        const outerBody = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const outerBodyBuilder = new GraphBuilder(outerBody, `tile_outer_${loopOp.id}`);

        const outerIter = outerBody
            .addNode(uniq(outerBody, "tile_iter"))
            .init(new TensorNode.Builder(DataType.INT64, [], "input"))
            .as(TensorNode);
        const outerCondIn = outerBody
            .addNode(uniq(outerBody, "tile_cond_in"))
            .init(new TensorNode.Builder(DataType.BOOL, [], "input"))
            .as(TensorNode);

        // Outer condition passthrough
        const idCond = outerBody
            .addNode(uniq(outerBody, `id_tile_cond_${loopOp.id}`))
            .init(new OperationNode.Builder("Identity", [outerCondIn]))
            .as(OperationNode);
        const outerCondOut = outerBody
            .addNode(uniq(outerBody, `tile_cond_out_${loopOp.id}`))
            .init(new TensorNode.Builder(DataType.BOOL, [], "output"))
            .as(TensorNode);
        outerBody
            .addEdge(outerCondIn, idCond)
            .init(new OnnxEdge.Builder(DataType.BOOL, outerCondIn.shape));
        outerBody
            .addEdge(idCond, outerCondOut)
            .init(new OnnxEdge.Builder(DataType.BOOL, outerCondOut.shape));

        // Create formal carry inputs in Outer Body matching the original Loop's carries
        const outerCarryInputs: TensorNode.Class[] = initialCarries.map((initVal, idx) =>
            outerBody
                .addNode(uniq(outerBody, `tile_carry_in_${idx}`))
                .init(new TensorNode.Builder(initVal.literalType, initVal.shape, "input"))
                .as(TensorNode),
        );

        // 3. Bubble Up RegionArgumentNodes for Scope Validation
        // Any outer variable captured by innerBody must also be present in outerBody
        const innerCaptures = innerBody.getNodes().filterIs(RegionArgumentNode).toArray();
        for (const cap of innerCaptures) {
            if (!outerBody.hasNode(cap.id)) {
                outerBody
                    .addNode(cap.id)
                    .init(
                        new RegionArgumentNode.Builder(
                            cap.index,
                            cap.originalName,
                            cap.literalType,
                            cap.shape,
                        ),
                    );
            }
        }

        // 4. Compute baseIdx = outerIter * T and Inner Loop Trip Count inside Outer Body
        const tileSizeOuterConst = outerBodyBuilder.createConstant(`tile_size`, scalarInt64(T));
        const [baseIdx] = outerBodyBuilder.createOp("Mul", [outerIter, tileSizeOuterConst]);

        let innerTripInput: ValueNode;
        if (staticTotalTrips !== undefined && staticTotalTrips % T === 0) {
            // Exact multiple fast-path
            innerTripInput = tileSizeOuterConst;
        } else if (staticTotalTrips !== undefined) {
            const totalConst = outerBodyBuilder.createConstant(
                `total_trips`,
                scalarInt64(staticTotalTrips),
            );
            const [rem] = outerBodyBuilder.createOp("Sub", [totalConst, baseIdx]);
            [innerTripInput] = outerBodyBuilder.createOp("Min", [tileSizeOuterConst, rem]);
        } else {
            // Capture dynamic totalTripNode into outerBody
            const capturedTotal = outerBody.hasNode(totalTripNode.id)
                ? (outerBody.getNodeById(totalTripNode.id) as ValueNode)
                : outerBody
                      .addNode(totalTripNode.id)
                      .init(
                          new RegionArgumentNode.Builder(
                              innerCaptures.length,
                              totalTripNode.id,
                              totalTripNode.literalType,
                              totalTripNode.shape,
                          ),
                      )
                      .as(RegionArgumentNode);
            const [rem] = outerBodyBuilder.createOp("Sub", [capturedTotal, baseIdx]);
            [innerTripInput] = outerBodyBuilder.createOp("Min", [tileSizeOuterConst, rem]);
        }

        // 5. Rewire `innerIterNode` in `innerBody` to `global_iter = baseIdx + innerIterNode`
        // (Only needed if the inner body actually uses `iter`, e.g., in Radix mode)
        const iterConsumers = innerIterNode.getOutgoers.targets.filterIs(OperationNode).toArray();
        if (iterConsumers.length > 0) {
            const innerBuilder = new GraphBuilder(innerBody, `tile_inner_${loopOp.id}`);
            const capturedBaseIdx = innerBody
                .addNode(baseIdx.id)
                .init(
                    new RegionArgumentNode.Builder(
                        innerCaptures.length + 1,
                        baseIdx.id,
                        DataType.INT64,
                        [],
                    ),
                )
                .as(RegionArgumentNode);

            const [globalIter] = innerBuilder.createOp("Add", [capturedBaseIdx, innerIterNode]);

            for (const consumer of iterConsumers) {
                const updatedInputs = (consumer.getInputs() ?? []).map((inp) =>
                    inp.id === innerIterNode.id ? globalIter : inp,
                );
                consumer.setInputs(updatedInputs);

                const oldEdge = innerBody.getEdge(innerIterNode.id, consumer.id);
                if (oldEdge) innerBuilder.removeEdge(oldEdge);
                innerBuilder.addEdge(
                    globalIter,
                    consumer,
                    globalIter.literalType,
                    globalIter.shape,
                );
            }
        }

        // 6. Instantiate Inner Loop inside Outer Body
        const innerCondInit = outerBodyBuilder.createConstant(`inner_cond_init`, bool(true));
        const innerLoopInputs: ValueNode[] = [innerTripInput, innerCondInit, ...outerCarryInputs];

        const innerLoopOp = outerBodyBuilder
            .addNode(
                uniq(outerBody, `Inner_${loopOp.id}`),
                new OperationNode.Builder("Loop", innerLoopInputs, {}, [innerBody], {
                    tiled: true,
                }),
            )
            .as(OperationNode);

        for (const inp of innerLoopInputs) {
            if (outerBody.hasNode(inp.id)) {
                outerBodyBuilder.addEdge(inp, innerLoopOp, inp.literalType, inp.shape);
            }
        }

        // Wire Inner Loop outputs to Outer Body formal outputs
        outerCarryInputs.forEach((cIn, idx) => {
            const innerOut = outerBodyBuilder
                .addNode(
                    uniq(outerBody, `inner_loop_out_${idx}`),
                    new TensorNode.Builder(cIn.literalType, cIn.shape, "intermediate"),
                )
                .as(TensorNode);
            outerBodyBuilder.addEdge(innerLoopOp, innerOut, cIn.literalType, cIn.shape);

            const idOut = outerBody
                .addNode(uniq(outerBody, `id_tile_out_${idx}`))
                .init(new OperationNode.Builder("Identity", [innerOut]))
                .as(OperationNode);
            const formalOut = outerBody
                .addNode(uniq(outerBody, `tile_carry_out_${idx}`))
                .init(new TensorNode.Builder(cIn.literalType, cIn.shape, "output"))
                .as(TensorNode);

            outerBody
                .addEdge(innerOut, idOut)
                .init(new OnnxEdge.Builder(cIn.literalType, cIn.shape));
            outerBody
                .addEdge(idOut, formalOut)
                .init(new OnnxEdge.Builder(cIn.literalType, cIn.shape));
        });

        // 7. Replace Original Loop in Root Graph with Tiled Outer Loop
        const outerLoopInputs: ValueNode[] = [outerTripInput, condInitNode, ...initialCarries];
        const tiledOuterLoop = rootBuilder
            .addNode(
                uniq(rootBuilder.graph, `Tiled_${loopOp.id}`),
                new OperationNode.Builder("Loop", outerLoopInputs, {}, [outerBody], {
                    tiled: true,
                }),
            )
            .as(OperationNode);

        for (const inp of outerLoopInputs) {
            if (rootBuilder.graph.hasNode(inp.id)) {
                rootBuilder.addEdge(inp, tiledOuterLoop, inp.literalType, inp.shape);
            }
        }

        // Reconnect original loop output tensors to the new tiled outer loop
        for (const outTensor of origLoopOutputs) {
            const oldEdge = rootBuilder.graph.getEdge(loopOp.id, outTensor.id);
            if (oldEdge) rootBuilder.removeEdge(oldEdge);
            rootBuilder.addEdge(tiledOuterLoop, outTensor, outTensor.literalType, outTensor.shape);
        }

        rootBuilder.removeNode(loopOp);
    }
}
