import OnnxGraph from "../../OnnxGraph.js";
import OperationNode from "../../OperationNode.js";
import TensorNode from "../../TensorNode.js";
import ConstantNode from "../../ConstantNode.js";
import OnnxEdge from "../../OnnxEdge.js";
import { GraphBuilder } from "../../GraphBuilder.js";
import { DataType } from "../../OnnxTypes.js";
import type { ConcreteValueNode, ValueNode, KnownShape } from "../../OnnxTypes.js";
import { uniq, scalarInt64, bool, zeroTensor, int64Vec, asStaticDims, UNKOWN_SHAPE } from "../../Utils.js";
import type { GraphPass } from "../../PassManager.js";
import Graph from "@specs-feup/flow/graph/Graph";

import type { LoopLoweringRecipe } from "./LoopLoweringRecipe.js";
import { LowerTransposeRecipe } from "./recipes/LowerTransposeRecipe.js";
import { LowerElementWiseRecipe } from "./recipes/LowerElementWiseRecipe.js";
import { unsqueezeIdx } from "../loop-lowering/BuildLoop.js";

export class LoopLoweringPass implements GraphPass {
    public readonly name = "LoopLoweringV2";
    
    private registry: Map<string, LoopLoweringRecipe> = new Map([
        ["Transpose", new LowerTransposeRecipe()],
        ["Add", new LowerElementWiseRecipe()],
        ["Mul", new LowerElementWiseRecipe()],
        ["Relu", new LowerElementWiseRecipe()],
    ]);

    constructor(private options: { coalesce: boolean, fuse: boolean } = { coalesce: true, fuse: true }) {}

    public run(graph: OnnxGraph.Class): boolean {
        let changed = false;

        const chains = this.options.fuse 
            ? this.findFuseableChains(graph) 
            : this.findSingleOpChains(graph);

        for (const chain of chains) {
            const hasMatMul = chain.some(op => op.type === "MatMul");

            if (hasMatMul && !this.options.coalesce) {
                // this.lowerNestedMatMulChain(graph, chain);
            } else {
                this.lowerFlatLoopChain(graph, chain);
            }
            changed = true;
        }

        return changed;
    }

    // =========================================================================
    // NEW: Safe operation checker using the Recipe's canApply() method!
    // =========================================================================
    private isSupported(op: OperationNode.Class): boolean {
        const recipe = this.registry.get(op.type);
        if (!recipe) return false;          // No recipe exists for this OpType
        return recipe.canApply(op);         // Recipe exists, but can it handle this specific node?
    }

    private lowerFlatLoopChain(graph: OnnxGraph.Class, chain: OperationNode.Class[]): void {
        const rootOp = chain[chain.length - 1];
        
        const rootOutNodeRaw = rootOp.getOutgoers.targets.filterIs(TensorNode).first()!;
        const originalOutShape: KnownShape = rootOutNodeRaw.shape === undefined ? UNKOWN_SHAPE : [...rootOutNodeRaw.shape as KnownShape];
        const elemTy = rootOutNodeRaw.literalType || DataType.FLOAT;

        const safeOut = asStaticDims(originalOutShape);
        const totalIters = safeOut.length <= 1 ? (safeOut[0] ?? 1) : safeOut.reduce((a, b) => a * b, 1);
        
        const tripConst = this.createConstant(graph, `trip_${rootOp.id}`, scalarInt64(totalIters));
        const condConst = this.createConstant(graph, `cond_${rootOp.id}`, bool(true));
        const initCarry = this.createConstant(graph, `init_carry_${rootOp.id}`, zeroTensor(elemTy, [totalIters]));

        const body = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const iter = body.addNode(uniq(body, "iter")).init(new TensorNode.Builder(DataType.INT64, [], "input")).as(TensorNode);
        const condIn = body.addNode(uniq(body, "cond_in")).init(new TensorNode.Builder(DataType.BOOL, [], "input")).as(TensorNode);
        const carry = body.addNode(uniq(body, "carry")).init(new TensorNode.Builder(elemTy, [totalIters], "input")).as(TensorNode);
        const axes = this.createConstant(body, "axes", int64Vec([0]));

        // =====================================================================
        // ORT FIX: Create cond_out immediately so it serializes as output index 0
        // =====================================================================
        const idCond = body.addNode(uniq(body, "id_cond")).init(new OperationNode.Builder("Identity", [condIn])).as(OperationNode);
        const condOut = body.addNode(uniq(body, "cond_out")).init(new TensorNode.Builder(DataType.BOOL, [], "output")).as(TensorNode);
        body.addEdge(condIn, idCond).init(new OnnxEdge.Builder(condIn.literalType, condIn.shape));
        body.addEdge(idCond, condOut).init(new OnnxEdge.Builder(condOut.literalType, condOut.shape));

        const valueMap = new Map<string, ValueNode>();
        let isLastKNode: ConcreteValueNode | null = null; 

        for (const op of chain) {
            const recipe = this.registry.get(op.type)!; 

            if (!recipe.canApply(op)) {
                throw new Error(`[LoopLoweringV2] Recipe rejected ${op.type} during lowering phase!`);
            }

            const scalarOut = recipe.apply(op, body, valueMap, iter, axes, originalOutShape);
            
            // =================================================================
            // DEFORESTATION FIX: Map the TENSOR ID, not the Operation ID!
            // =================================================================
            const outTensors = op.getOutgoers.targets.filterIs(TensorNode).toArray();
            if (outTensors.length > 0) {
                valueMap.set(outTensors[0].id, scalarOut);
            }
        }

        const finalOp = chain[chain.length - 1];
        let finalOutTensor = finalOp.getOutgoers.targets.filterIs(TensorNode).first()!;
        let finalScalar = valueMap.get(finalOutTensor.id)!;

        if (isLastKNode) {
            // Find the core MatMul scalar via its output tensor
            const matmulOp = chain.find(o => o.type === "MatMul")!;
            const matmulOutTensor = matmulOp.getOutgoers.targets.filterIs(TensorNode).first()!;
            const matmulScalar = valueMap.get(matmulOutTensor.id)!;

            const gateNode = body.addNode(uniq(body, "epilogue_gate")).init(
                new OperationNode.Builder("Where", [isLastKNode, finalScalar, matmulScalar])
            ).as(OperationNode);
            
            const gatedScalar = body.addNode(uniq(body, "gated_scalar")).init(
                new TensorNode.Builder(elemTy, [], "intermediate")
            ).as(TensorNode);
            
            body.addEdge(gateNode, gatedScalar).init(new OnnxEdge.Builder(elemTy, [])).as(OnnxEdge);
            finalScalar = gatedScalar;
        }

        const finalUnsq = unsqueezeIdx(body, finalScalar as ConcreteValueNode, axes, "final_unsq");
        const iterUnsq = unsqueezeIdx(body, iter, axes, "iter_unsq");

        const scatter = body.addNode(uniq(body, "scatter")).init(
            new OperationNode.Builder("ScatterElements", [carry, iterUnsq, finalUnsq], { axis: 0 })
        ).as(OperationNode);

        const carryOut = body.addNode(uniq(body, "carry_out")).init(
            new TensorNode.Builder(elemTy, [totalIters], "output")
        ).as(TensorNode);
        
        body.addEdge(carry, scatter).init(new OnnxEdge.Builder(carry.literalType, carry.shape));
        body.addEdge(iterUnsq, scatter).init(new OnnxEdge.Builder(iterUnsq.literalType, iterUnsq.shape));
        body.addEdge(finalUnsq, scatter).init(new OnnxEdge.Builder(finalUnsq.literalType, finalUnsq.shape));
        body.addEdge(scatter, carryOut).init(new OnnxEdge.Builder(carryOut.literalType, carryOut.shape));

        const loopInputs = [tripConst, condConst, initCarry];
        const loop = graph.addNode(uniq(graph, `Loop_${rootOp.id}`)).init(
            new OperationNode.Builder("Loop", loopInputs, {}, [body])
        ).as(OperationNode);

        loopInputs.forEach(inp => graph.addEdge(inp, loop).init(new OnnxEdge.Builder(inp.literalType, inp.shape)).as(OnnxEdge));

        const loopOut = graph.addNode(uniq(graph, `loop_out_${rootOp.id}`)).init(
            new TensorNode.Builder(elemTy, [totalIters], "intermediate")
        ).as(TensorNode);
        graph.addEdge(loop, loopOut).init(new OnnxEdge.Builder(elemTy, [totalIters])).as(OnnxEdge);

        const shapeConst = this.createConstant(graph, `shape_${rootOp.id}`, int64Vec(originalOutShape as number[]));
        const reshape = graph.addNode(uniq(graph, `reshape_out_${rootOp.id}`)).init(
            new OperationNode.Builder("Reshape", [loopOut, shapeConst])
        ).as(OperationNode);

        finalOutTensor = graph.addNode(uniq(graph, `final_out_${rootOp.id}`)).init(
            new TensorNode.Builder(elemTy, originalOutShape, rootOutNodeRaw.type)
        ).as(TensorNode);
        
        graph.addEdge(loopOut, reshape).init(new OnnxEdge.Builder(elemTy, [totalIters])).as(OnnxEdge);
        graph.addEdge(shapeConst, reshape).init(new OnnxEdge.Builder(shapeConst.literalType, shapeConst.shape)).as(OnnxEdge);
        graph.addEdge(reshape, finalOutTensor).init(new OnnxEdge.Builder(elemTy, originalOutShape)).as(OnnxEdge);

        const builder = new GraphBuilder(graph);
        builder.replaceAllUsesWith(rootOutNodeRaw, finalOutTensor);
        chain.forEach(op => op.remove());
    }

    // =========================================================================
    // NEW: The "No Fusion" chain finder
    // =========================================================================
    private findSingleOpChains(graph: OnnxGraph.Class): OperationNode.Class[][] {
        const chains: OperationNode.Class[][] = [];
        const ops = graph.getOperationNodes().toArray();
        
        for (const op of ops) {
            // If the recipe supports it, put it in its own isolated chain
            if (this.isSupported(op)) {
                chains.push([op]);
            }
        }
        return chains;
    }

    // =========================================================================
    // UPDATED: Chain finder now verifies using `this.isSupported(op)`
    // =========================================================================
    private findFuseableChains(graph: OnnxGraph.Class): OperationNode.Class[][] {
        const chains: OperationNode.Class[][] = [];
        const ops = graph.getOperationNodes().toArray();
        const visited = new Set<string>();

        // 1. Find "roots" (ops whose outputs leave the fused block)
        const roots = ops.filter(op => {
            if (!this.isSupported(op)) return false;
            
            const outTensors = op.getOutgoers.targets.filterIs(TensorNode).toArray();
            
            // It's a root if it outputs directly to the graph output
            const goesToGraphOutput = outTensors.some(t => t.type === "output");
            
            // Or if it's consumed by an op we don't support (breaking the chain)
            const consumedByUnsupported = outTensors.some(t => 
                t.outgoers.targets.filterIs(OperationNode).some(consumer => !this.isSupported(consumer))
            );

            return goesToGraphOutput || consumedByUnsupported;
        });

        // 2. For each root, walk backwards to collect the whole fused cluster
        for (const root of roots) {
            if (visited.has(root.id)) continue;

            const cluster: OperationNode.Class[] = [];
            const queue: OperationNode.Class[] = [root];

            while (queue.length > 0) {
                const current = queue.shift()!;
                if (visited.has(current.id)) continue;
                
                visited.add(current.id);
                cluster.push(current);

                // Walk up to producers (only through intermediate tensors)
                const inTensors = current.getInputs()?.filter(n => n.is(TensorNode) && n.as(TensorNode).type === "intermediate") || [];
                for (const t of inTensors) {
                    const producers = t.incomers.sources.filterIs(OperationNode).toArray();
                    for (const prod of producers) {
                        if (this.isSupported(prod)) {
                            queue.push(prod);
                        }
                    }
                }
            }

            // 3. Topologically sort the cluster (producers first)
            const sortedCluster = this.topologicalSort(cluster);
            chains.push(sortedCluster);
        }

        return chains;
    }

    private topologicalSort(ops: OperationNode.Class[]): OperationNode.Class[] {
        const visited = new Set<string>();
        const sorted: OperationNode.Class[] = [];

        const visit = (node: OperationNode.Class) => {
            if (visited.has(node.id)) return;
            visited.add(node.id);

            // Find all operations that produce inputs for this node
            const inTensors = node.getInputs()?.filter(v => v.is(TensorNode)) || [];
            for (const t of inTensors) {
                const producers = t.incomers.sources.filterIs(OperationNode).toArray();
                for (const prod of producers) {
                    // FIX: Compare by ID rather than object reference!
                    if (ops.some(o => o.id === prod.id)) visit(prod); 
                }
            }
            sorted.push(node);
        };

        for (const op of ops) visit(op);
        return sorted;
    }

    private createConstant(graph: OnnxGraph.Class, id: string, proto: any): ConstantNode.Class {
        return graph.addNode(uniq(graph, id)).init(new ConstantNode.Builder(proto)).as(ConstantNode);
    }
}