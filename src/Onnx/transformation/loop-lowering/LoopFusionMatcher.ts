import type OnnxGraph from "../../OnnxGraph.js";
import OperationNode from "../../OperationNode.js";
import TensorNode from "../../TensorNode.js";
import { GraphBuilder } from "../../GraphBuilder.js";
import { DataType, ValueNode, KnownShape, StaticShape, ConcreteValueNode } from "../../OnnxTypes.js";
import { asStaticDims, UNKNOWN_SHAPE, topologicalSortOperationNodes, int64Vec } from "../../Utils.js";
import { buildForLoopRegion } from "./LoopBuilder.js";
import { OpCategory } from "../../Schema/OpSchema.js";
import { OpRegistry } from "../../Schema/OpRegistry.js";
import { TransformationOpportunity } from "../TransformationOpportunity.js";

// Strategy Recipes
import type { LoopLoweringRecipe } from "./LoopLoweringRecipe.js";
import { LowerTransposeRecipe } from "./recipes/LowerTransposeRecipe.js";
import { LowerElementWiseRecipe } from "./recipes/LowerElementWiseRecipe.js";
import { LowerRangeRecipe } from "./recipes/LowerRangeRecipe.js";
import { LowerReductionRecipe } from "./recipes/LowerReductionRecipe.js";
import { LowerMatMulRecipe } from "./recipes/LowerMatMulRecipe.js";
import { LowerConvRecipe } from "./recipes/LowerConvRecipe.js";
import { LowerCoalescedMatMulRecipe } from "./recipes/LowerCoalescedMatMulRecipe.js";

export class LoopFusionMatcher {
    private boundsCache = new Map<string, { totalIters: number | ValueNode; carryShape: KnownShape | ValueNode; targetShape?: KnownShape | ValueNode }>();
    private recipes: LoopLoweringRecipe[];

    constructor(private options: { coalesce: boolean; fuse: boolean }) {
        this.recipes = [
            new LowerElementWiseRecipe(),
            new LowerReductionRecipe(),
            new LowerTransposeRecipe(),
            new LowerRangeRecipe(),
            this.options.coalesce ? new LowerCoalescedMatMulRecipe() : new LowerMatMulRecipe(),
            new LowerConvRecipe(),
        ];
    }

    public findOpportunities(graph: OnnxGraph.Class): TransformationOpportunity[] {
        this.boundsCache.clear();
        const opportunities: TransformationOpportunity[] = [];

        const chains = this.options.fuse
            ? this.findFuseableChains(graph)
            : this.findSingleOpChains(graph);

        for (const chain of chains) {
            const rootOp = chain[chain.length - 1];
            
            opportunities.push(new TransformationOpportunity(
                "LoopLowering",
                chain.map(o => o.id).join(","), // Target is the whole chain
                `Fuse ${chain.length} operations into an ONNX Loop (Root: ${rootOp.type})`,
                (builder: GraphBuilder) => {
                    this.lowerFlatLoopChain(graph, chain, builder);
                }
            ));
        }

        return opportunities;
    }

    private lowerFlatLoopChain(graph: OnnxGraph.Class, chain: OperationNode.Class[], outerBuilder: GraphBuilder): void {
        const rootOp = chain[chain.length - 1];
        const rootOutNodeRaw = rootOp.getOutgoers.targets.filterIs(TensorNode).first()!;
        const originalOutShape: KnownShape = [...(rootOutNodeRaw.shape as KnownShape)];
        const elemTy = rootOutNodeRaw.literalType || DataType.FLOAT;

        const bounds = this.getBoundsFor(rootOp);
        const totalIters = bounds.totalIters;
        const carryShape = bounds.carryShape as StaticShape;

        // 1. Initialize Loop Region
        const { innerBuilder, trip, vInitial, loopOutput, finalize } = buildForLoopRegion(
            outerBuilder, totalIters, elemTy, carryShape, `Loop_${rootOp.id}`, `lowering_${rootOp.id}`
        );

        const body = innerBuilder.graph;
        const axes = innerBuilder.createConstant("axes", int64Vec([0]));
        const valueMap = new Map<string, ValueNode>();
        let finalNextCarry: ValueNode | null = null;

        // 2. Execute the strategies (Recipes) inside the loop body
        for (const op of chain) {
            const recipe = this.getRecipeFor(op)!;
            
            // Safeguard: Ensure target shape isn't a naked array
            let safeTargetShapeNode: ValueNode | undefined;
            if (bounds.targetShape && !Array.isArray(bounds.targetShape)) {
                safeTargetShapeNode = bounds.targetShape as ValueNode;
            }

            const applyRes = recipe.apply(
                op, body, valueMap, trip, axes, originalOutShape, vInitial, safeTargetShapeNode
            );

            let scalarOut: ValueNode;
            // Robust runtime check to prevent array leaks
            if (Array.isArray(applyRes)) {
                scalarOut = applyRes[0] as ValueNode;
            } else if (typeof applyRes === "object" && applyRes !== null && "nextCarry" in applyRes) {
                scalarOut = (applyRes as any).resultNode;
                if (op.id === rootOp.id) finalNextCarry = (applyRes as any).nextCarry;
            } else {
                scalarOut = applyRes as ValueNode;
            }

            const outTensors = op.getOutgoers.targets.filterIs(TensorNode).toArray();
            if (outTensors.length > 0) {
                valueMap.set(outTensors[0].id, scalarOut);
            }
        }

        // 3. Finalize Loop Body result
        let carryOutNode: ConcreteValueNode;
        if (finalNextCarry) {
            carryOutNode = finalNextCarry as ConcreteValueNode;
        } else {
            const finalScalar = valueMap.get(rootOutNodeRaw.id)!;
            const oneShape = innerBuilder.createConstant("one_shape", int64Vec([1]));
            const [iterUnsq] = innerBuilder.createOp("Reshape", [trip, oneShape]);
            const [finalUnsq] = innerBuilder.createOp("Reshape", [finalScalar, oneShape]);
            [carryOutNode] = innerBuilder.createOp("ScatterElements", [vInitial, iterUnsq, finalUnsq], { axis: 0 });
        }
        finalize([carryOutNode]);

        // 4. Post-processing (Outer Graph)
        let processedOut: ValueNode = loopOutput;
        const rootRecipe = this.getRecipeFor(rootOp)!;
        if (rootRecipe.postProcess) {
            const ppRes = rootRecipe.postProcess(rootOp, outerBuilder, loopOutput);
            processedOut = Array.isArray(ppRes) ? ppRes[0] : ppRes; // Safeguard
        }

        // 5. Final Reshape to target shape
        const originalIsDynamic = originalOutShape.includes(-1) || originalOutShape.length === 0;
        let targetShapeNode: ValueNode;

        // Guaranteed to be a ValueNode
        if (bounds.targetShape && !Array.isArray(bounds.targetShape)) {
            targetShapeNode = bounds.targetShape as ValueNode;
        } else {
            let shapeArr: number[];
            if (Array.isArray(bounds.targetShape)) {
                shapeArr = bounds.targetShape as number[];
            } else if (Array.isArray(carryShape)) {
                shapeArr = originalIsDynamic ? (carryShape as number[]) : (originalOutShape as number[]);
            } else {
                targetShapeNode = carryShape as ValueNode; 
                shapeArr = []; // won't be used
            }

            if (!targetShapeNode!) {
                targetShapeNode = outerBuilder.createConstant(`final_shape_${rootOp.id}`, int64Vec(shapeArr));
            }
        }

        const [reshaped] = outerBuilder.createOp("Reshape", [processedOut, targetShapeNode!], {}, [{ type: elemTy, shape: originalOutShape }]);

        // 6. Integrate results back into the main graph
        outerBuilder.replaceAllUsesWith(rootOutNodeRaw, reshaped);
        chain.forEach((op) => op.remove());
    }

    // --- Core Matcher Helpers ---
    private getRecipeFor(op: OperationNode.Class): LoopLoweringRecipe | undefined {
        return this.recipes.find((recipe) => recipe.canApply(op));
    }

    private isSupported(op: OperationNode.Class): boolean {
        return this.getRecipeFor(op) !== undefined;
    }

    private getBoundsFor(op: OperationNode.Class): { totalIters: number | ValueNode; carryShape: KnownShape | ValueNode; targetShape?: KnownShape | ValueNode } {
        if (this.boundsCache.has(op.id)) return this.boundsCache.get(op.id)!;

        const recipe = this.getRecipeFor(op)!;
        const outTensors = op.getOutgoers.targets.filterIs(TensorNode).toArray();
        const originalOutShape: KnownShape = outTensors.length > 0 ? [...(outTensors[0].shape as KnownShape)] : UNKNOWN_SHAPE;
        const safeOut = asStaticDims(originalOutShape);

        if (recipe.getLoopBounds) {
            const result = recipe.getLoopBounds(op, originalOutShape);
            this.boundsCache.set(op.id, result);
            return result;
        }

        const totalIters = safeOut.length === 0 ? 1 : safeOut.reduce((a, b) => a * b, 1);
        const result = { totalIters, carryShape: [totalIters] };
        this.boundsCache.set(op.id, result);
        return result;
    }

    private compareIters(a: number | ValueNode, b: number | ValueNode): boolean {
        if (typeof a === "number" && typeof b === "number") return a === b;
        if (typeof a !== "number" && typeof b !== "number") return a.id === b.id;
        return false;
    }

    private compareCarryShapes(a: KnownShape | ValueNode, b: KnownShape | ValueNode): boolean {
        const isArrayA = Array.isArray(a), isArrayB = Array.isArray(b);
        if (isArrayA && isArrayB) return (a as KnownShape).join(",") === (b as KnownShape).join(",");
        if (!isArrayA && !isArrayB) return (a as ValueNode).id === (b as ValueNode).id;
        return false;
    }

    private findSingleOpChains(graph: OnnxGraph.Class): OperationNode.Class[][] {
        return topologicalSortOperationNodes(graph).filter((op) => this.isSupported(op)).map((op) => [op]);
    }

    private findFuseableChains(graph: OnnxGraph.Class): OperationNode.Class[][] {
        const chains: OperationNode.Class[][] = [];
        const ops = topologicalSortOperationNodes(graph);
        const visited = new Set<string>();
        const registry = OpRegistry.getInstance();

        const roots = ops.filter((op) => {
            if (!this.isSupported(op)) return false;
            const outTensors = op.getOutgoers.targets.filterIs(TensorNode).toArray();
            const goesToGraphOutput = outTensors.some((t) => t.type === "output");

            const consumers = new Set<OperationNode.Class>();
            for (const t of outTensors) {
                t.outgoers.targets.filterIs(OperationNode).forEach((c) => consumers.add(c));
            }

            if (goesToGraphOutput || consumers.size !== 1) return true;

            const consumer = Array.from(consumers)[0];
            if (!this.isSupported(consumer)) return true;

            const consumerSchema = registry.get(consumer.type, 19);
            if (consumerSchema?.category !== OpCategory.ElementWise) return true;

            const prodBounds = this.getBoundsFor(op);
            const consBounds = this.getBoundsFor(consumer);
            if (!this.compareIters(prodBounds.totalIters, consBounds.totalIters) ||
                !this.compareCarryShapes(prodBounds.carryShape, consBounds.carryShape)) {
                return true;
            }

            return false;
        });

        for (const root of roots) {
            if (visited.has(root.id)) continue;

            const rootBounds = this.getBoundsFor(root);
            const cluster: OperationNode.Class[] = [];
            const queue: OperationNode.Class[] = [root];

            while (queue.length > 0) {
                const current = queue.shift()!;
                if (visited.has(current.id)) continue;

                visited.add(current.id);
                cluster.push(current);

                const inTensors = current.getInputs()?.filter((n) => n.is(TensorNode) && n.as(TensorNode).type === "intermediate") || [];

                for (const t of inTensors) {
                    const producers = t.incomers.sources.filterIs(OperationNode).toArray();
                    for (const prod of producers) {
                        if (this.isSupported(prod) && !visited.has(prod.id)) {
                            if (!roots.includes(prod)) {
                                const prodBounds = this.getBoundsFor(prod);
                                if (this.compareIters(prodBounds.totalIters, rootBounds.totalIters) &&
                                    this.compareCarryShapes(prodBounds.carryShape, rootBounds.carryShape)) {
                                    queue.push(prod);
                                }
                            }
                        }
                    }
                }
            }
            chains.push(this.localTopologicalSort(cluster));
        }
        return chains;
    }

    private localTopologicalSort(ops: OperationNode.Class[]): OperationNode.Class[] {
        const visited = new Set<string>();
        const sorted: OperationNode.Class[] = [];

        const visit = (node: OperationNode.Class) => {
            if (visited.has(node.id)) return;
            visited.add(node.id);

            const inTensors = node.getInputs()?.filter((v) => v.is(TensorNode)) || [];
            for (const t of inTensors) {
                const producers = t.incomers.sources.filterIs(OperationNode).toArray();
                for (const prod of producers) {
                    if (ops.some((o) => o.id === prod.id)) visit(prod);
                }
            }
            sorted.push(node);
        };

        for (const op of ops) visit(op);
        return sorted;
    }
}