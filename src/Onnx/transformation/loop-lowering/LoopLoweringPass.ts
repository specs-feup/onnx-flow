import type OnnxGraph from "../../OnnxGraph.js";
import OperationNode from "../../OperationNode.js";
import TensorNode from "../../TensorNode.js";
import { GraphBuilder } from "../../GraphBuilder.js";
import { DataType } from "../../OnnxTypes.js";
import { buildForLoopRegion } from "./LoopBuilder.js";
import type { ConcreteValueNode, ValueNode, KnownShape, StaticShape } from "../../OnnxTypes.js";
import {
    int64Vec,
    asStaticDims,
    UNKNOWN_SHAPE,
    topologicalSortOperationNodes,
} from "../../Utils.js";
import type { GraphPass } from "../../PassManager.js";

import type { LoopLoweringRecipe } from "./LoopLoweringRecipe.js";
import { LowerTransposeRecipe } from "./recipes/LowerTransposeRecipe.js";
import { LowerElementWiseRecipe } from "./recipes/LowerElementWiseRecipe.js";
import { LowerRangeRecipe } from "./recipes/LowerRangeRecipe.js";
import { LowerReductionRecipe } from "./recipes/LowerReductionRecipe.js";
import { LowerMatMulRecipe } from "./recipes/LowerMatMulRecipe.js";
import { LowerConvRecipe } from "./recipes/LowerConvRecipe.js";
import { LowerCoalescedMatMulRecipe } from "./recipes/LowerCoalescedMatMulRecipe.js";
import { OpCategory } from "../../Schema/OpSchema.js";
import { OpRegistry } from "../../Schema/OpRegistry.js";

export class LoopLoweringPass implements GraphPass {
    public readonly name = "LoopLoweringV2";

    private boundsCache = new Map<
        string,
        { totalIters: number | ValueNode; carryShape: KnownShape | ValueNode }
    >();

    private recipes: LoopLoweringRecipe[];

    constructor(
        private options: { coalesce: boolean; fuse: boolean } = { coalesce: true, fuse: true },
    ) {
        this.recipes = [
            new LowerElementWiseRecipe(),
            new LowerReductionRecipe(),
            new LowerTransposeRecipe(),
            new LowerRangeRecipe(),
            this.options.coalesce ? new LowerCoalescedMatMulRecipe() : new LowerMatMulRecipe(),
            new LowerConvRecipe(),
        ];
    }

    public run(graph: OnnxGraph.Class): boolean {
        this.boundsCache.clear();

        let changed = false;

        const chains = this.options.fuse
            ? this.findFuseableChains(graph)
            : this.findSingleOpChains(graph);

        for (const chain of chains) {
            this.lowerFlatLoopChain(graph, chain);
            changed = true;
        }

        return changed;
    }

    private getRecipeFor(op: OperationNode.Class): LoopLoweringRecipe | undefined {
        return this.recipes.find((recipe) => recipe.canApply(op));
    }

    private isSupported(op: OperationNode.Class): boolean {
        return this.getRecipeFor(op) !== undefined;
    }

    private getBoundsFor(op: OperationNode.Class): {
        totalIters: number | ValueNode;
        carryShape: KnownShape | ValueNode;
        targetShape?: KnownShape | ValueNode;
    } {
        if (this.boundsCache.has(op.id)) {
            return this.boundsCache.get(op.id)!;
        }

        const recipe = this.getRecipeFor(op)!;
        const outTensors = op.getOutgoers.targets.filterIs(TensorNode).toArray();
        const originalOutShape: KnownShape =
            outTensors.length > 0 ? [...(outTensors[0].shape as KnownShape)] : UNKNOWN_SHAPE;

        const safeOut = asStaticDims(originalOutShape);

        /*
        console.log(`\n[Bounds] Evaluating Op: ${op.id} (${op.type})`);
        console.log(`[Bounds]   - originalOutShape: [${originalOutShape}]`);
        console.log(`[Bounds]   - safeOut: [${safeOut}]`);
        */

        if (recipe.getLoopBounds) {
            const result = recipe.getLoopBounds(op, originalOutShape);
            this.boundsCache.set(op.id, result);

            /*
            console.log(
                `[Bounds]   -> Recipe provided bounds. totalIters: ${typeof result.totalIters === "number" ? result.totalIters : "Dynamic"}, carryShape: ${Array.isArray(result.carryShape) ? `[${result.carryShape}]` : "Dynamic"}`,
            );
            */

            return result;
        }

        // --- DYNAMIC FALLBACK ---
        // If the shape has dynamic dimensions (-1) or is completely unknown, build the bounds dynamically
        if (
            (originalOutShape.includes(-1) || originalOutShape.length === 0) &&
            outTensors.length > 0
        ) {
            throw new Error(
                `[LoopLoweringPass] Operation ${op.type} (${op.id}) has a dynamic output shape [${originalOutShape.join(",")}] ` +
                    `but its recipe does not implement getLoopBounds(). Cannot safely lower without creating a cycle.`,
            );
        }

        const totalIters = safeOut.length === 0 ? 1 : safeOut.reduce((a, b) => a * b, 1);

        const result = { totalIters, carryShape: [totalIters] };
        this.boundsCache.set(op.id, result);

        //console.log(`[Bounds]   -> Fallback (Static). totalIters: ${totalIters}`);

        return result;
    }

    private lowerFlatLoopChain(graph: OnnxGraph.Class, chain: OperationNode.Class[]): void {
        const rootOp = chain[chain.length - 1];
        const rootOutNodeRaw = rootOp.getOutgoers.targets.filterIs(TensorNode).first()!;
        const originalOutShape: KnownShape = [...(rootOutNodeRaw.shape as KnownShape)];
        const elemTy = rootOutNodeRaw.literalType || DataType.FLOAT;

        const bounds = this.getBoundsFor(rootOp);
        const totalIters = bounds.totalIters;
        const carryShape = bounds.carryShape as StaticShape;

        /*
        console.log(`\n[LowerChain] Lowering chain: [${chain.map((o) => o.type).join(" -> ")}]`);
        console.log(
            `[LowerChain] Root Op: ${rootOp.type}, final totalIters: ${typeof totalIters === "number" ? totalIters : "Dynamic"}, final carryShape: ${Array.isArray(carryShape) ? `[${carryShape}]` : "Dynamic"}`,
        );
        */

        const outerBuilder = new GraphBuilder(graph, `lowering_${rootOp.id}`);

        // 1. Initialize Loop Region via GraphBuilder helper
        // This handles creating the inner graph, standard inputs (iter, cond, carry),
        // the outer constants, and the Loop node itself with the region attached.
        const { innerBuilder, trip, vInitial, loopOutput, finalize } = buildForLoopRegion(
            outerBuilder,
            totalIters,
            elemTy,
            carryShape,
            `Loop_${rootOp.id}`,
            `lowering_${rootOp.id}`,
        );

        const body = innerBuilder.graph;
        const axes = innerBuilder.createConstant("axes", int64Vec([0]));

        const valueMap = new Map<string, ValueNode>();
        let finalNextCarry: ValueNode | null = null;

        // 2. Build Loop Body by applying recipes in sequence
        for (const op of chain) {
            const recipe = this.getRecipeFor(op)!;
            // iter is 'trip', carry is 'vInitial'
            const applyRes = recipe.apply(
                op,
                body,
                valueMap,
                trip,
                axes,
                originalOutShape,
                vInitial,
                bounds.targetShape as ValueNode | undefined,
            );

            let scalarOut: ValueNode;
            if (typeof applyRes === "object" && "nextCarry" in applyRes) {
                scalarOut = applyRes.resultNode;
                // Only the root of the chain should dictate the final loop-carried update
                if (op.id === rootOp.id) finalNextCarry = applyRes.nextCarry;
            } else {
                scalarOut = applyRes as ValueNode;
            }

            const outTensors = op.getOutgoers.targets.filterIs(TensorNode).toArray();
            if (outTensors.length > 0) {
                valueMap.set(outTensors[0].id, scalarOut);
            }
        }

        // 3. Finalize Loop Body result (Scatter current scalar result into the carried tensor)
        let carryOutNode: ConcreteValueNode;
        if (finalNextCarry) {
            carryOutNode = finalNextCarry as ConcreteValueNode;
        } else {
            const finalScalar = valueMap.get(rootOutNodeRaw.id)!;

            const oneShape = innerBuilder.createConstant("one_shape", int64Vec([1]));
            const [iterUnsq] = innerBuilder.createOp("Reshape", [trip, oneShape]);
            const [finalUnsq] = innerBuilder.createOp("Reshape", [finalScalar, oneShape]);

            [carryOutNode] = innerBuilder.createOp(
                "ScatterElements",
                [vInitial, iterUnsq, finalUnsq],
                { axis: 0 },
            );
        }

        // Connect the final computed state to the loop output boundary
        finalize([carryOutNode]);

        // 4. Post-processing (Outer Graph)
        // This handles cases like ReduceMean (divide by count) or ReduceL2 (sqrt) after the loop.
        let processedOut: ValueNode = loopOutput;
        const rootRecipe = this.getRecipeFor(rootOp)!;
        if (rootRecipe.postProcess) {
            processedOut = rootRecipe.postProcess(rootOp, outerBuilder, loopOutput);
        }

        // 5. Final Reshape to target shape
        const originalIsDynamic = originalOutShape.includes(-1) || originalOutShape.length === 0;
        const carryIsStatic = Array.isArray(carryShape);

        let targetShapeNode: ValueNode;

        if (bounds.targetShape) {
            // Use the explicit N-D target shape if the recipe or fallback provided it
            targetShapeNode = bounds.targetShape as ValueNode;
        } else if (carryIsStatic) {
            // Static Carry Shape Logic
            const staticCarryShape = carryShape as number[];
            const expectedSize = originalIsDynamic
                ? -1
                : originalOutShape.reduce((a, b) => (a as number) * (b as number), 1);
            const actualSize = staticCarryShape.reduce((a, b) => a * b, 1);

            let safeFinalShape = originalOutShape as number[];

            if (!originalIsDynamic && expectedSize !== actualSize) {
                console.warn(
                    `[LoopLoweringPass] WARNING: Shape mismatch for node ${rootOp.id}. ` +
                        `Original model expects shape [${originalOutShape.join(",")}] (${expectedSize} elements), ` +
                        `but loop lowered to ${actualSize} elements. Falling back to the inferred carry shape.`,
                );
                safeFinalShape = staticCarryShape;
            }

            targetShapeNode = outerBuilder.createConstant(`final_shape`, int64Vec(safeFinalShape));
        } else {
            // Dynamic Carry Shape Logic
            // The carryShape is already a ValueNode containing the dynamic 1D shape tensor
            if (!originalIsDynamic) {
                console.warn(
                    `[LoopLoweringPass] WARNING: Original shape for ${rootOp.id} is static [${originalOutShape.join(",")}], ` +
                        `but the loop lowered to a dynamic shape. Falling back to the dynamic inferred shape.`,
                );
            }
            targetShapeNode = carryShape as ValueNode;
        }

        const expectedShapeOut = [{ type: elemTy, shape: originalOutShape as KnownShape }];
        const [reshaped] = outerBuilder.createOp(
            "Reshape",
            [processedOut, targetShapeNode],
            {},
            expectedShapeOut,
        );

        // 6. Integrate results back into the main graph
        outerBuilder.replaceAllUsesWith(rootOutNodeRaw, reshaped);
        chain.forEach((op) => op.remove());
    }

    private compareIters(a: number | ValueNode, b: number | ValueNode): boolean {
        if (typeof a === "number" && typeof b === "number") {
            return a === b;
        }
        if (typeof a !== "number" && typeof b !== "number") {
            // Both are ValueNodes; compare by ID
            return a.id === b.id;
        }
        return false; // One is static, one is dynamic
    }

    private compareCarryShapes(a: KnownShape | ValueNode, b: KnownShape | ValueNode): boolean {
        const isArrayA = Array.isArray(a);
        const isArrayB = Array.isArray(b);

        if (isArrayA && isArrayB) {
            // Both are static KnownShapes (arrays); use the string comparison
            return (a as KnownShape).join(",") === (b as KnownShape).join(",");
        }
        if (!isArrayA && !isArrayB) {
            // Both are ValueNodes; compare by ID
            return (a as ValueNode).id === (b as ValueNode).id;
        }
        return false; // One is static, one is dynamic
    }

    private findSingleOpChains(graph: OnnxGraph.Class): OperationNode.Class[][] {
        return topologicalSortOperationNodes(graph)
            .filter((op) => this.isSupported(op))
            .map((op) => [op]);
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

            // 1. STRICT FUSION RULE:
            // A node MUST be a root (cannot be fused) if it goes to a graph output,
            // or if it has multiple consumers (it must materialize its output for all of them).
            if (goesToGraphOutput || consumers.size !== 1) {
                return true;
            }

            // It has exactly 1 consumer. We can only fuse if bounds match.
            const consumer = Array.from(consumers)[0];
            if (!this.isSupported(consumer)) return true;

            // ===================================================================
            // NEW SCHEMA CHECK:
            // We can only safely fuse into scalar loops if the consumer is
            // strictly an ElementWise operation (e.g. Add, Mul, Relu).
            // DataMovement operations (like Transpose) require full tensor access.
            // ===================================================================
            const consumerSchema = registry.get(consumer.type, 19);
            if (consumerSchema?.category !== OpCategory.ElementWise) {
                return true;
            }

            const prodBounds = this.getBoundsFor(op);
            const consBounds = this.getBoundsFor(consumer);
            if (
                !this.compareIters(prodBounds.totalIters, consBounds.totalIters) ||
                !this.compareCarryShapes(prodBounds.carryShape, consBounds.carryShape)
            ) {
                return true; // Different loop iteration spaces
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

                const inTensors =
                    current
                        .getInputs()
                        ?.filter(
                            (n) => n.is(TensorNode) && n.as(TensorNode).type === "intermediate",
                        ) || [];

                for (const t of inTensors) {
                    const producers = t.incomers.sources.filterIs(OperationNode).toArray();
                    for (const prod of producers) {
                        if (this.isSupported(prod) && !visited.has(prod.id)) {
                            // 2. ONLY absorb the producer if it is NOT a root itself!
                            if (!roots.includes(prod)) {
                                const prodBounds = this.getBoundsFor(prod);
                                if (
                                    this.compareIters(
                                        prodBounds.totalIters,
                                        rootBounds.totalIters,
                                    ) &&
                                    this.compareCarryShapes(
                                        prodBounds.carryShape,
                                        rootBounds.carryShape,
                                    )
                                ) {
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
