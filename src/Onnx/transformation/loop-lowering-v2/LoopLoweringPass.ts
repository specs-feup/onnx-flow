import OnnxGraph from "../../OnnxGraph.js";
import OperationNode from "../../OperationNode.js";
import TensorNode from "../../TensorNode.js";
import { GraphBuilder } from "../../GraphBuilder.js";
import { DataType } from "../../OnnxTypes.js";
import type { ConcreteValueNode, ValueNode, KnownShape, StaticShape } from "../../OnnxTypes.js";
import { int64Vec, asStaticDims, UNKOWN_SHAPE } from "../../Utils.js";
import type { GraphPass } from "../../PassManager.js";

import type { LoopLoweringRecipe } from "./LoopLoweringRecipe.js";
import { LowerTransposeRecipe } from "./recipes/LowerTransposeRecipe.js";
import { LowerElementWiseRecipe } from "./recipes/LowerElementWiseRecipe.js";
import { LowerRangeRecipe } from "./recipes/LowerRangeRecipe.js";
import { LowerReductionRecipe } from "./recipes/LowerReductionRecipe.js";
import { LowerMatMulRecipe } from "./recipes/LowerMatMulRecipe.js";
import { LowerConvRecipe } from "./recipes/LowerConvRecipe.js";
import { LowerCoalescedMatMulRecipe } from "./recipes/LowerCoalescedMatMulRecipe.js";

export class LoopLoweringPass implements GraphPass {
    public readonly name = "LoopLoweringV2";

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

    private getBoundsFor(op: OperationNode.Class): { totalIters: number; carryShape: KnownShape } {
        const recipe = this.getRecipeFor(op)!;
        const outTensors = op.getOutgoers.targets.filterIs(TensorNode).toArray();
        const originalOutShape: KnownShape =
            outTensors.length > 0 && outTensors[0].shape !== undefined
                ? [...(outTensors[0].shape as KnownShape)]
                : UNKOWN_SHAPE;

        const safeOut = asStaticDims(originalOutShape);

        if (recipe.getLoopBounds) {
            return recipe.getLoopBounds(op, safeOut);
        }

        const totalIters = safeOut.length === 0 ? 1 : safeOut.reduce((a, b) => a * b, 1);
        return { totalIters, carryShape: [totalIters] };
    }

    private lowerFlatLoopChain(graph: OnnxGraph.Class, chain: OperationNode.Class[]): void {
        const rootOp = chain[chain.length - 1];
        const rootOutNodeRaw = rootOp.getOutgoers.targets.filterIs(TensorNode).first()!;
        const originalOutShape: KnownShape =
            rootOutNodeRaw.shape === undefined
                ? UNKOWN_SHAPE
                : [...(rootOutNodeRaw.shape as KnownShape)];
        const elemTy = rootOutNodeRaw.literalType || DataType.FLOAT;

        const bounds = this.getBoundsFor(rootOp);
        const totalIters = bounds.totalIters;
        const carryShape = bounds.carryShape as StaticShape;

        const outerBuilder = new GraphBuilder(graph, `lowering_${rootOp.id}`);

        // 1. Initialize Loop Region via GraphBuilder helper
        // This handles creating the inner graph, standard inputs (iter, cond, carry),
        // the outer constants, and the Loop node itself with the region attached.
        const { innerBuilder, trip, vInitial, loopOutput, finalize } =
            outerBuilder.createLoopRegion(
                outerBuilder,
                totalIters,
                elemTy,
                carryShape,
                `Loop_${rootOp.id}`,
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
            );

            let scalarOut: ValueNode;
            if (applyRes && typeof applyRes === "object" && "nextCarry" in applyRes) {
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

            const [iterUnsq] = innerBuilder.createOp("Unsqueeze", [trip, axes]);
            const [finalUnsq] = innerBuilder.createOp("Unsqueeze", [finalScalar, axes]);

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
        const shapeConst = outerBuilder.createConstant(
            `final_shape`,
            int64Vec(originalOutShape as number[]),
        );
        const [reshaped] = outerBuilder.createOp("Reshape", [processedOut, shapeConst]);

        // 6. Integrate results back into the main graph
        outerBuilder.replaceAllUsesWith(rootOutNodeRaw, reshaped);
        chain.forEach((op) => op.remove());
    }

    private findSingleOpChains(graph: OnnxGraph.Class): OperationNode.Class[][] {
        return graph
            .getOperationNodes()
            .toArray()
            .filter((op) => this.isSupported(op))
            .map((op) => [op]);
    }

    private findFuseableChains(graph: OnnxGraph.Class): OperationNode.Class[][] {
        const chains: OperationNode.Class[][] = [];
        const ops = graph.getOperationNodes().toArray();
        const visited = new Set<string>();

        const roots = ops.filter((op) => {
            if (!this.isSupported(op)) return false;
            const outTensors = op.getOutgoers.targets.filterIs(TensorNode).toArray();
            const goesToGraphOutput = outTensors.some((t) => t.type === "output");
            const consumedByUnsupported = outTensors.some((t) =>
                t.outgoers.targets
                    .filterIs(OperationNode)
                    .some((consumer) => !this.isSupported(consumer)),
            );
            return goesToGraphOutput || consumedByUnsupported;
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
                        if (this.isSupported(prod)) {
                            const prodBounds = this.getBoundsFor(prod);
                            if (
                                prodBounds.totalIters === rootBounds.totalIters &&
                                prodBounds.carryShape.join(",") === rootBounds.carryShape.join(",")
                            ) {
                                queue.push(prod);
                            }
                        }
                    }
                }
            }

            chains.push(this.topologicalSort(cluster));
        }

        return chains;
    }

    private topologicalSort(ops: OperationNode.Class[]): OperationNode.Class[] {
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
