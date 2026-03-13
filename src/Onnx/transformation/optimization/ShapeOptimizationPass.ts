import type OnnxGraph from "../../OnnxGraph.js";
import type { GraphPass } from "../../PassManager.js";
import type OperationNode from "../../OperationNode.js";
import { GraphBuilder } from "../../GraphBuilder.js";
import { topologicalSortOperationNodes, toStaticShape, makeTensorProto } from "../../Utils.js";
import { DataType } from "../../OnnxTypes.js";

export class ShapeOptimizationPass implements GraphPass {
    public readonly name = "ShapeOptimization";

    run(graph: OnnxGraph.Class): boolean {
        let globalChanged = false;
        let localChanged = true;

        // Run until we reach a fixed point (optimizing one shape might unlock another)
        while (localChanged) {
            localChanged = false;
            const ops = topologicalSortOperationNodes(graph);

            for (const node of ops) {
                if (!graph.hasNode(node.id)) continue;

                // 1. Fold Static Shapes
                if (node.type === "Shape" && this.foldStaticShape(node, graph)) {
                    localChanged = true;
                    continue; // Node was removed, move to next
                }

                // 2. Eliminate Degenerate Ops (e.g., Concat of 1)
                if (node.type === "Concat" && this.eliminateDegenerateConcat(node, graph)) {
                    localChanged = true;
                    continue;
                }

                // 3. Scalarize 1x1 Math Operations (Legacy Port)
                if (["Add", "Sub", "Mul", "Div"].includes(node.type)) {
                    if (this.optimizeScalarMath(node, graph)) {
                        localChanged = true;
                        continue;
                    }
                }

                // 4. MatMul Outer Product Optimization
                if (node.type === "MatMul") {
                    if (this.optimizeOuterProductMatMul(node, graph)) {
                        localChanged = true;
                        continue;
                    }
                }
            }
            if (localChanged) globalChanged = true;
        }

        return globalChanged;
    }

    /**
     * Replaces `Shape(X)` with a Constant if `X` has statically known dimensions.
     */
    private foldStaticShape(node: OperationNode.Class, graph: OnnxGraph.Class): boolean {
        const inputs = node.getInputs();
        if (!inputs || inputs.length === 0) return false;

        const target = inputs[0];
        const shape = toStaticShape(target.shape);

        // If the shape is fully known (no -1s) and not a scalar/unknown (length > 0)
        if (shape.length > 0 && !shape.includes(-1)) {
            const builder = new GraphBuilder(graph, `opt_shape_${node.id}`);

            const shapeConst = builder.createConstant(
                `static_shape_${target.id}`,
                makeTensorProto(DataType.INT64, [shape.length], shape),
            );

            builder.replaceAllUsesWith(node.getOutputs()[0], shapeConst);
            node.remove();
            return true;
        }
        return false;
    }

    /**
     * A Concat with only 1 input is useless. Replace it with Identity.
     */
    private eliminateDegenerateConcat(node: OperationNode.Class, graph: OnnxGraph.Class): boolean {
        const inputs = node.getInputs();
        if (inputs && inputs.length === 1) {
            const builder = new GraphBuilder(graph, `opt_concat_${node.id}`);

            // Just pass the single input straight through
            const [identity] = builder.createOp("Identity", [inputs[0]]);

            builder.replaceAllUsesWith(node.getOutputs()[0], identity);
            node.remove();
            return true;
        }
        return false;
    }

    /**
     * Ports the legacy 1x1 shape optimization to standard ONNX.
     * Replaces Add( [1,1], [1,1] ) with Gather(0) -> Math -> Reshape([1,1])
     */
    private optimizeScalarMath(node: OperationNode.Class, graph: OnnxGraph.Class): boolean {
        const inputs = node.getInputs();
        if (!inputs || inputs.length !== 2) return false;

        const A = inputs[0];
        const B = inputs[1];

        const shapeA = toStaticShape(A.shape);
        const shapeB = toStaticShape(B.shape);

        // Check if both are exactly [1, 1]
        const isOneByOne = (s: number[]) => s.length === 2 && s[0] === 1 && s[1] === 1;

        if (isOneByOne(shapeA) && isOneByOne(shapeB)) {
            const builder = new GraphBuilder(graph, `opt_scalar_${node.id}`);

            // Equivalent to legacy "zero_offset"
            const zeroIdx = builder.createConstant(
                `zero_idx_${node.id}`,
                makeTensorProto(DataType.INT64, [1], [0]),
            );
            const flatShape = builder.createConstant(
                `flat_${node.id}`,
                makeTensorProto(DataType.INT64, [1], [-1]),
            );

            // Equivalent to Legacy "Load": Flatten to 1D, then Gather index 0 to make it a 0D scalar
            const [flatA] = builder.createOp("Reshape", [A, flatShape]);
            const [loadA] = builder.createOp("Gather", [flatA, zeroIdx], { axis: 0 });

            const [flatB] = builder.createOp("Reshape", [B, flatShape]);
            const [loadB] = builder.createOp("Gather", [flatB, zeroIdx], { axis: 0 });

            // Squeeze to pure 0D scalars
            const zeroAxis = builder.createConstant(
                `ax_${node.id}`,
                makeTensorProto(DataType.INT64, [1], [0]),
            );
            const [scalarA] = builder.createOp("Squeeze", [loadA, zeroAxis]);
            const [scalarB] = builder.createOp("Squeeze", [loadB, zeroAxis]);

            // Perform Math on scalars
            const [mathOut] = builder.createOp(node.type, [scalarA, scalarB]);

            // Equivalent to Legacy "Store": Reshape the 0D result back to [1, 1] for the rest of the graph
            const outShape = builder.createConstant(
                `out_sh_${node.id}`,
                makeTensorProto(DataType.INT64, [2], [1, 1]),
            );
            const [storeOut] = builder.createOp("Reshape", [mathOut, outShape]);

            builder.replaceAllUsesWith(node.getOutputs()[0], storeOut);
            node.remove();
            return true;
        }

        return false;
    }

    /**
     * Replaces an Outer Product MatMul with a broadcasted element-wise Mul.
     * Matches: MatMul( [M, 1], [1, N] ) -> Mul( [M, 1], [1, N] )
     */
    private optimizeOuterProductMatMul(node: OperationNode.Class, graph: OnnxGraph.Class): boolean {
        const inputs = node.getInputs();
        if (!inputs || inputs.length !== 2) return false;

        const A = inputs[0];
        const B = inputs[1];

        const shapeA = toStaticShape(A.shape);
        const shapeB = toStaticShape(B.shape);

        // For simplicity in the base version, we only match explicit 2D tensors.
        // (ONNX MatMul has complex broadcasting for 3D/4D and 1D tensors)
        if (shapeA.length === 2 && shapeB.length === 2) {
            const K_A = shapeA[1]; // Inner dimension of A
            const K_B = shapeB[0]; // Inner dimension of B

            // If the shared inner dimension is exactly 1, it's an outer product!
            if (K_A === 1 && K_B === 1) {
                const builder = new GraphBuilder(graph, `opt_outer_prod_${node.id}`);

                // Mul will automatically broadcast [M, 1] and [1, N] to [M, N]
                const [mulOut] = builder.createOp("Mul", [A, B]);

                builder.replaceAllUsesWith(node.getOutputs()[0], mulOut);
                node.remove();

                return true;
            }
        }

        return false;
    }
}
