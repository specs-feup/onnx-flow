import type Graph from "@specs-feup/flow/graph/Graph";
import type { DecompositionOptions } from "@specs-feup/onnx-flow/DecompositionOptions";
import { defaultDecompositionOptions } from "@specs-feup/onnx-flow/DecompositionOptions";
import type OnnxGraph from "../OnnxGraph.js";
import { initializeSchemaRegistry } from "../Schema/index.js";
import { PassManager } from "../PassManager.js";
import { CanonicalizationPass } from "./canonicalization/CanonicalizationPass.js";
import { InferShapesPass } from "./InferShapesPass.js";
import { LoopLoweringPass } from "./loop-lowering/LoopLoweringPass.js";
import transformForCgra from "./cgra-decomposition-example/TransformForCgra.js";

export default class OnnxGraphTransformer implements Graph.Transformation<
    OnnxGraph.Class,
    OnnxGraph.Class
> {
    private canonicalize: boolean;
    private fuse: boolean;
    private recurse: boolean;
    private coalesce: boolean;
    private loopLowering: boolean;
    private decomposeForCgra: boolean;

    // Overload signatures (for TypeScript type checking)
    constructor();
    constructor(options: Partial<DecompositionOptions>);
    constructor(
        canonicalize: boolean,
        fuse: boolean,
        recurse: boolean,
        coalesce: boolean,
        decomposeForCgra: boolean,
    );

    constructor(
        fuseOrOptions: boolean | Partial<DecompositionOptions> = defaultDecompositionOptions.fuse,
        canonicalize: boolean = defaultDecompositionOptions.canonicalize,
        recurse: boolean = defaultDecompositionOptions.recurse,
        coalesce: boolean = defaultDecompositionOptions.coalesce,
        loopLowering: boolean = defaultDecompositionOptions.loopLowering,
        decomposeForCgra: boolean = defaultDecompositionOptions.decomposeForCgra,
    ) {
        if (typeof fuseOrOptions === "boolean") {
            this.canonicalize = canonicalize;
            this.fuse = fuseOrOptions;
            this.recurse = recurse;
            this.coalesce = coalesce;
            this.loopLowering = loopLowering;
            this.decomposeForCgra = decomposeForCgra;
        } else {
            this.canonicalize =
                fuseOrOptions.canonicalize ?? defaultDecompositionOptions.canonicalize;
            this.fuse = fuseOrOptions.fuse ?? defaultDecompositionOptions.fuse;
            this.recurse = fuseOrOptions.recurse ?? defaultDecompositionOptions.recurse;
            this.coalesce = fuseOrOptions.coalesce ?? defaultDecompositionOptions.coalesce;
            this.loopLowering =
                fuseOrOptions.loopLowering ?? defaultDecompositionOptions.loopLowering;
            this.decomposeForCgra =
                fuseOrOptions.decomposeForCgra ?? defaultDecompositionOptions.decomposeForCgra;
        }
    }

    apply(graph: OnnxGraph.Class): OnnxGraph.Class {
        initializeSchemaRegistry();
        const pm = new PassManager();

        if (this.decomposeForCgra) {
            // =========================================================================
            // PHASE 0: CGRA Decomposition Example
            // =========================================================================
            // We run a simple CGRA Decomposition example instead of the other transformations
            pm.addPass({
                name: "TransformForCgraExample",
                run: (g) => {
                    transformForCgra(g);
                    return false; // Return false so the PassManager doesn't loop it infinitely
                },
            });
            pm.run(graph);
            return graph;
        }

        if (this.canonicalize) {
            // =========================================================================
            // PHASE 1: Normalization (Canonicalization)
            // =========================================================================
            // We run this first to break down complex ops (like Gemm -> MatMul + Add)
            // or Softmax into their primitive mathematical components.
            pm.addPass(new CanonicalizationPass());

            // Clean up any dangling nodes left over by the canonicalization rewrites
            //pm.addPass(new DeadCodeEliminationPass());

            // Ensure shapes are strictly inferred before we try to lower anything,
            // as our loop recipes rely heavily on broadcasting shape math.
            pm.addPass(new InferShapesPass());
        }

        if (this.loopLowering) {
            // =========================================================================
            // PHASE 2: Lowering & Early Fusion
            // =========================================================================
            // This looks at our clean, primitive graph and finds chains (Transpose -> Add)
            // and lowers them directly into single, highly optimized ONNX Loops.
            pm.addPass(new LoopLoweringPass({ coalesce: this.coalesce, fuse: this.fuse }));
        }

        // =========================================================================
        // PHASE 3: Final Cleanup
        // =========================================================================
        // The LoopLoweringPass leaves the original, un-looped operations (like the
        // outer Transpose and Add nodes) dangling in the graph. A final DCE sweeps
        // them away, leaving only the beautifully fused Loop!
        //pm.addPass(new DeadCodeEliminationPass());

        pm.addPass(new InferShapesPass());

        // Execute the pipeline!
        pm.run(graph);
        return graph;
    }
}
