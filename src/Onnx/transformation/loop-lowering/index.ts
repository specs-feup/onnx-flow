import type Graph from "@specs-feup/flow/graph/Graph";
import type { DecompositionOptions } from "@specs-feup/onnx-flow/DecompositionOptions";
import { defaultDecompositionOptions } from "@specs-feup/onnx-flow/DecompositionOptions";
import type OnnxGraph from "../../OnnxGraph.js";
import { PassManager } from "../../PassManager.js";
import { OrchestratorPass } from "../OrchestratorPass.js";
import { MatMulGridDecompositionRecipe } from "../cgra-decomposition/MatMul.js";
import { AddGridDecompositionRecipe } from "../cgra-decomposition/Add.js";
import { ReluGridDecompositionRecipe } from "../cgra-decomposition/Relu.js";
import { LowerConvRecipe } from "./recipes/LowerConvRecipe.js";
import { LowerTransposeRecipe } from "./recipes/LowerTransposeRecipe.js";
import { LowerRangeRecipe } from "./recipes/LowerRangeRecipe.js";
import { LowerElementWiseRecipe } from "./recipes/LowerElementWiseRecipe.js";
import { LowerReductionRecipe } from "./recipes/LowerReductionRecipe.js";
import { InferShapesPass } from "../InferShapesPass.js";
import { initializeSchemaRegistry } from "../../Schema/index.js";
import { LowerClipRecipe } from "../canonicalization/recipes/LowerClipRecipe.js";
import { LowerGemmRecipe } from "../canonicalization/recipes/LowerGemmRecipe.js";
import { LowerAveragePoolRecipe } from "../canonicalization/recipes/LowerAveragePoolRecipe.js";
import { LowerConcatRecipe } from "../canonicalization/recipes/LowerConcatRecipe.js";
import { LowerDequantizeLinearRecipe } from "../canonicalization/recipes/LowerDequantizeLinearRecipe.js";
import { LowerExpandRecipe } from "../canonicalization/recipes/LowerExpandRecipe.js";
import { LowerPadRecipe } from "../canonicalization/recipes/LowerPadRecipe.js";
import { LowerQuantizeLinearRecipe } from "../canonicalization/recipes/LowerQuantizeLinearRecipe.js";
import { LowerSliceRecipe } from "../canonicalization/recipes/LowerSliceRecipe.js";
import { LowerSoftmaxRecipe } from "../canonicalization/recipes/LowerSoftmaxRecipe.js";
import { LowerCoalescedMatMulRecipe } from "./recipes/LowerCoalescedMatMulRecipe.js";
import { CanonicalizationPass } from "../canonicalization/CanonicalizationPass.js";
import { DeadCodeEliminationPass } from "../optimization/DeadCodeEliminationPass.js";
import { LoopLoweringPass } from "../loop-lowering-v2/LoopLoweringPass.js";

export default class OnnxGraphTransformer implements Graph.Transformation<
    OnnxGraph.Class,
    OnnxGraph.Class
> {
    private fuse: boolean;
    private recurse: boolean;
    private coalesce: boolean;
    private loopLowering: boolean;
    private decomposeForCgra: boolean;

    // Overload signatures (for TypeScript type checking)
    constructor();
    constructor(options: Partial<DecompositionOptions>);
    constructor(fuse: boolean, recurse: boolean, coalesce: boolean, decomposeForCgra: boolean);

    constructor(
        fuseOrOptions: boolean | Partial<DecompositionOptions> = defaultDecompositionOptions.fuse,
        recurse: boolean = defaultDecompositionOptions.recurse,
        coalesce: boolean = defaultDecompositionOptions.coalesce,
        loopLowering: boolean = defaultDecompositionOptions.loopLowering,
        decomposeForCgra: boolean = defaultDecompositionOptions.decomposeForCgra,
    ) {
        if (typeof fuseOrOptions === "boolean") {
            this.fuse = fuseOrOptions;
            this.recurse = recurse;
            this.coalesce = coalesce;
            this.loopLowering = loopLowering;
            this.decomposeForCgra = decomposeForCgra;
        } else {
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

        // =========================================================================
        // PHASE 2: Lowering & Early Fusion
        // =========================================================================
        // This looks at our clean, primitive graph and finds chains (Transpose -> Add)
        // and lowers them directly into single, highly optimized ONNX Loops.
        if (this.loopLowering)
            pm.addPass(new LoopLoweringPass({ coalesce: this.coalesce, fuse: this.fuse }));

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
