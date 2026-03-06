import type Graph from "@specs-feup/flow/graph/Graph";
import type { DecompositionOptions } from "@specs-feup/onnx-flow/DecompositionOptions";
import { defaultDecompositionOptions } from "@specs-feup/onnx-flow/DecompositionOptions";
import type OnnxGraph from "../../OnnxGraph.js";
import { PassManager } from "../../PassManager.js";
import { OrchestratorPass } from "../OrchestratorPass.js";
import { MatMulGridDecompositionRecipe } from "../cgra-decomposition/MatMul.js";
import { AddGridDecompositionRecipe } from "../cgra-decomposition/Add.js";
import { ReluGridDecompositionRecipe } from "../cgra-decomposition/Relu.js";
import { LowerMatMulRecipe } from "./recipes/LowerMatMulRecipe.js";
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

        // 1) If CGRA decomposition is enabled, perform it only
        if (this.decomposeForCgra === true) {
            pm.addPass(
                new OrchestratorPass("SpatialDecomposition", [
                    new MatMulGridDecompositionRecipe(),
                    new AddGridDecompositionRecipe(),
                    new ReluGridDecompositionRecipe(),
                ]),
            );

            pm.run(graph);
            return graph;
        }

        // 2) Canonical version of high-level operations (no explicit Loop needed)
        pm.addPass(
            new OrchestratorPass("CanonicalizationPass", [
                new LowerAveragePoolRecipe(),
                new LowerClipRecipe(),
                new LowerConcatRecipe(),
                new LowerDequantizeLinearRecipe(),
                new LowerExpandRecipe(),
                new LowerGemmRecipe(),
                new LowerPadRecipe(),
                new LowerQuantizeLinearRecipe(),
                new LowerSliceRecipe(),
                new LowerSoftmaxRecipe(),
            ]),
        );

        //pm.addPass(new DeadCodeEliminationPass());

        pm.addPass(new InferShapesPass());

        // 3) Optionally perform loop-lowering
        if (this.loopLowering === true) {
            pm.addPass(
                new OrchestratorPass("TemporalLowering", [
                    //new LowerMatMulRecipe(),
                    new LowerCoalescedMatMulRecipe(),
                    new LowerConvRecipe(),
                    new LowerTransposeRecipe(),
                    new LowerRangeRecipe(),
                    new LowerElementWiseRecipe(),
                    new LowerReductionRecipe(),
                ]),
            );

            //pm.addPass(new DeadCodeEliminationPass());
            pm.addPass(new InferShapesPass());
        }

        pm.run(graph);
        return graph;
    }
}
