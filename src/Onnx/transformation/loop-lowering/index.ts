import Graph from "@specs-feup/flow/graph/Graph";
import {
    DecompositionOptions,
    defaultDecompositionOptions,
} from "@specs-feup/onnx-flow/DecompositionOptions";
import OnnxGraph from "../../OnnxGraph.js";
import applyCanonicalization from "../canonicalization/index.js";
import TransformChain from "./TransformChain.js";
import transformForCgra from "../cgra-decomposition/index.js";

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
        // 1) If CGRA decomposition is enabled, perform it only
        if (this.decomposeForCgra) {
            return transformForCgra(graph);
        }

        // 2) Canonical version of high-level operations (no explicit Loop needed)
        const canon = applyCanonicalization(graph);

        // 3) Optionally perform loop-lowering
        if (!this.loopLowering) {
            // Return canonicalised graph with no explicit Loop nodes
            return canon;
        }

        return new TransformChain(this.fuse, this.recurse, this.coalesce).apply(canon);
    }
}
