export enum IndexingMode {
    Pointer = "pointer",
    Radix = "radix",
}

export interface DecompositionOptions {
    /** Break down complex operations into simpler primitives before lowering */
    canonicalize: boolean;

    /** Fuse supported ops into a single Loop when possible */
    fuse: boolean;

    /** Recursively decompose inside generated Loop bodies */
    recurse: boolean;

    /** Use coalesced scalar MAC for MatMul inside Loop bodies */
    coalesce: boolean;

    /** Apply loop lowering at all (create Loop nodes) */
    loopLowering: boolean;

    /** Run DeadCodeElimination, ShapeOptimization and other optimization passes */
    optimize: boolean;

    /** Apply example CGRA decomposition of MatMuls/Relus */
    decomposeForCgra: boolean;

    /** Addressing mode inside lowered Loop bodies: 'pointer' (AGU math) or 'radix' (div/mod) */
    indexingMode: IndexingMode;

    /** Optional loop tiling (strip-mining) factor; disabled when <= 1 or undefined */
    tileSize?: number;

    /** Target ONNX opset version for schema resolution & export (up to 28) */
    targetOpset?: number;

    /** Explicit ONNX IR version override for export (up to 14) */
    irVersion?: number;
}

/**
 * Defaults chosen to match the current (best?) behavior:
 *  - canonicalize: true
 *  - fuse: true
 *  - recurse: false
 *  - coalesce: true
 *  - loopLowering: true
 *  - optimize: false
 *  - decomposeForCgra: false
 *  - indexingMode: "pointer"
 */
export const defaultDecompositionOptions: DecompositionOptions = {
    canonicalize: true,
    fuse: true,
    recurse: false,
    coalesce: true,
    loopLowering: true,
    optimize: false,
    decomposeForCgra: false,
    indexingMode: IndexingMode.Pointer,
    tileSize: 0,
};
