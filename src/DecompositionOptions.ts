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

    /** Apply example CGRA decomposition of MatMuls/Relus */
    decomposeForCgra: boolean;
}

/**
 * Defaults chosen to match the current (best?) behavior:
 *  - canonicalize: true
 *  - fuse: true
 *  - recurse: false
 *  - coalesce: true
 *  - loopLowering: true
 *  - decomposeForCgra: false
 */
export const defaultDecompositionOptions: DecompositionOptions = {
    canonicalize: true,
    fuse: true,
    recurse: false,
    coalesce: true,
    loopLowering: true,
    decomposeForCgra: false,
};
