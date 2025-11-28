export interface DecompositionOptions {
  /** Fuse supported ops into a single Loop when possible */
  fuse: boolean;

  /** Recursively decompose inside generated Loop bodies */
  recurse: boolean;

  /** Use coalesced scalar MAC for MatMul inside Loop bodies */
  coalesce: boolean;
  
  decomposeForCgra: boolean;
}

/**
 * Defaults chosen to match the current behavior:
 *  - fuse: true
 *  - recurse: false
 *  - coalesce: true
 */
export const defaultDecompositionOptions: DecompositionOptions = {
  fuse: true,
  recurse: false,
  coalesce: true,
  decomposeForCgra: false,
};
