import { OpRegistry } from "./OpRegistry.js";
import { StandardOps } from "./definitions/standard-ops/index.js";

export function initializeSchemaRegistry(verbose: boolean = false): void {
    const registry = OpRegistry.getInstance();
    registry.registerAll(StandardOps);
    if (verbose) {
        console.log(`[Schema] Registered ${StandardOps.length} Standard Ops.`);
    }
}

export { OpRegistry } from "./OpRegistry.js";
export type { OpSchema } from "./OpSchema.js";
