import { OpRegistry } from "./OpRegistry.js";
import { StandardOps } from "./definitions/StandardOps/index.js";

export function initializeSchemaRegistry(): void {
    const registry = OpRegistry.getInstance();
    registry.registerAll(StandardOps);
    // Add other ops here
    console.log(`[Schema] Registered ${StandardOps.length} Standard Ops.`);
}

export { OpRegistry } from "./OpRegistry.js";
export { OpSchema } from "./OpSchema.js";
