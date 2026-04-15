import type { OpSchema } from "./OpSchema.js";
import { StandardOps } from "./definitions/StandardOps/index.js";

export class OpRegistry {
    private static instance: OpRegistry | undefined;
    // Map<OpType, Map<Version, Schema>>
    private schemas: Map<string, Map<number, OpSchema>> = new Map();

    private constructor() {}

    public static getInstance(): OpRegistry {
        if (OpRegistry.instance === undefined) {
            OpRegistry.instance = new OpRegistry();

            // Auto-initialize standard schemas so the registry is never empty
            OpRegistry.instance.registerAll(StandardOps);
        }
        return OpRegistry.instance;
    }

    /**
     * Register a new operator schema.
     */
    public register(schema: OpSchema): void {
        if (!this.schemas.has(schema.opType)) {
            this.schemas.set(schema.opType, new Map());
        }
        this.schemas.get(schema.opType)!.set(schema.sinceVersion, schema);
    }

    /**
     * Register multiple schemas at once.
     */
    public registerAll(schemas: OpSchema[]): void {
        schemas.forEach((s) => this.register(s));
    }

    /**
     * Retrieve the schema for a specific operator and opset version.
     */
    public get(opType: string, opsetVersion: number): OpSchema | undefined {
        const versionMap = this.schemas.get(opType);
        if (!versionMap) return undefined;

        let bestVersion = -1;
        for (const v of versionMap.keys()) {
            if (v <= opsetVersion && v > bestVersion) {
                bestVersion = v;
            }
        }

        if (bestVersion === -1) return undefined;
        return versionMap.get(bestVersion);
    }
}
