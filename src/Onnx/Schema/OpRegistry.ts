import type { OpSchema } from "./OpSchema.js";

export class OpRegistry {
    private static instance: OpRegistry | undefined;
    // Map<OpType, Map<Version, Schema>>
    private schemas: Map<string, Map<number, OpSchema>> = new Map();

    private constructor() {}

    public static getInstance(): OpRegistry {
        if (OpRegistry.instance === undefined) {
            OpRegistry.instance = new OpRegistry();
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
     * It implements "backwards compatibility" logic: finding the highest version
     * defined that is less than or equal to the requested version.
     */
    public get(opType: string, opsetVersion: number): OpSchema | undefined {
        const versionMap = this.schemas.get(opType);
        if (!versionMap) return undefined;

        let bestVersion = -1;
        // Find the closest version definition (standard ONNX resolution logic)
        for (const v of versionMap.keys()) {
            if (v <= opsetVersion && v > bestVersion) {
                bestVersion = v;
            }
        }

        if (bestVersion === -1) return undefined;
        return versionMap.get(bestVersion);
    }
}
