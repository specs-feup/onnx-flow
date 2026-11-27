import type OnnxGraph from "../ir/OnnxGraph.js";
import inferShapes from "../schema/InferShapes.js";
import validateGraph, { type ValidationOptions } from "../validation/ValidateGraph.js";

/** A single self-contained subgraph produced by partitioning a larger `OnnxGraph`. */
export interface GraphPartition {
    readonly id: string;
    readonly graph: OnnxGraph.Class;
    readonly inputs: readonly string[];
    readonly outputs: readonly string[];
    readonly metadata?: Record<string, unknown>;
}

/** Directed tensor dependency crossing from one partition's output to another's input. */
export interface PartitionConnection {
    readonly fromPartition: string;
    readonly toPartition: string;
    readonly tensorId: string;
}

/**
 * Container holding multiple partitioned `OnnxGraph` instances and their inter-partition
 * tensor connections. Supports N-way execution graphs while preserving `head` / `tail`
 * accessors for 2-way split workflows.
 */
export class PartitionedGraph {
    private readonly partitions: Map<string, GraphPartition>;
    public readonly connections: readonly PartitionConnection[];

    constructor(
        partitions: Iterable<GraphPartition>,
        connections: readonly PartitionConnection[] = [],
    ) {
        this.partitions = new Map();
        for (const p of partitions) {
            if (this.partitions.has(p.id)) {
                throw new Error(`[PartitionedGraph] Duplicate partition id '${p.id}'.`);
            }
            this.partitions.set(p.id, p);
        }
        this.connections = [...connections];
    }

    /** Number of partitions contained in this partitioned graph. */
    public get size(): number {
        return this.partitions.size;
    }

    /** Backwards-compatible accessor for the `"head"` (or first) partition graph. */
    public get head(): OnnxGraph.Class {
        const named = this.partitions.get("head");
        if (named !== undefined) return named.graph;
        const first = this.partitions.values().next().value as GraphPartition | undefined;
        if (first === undefined) {
            throw new Error(
                "[PartitionedGraph] Cannot access 'head' on an empty PartitionedGraph.",
            );
        }
        return first.graph;
    }

    /** Backwards-compatible accessor for the `"tail"` (or second) partition graph. */
    public get tail(): OnnxGraph.Class {
        const named = this.partitions.get("tail");
        if (named !== undefined) return named.graph;
        const all = this.getPartitions();
        if (all.length < 2) {
            throw new Error(
                "[PartitionedGraph] Cannot access 'tail' when fewer than 2 partitions exist.",
            );
        }
        return all[1].graph;
    }

    /** Returns the partition descriptor for `id`, if present. */
    public getPartition(id: string): GraphPartition | undefined {
        return this.partitions.get(id);
    }

    /** Returns the `OnnxGraph` for partition `id`, if present. */
    public getGraph(id: string): OnnxGraph.Class | undefined {
        return this.partitions.get(id)?.graph;
    }

    /** Returns all partition descriptors in insertion order. */
    public getPartitions(): GraphPartition[] {
        return Array.from(this.partitions.values());
    }

    /** Returns all partition `OnnxGraph` instances in insertion order. */
    public getGraphs(): OnnxGraph.Class[] {
        return this.getPartitions().map((p) => p.graph);
    }

    /** Returns all partition IDs in insertion order. */
    public getPartitionIds(): string[] {
        return Array.from(this.partitions.keys());
    }

    /**
     * Returns boundary tensor connections, optionally filtered by source and/or target partition ID.
     */
    public getBoundaryTensors(fromId?: string, toId?: string): PartitionConnection[] {
        return this.connections.filter(
            (c) =>
                (fromId === undefined || c.fromPartition === fromId) &&
                (toId === undefined || c.toPartition === toId),
        );
    }

    /**
     * Returns partitions in a valid topological execution order derived from `connections`,
     * breaking ties by original insertion order.
     */
    public getExecutionOrder(): GraphPartition[] {
        const orderIndex = new Map<string, number>();
        const inDegree = new Map<string, number>();
        const adjacency = new Map<string, Set<string>>();

        let idx = 0;
        for (const id of this.partitions.keys()) {
            orderIndex.set(id, idx++);
            inDegree.set(id, 0);
            adjacency.set(id, new Set());
        }

        for (const conn of this.connections) {
            if (conn.fromPartition === conn.toPartition) continue;
            const targets = adjacency.get(conn.fromPartition);
            if (targets !== undefined && !targets.has(conn.toPartition)) {
                targets.add(conn.toPartition);
                inDegree.set(conn.toPartition, (inDegree.get(conn.toPartition) ?? 0) + 1);
            }
        }

        const queue: string[] = [];
        for (const [id, deg] of inDegree.entries()) {
            if (deg === 0) queue.push(id);
        }
        queue.sort((a, b) => (orderIndex.get(a) ?? 0) - (orderIndex.get(b) ?? 0));

        const sorted: GraphPartition[] = [];
        while (queue.length > 0) {
            const currentId = queue.shift()!;
            sorted.push(this.partitions.get(currentId)!);

            const neighbors = adjacency.get(currentId);
            if (neighbors !== undefined) {
                const newlyReady: string[] = [];
                for (const nextId of neighbors) {
                    const nextDeg = (inDegree.get(nextId) ?? 1) - 1;
                    inDegree.set(nextId, nextDeg);
                    if (nextDeg === 0) newlyReady.push(nextId);
                }
                if (newlyReady.length > 0) {
                    queue.push(...newlyReady);
                    queue.sort((a, b) => (orderIndex.get(a) ?? 0) - (orderIndex.get(b) ?? 0));
                }
            }
        }

        if (sorted.length !== this.partitions.size) {
            throw new Error(
                "[PartitionedGraph] Cyclic dependency detected between graph partitions.",
            );
        }

        return sorted;
    }

    /** Runs shape inference across every partition in topological execution order. */
    public inferAllShapes(): void {
        for (const partition of this.getExecutionOrder()) {
            inferShapes(partition.graph);
        }
    }

    /** Validates every partition graph in topological execution order. */
    public validateAll(options?: ValidationOptions): void {
        for (const partition of this.getExecutionOrder()) {
            validateGraph(partition.graph, options);
        }
    }
}
