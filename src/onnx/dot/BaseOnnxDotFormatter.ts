import DefaultDotFormatter from "@specs-feup/flow/graph/dot/DefaultDotFormatter";
import type OnnxGraph from "../OnnxGraph.js";
import type { DotEdge, DotNode, DotStatement } from "@specs-feup/flow/graph/dot/dot";
import Dot from "@specs-feup/flow/graph/dot/dot";
import type BaseNode from "@specs-feup/flow/graph/BaseNode";
import type BaseEdge from "@specs-feup/flow/graph/BaseEdge";
import TensorNode from "../TensorNode.js";
import type { Shape } from "../OnnxTypes.js";

/**
 * Abstract base DOT formatter for `OnnxGraph` instances.
 *
 * Encapsulates shared ID-prefix scoping, shape label formatting, and
 * intermediate-tensor short-circuiting so concrete formatters (`OnnxDotFormatter`,
 * `CgraDotFormatter`) only need to implement target-specific styling and op lowering.
 */
export default abstract class BaseOnnxDotFormatter<
    G extends OnnxGraph.Class = OnnxGraph.Class,
> extends DefaultDotFormatter<G> {
    protected idPrefix: string;

    /** Formats a tensor shape as `{d0,d1,...}`, or `"sc"` for rank-0 scalars. */
    public static shapeToLabel(shape?: Shape): string {
        if (!shape || !Array.isArray(shape)) {
            return "";
        }
        const shapeString = `{${shape.join(",")}}`;
        return shapeString === "{}" ? "sc" : shapeString;
    }

    constructor(
        idPrefix: string = "",
        getNodeAttrs: (node: BaseNode.Class) => Record<string, string>,
        getEdgeAttrs: (edge: BaseEdge.Class) => Record<string, string>,
        getContainer: (node: BaseNode.Class) => BaseNode.Class | undefined,
        getGraphAttrs: () => Record<string, string>,
    ) {
        super(getNodeAttrs, getEdgeAttrs, getContainer, getGraphAttrs);
        this.idPrefix = idPrefix;
    }

    public override nodeToDot(node: BaseNode.Class): DotNode {
        const id = this.idPrefix + node.id;
        const attrs = this.getNodeAttrs(node);
        return Dot.node(id, attrs);
    }

    public abstract createDotEdge(
        sourceId: string,
        targetId: string,
        attrs?: Record<string, string>,
    ): DotEdge;

    public override edgeToDot(edge: BaseEdge.Class): DotEdge {
        return this.createDotEdge(edge.source.id, edge.target.id, this.getEdgeAttrs(edge));
    }

    /** Returns `node` typed as `TensorNode.Class` if it is an `"intermediate"` tensor. */
    public tryAsIntermediateTensor(node: BaseNode.Class): TensorNode.Class | undefined {
        const tensorNode = node.tryAs(TensorNode);
        if (tensorNode === undefined || tensorNode.type !== "intermediate") {
            return undefined;
        }
        return tensorNode;
    }

    /**
     * Short-circuits edges passing through an intermediate tensor node so only
     * direct producer-to-consumer edges appear in the rendered DOT graph.
     */
    public intermediateTensorToDot(node: TensorNode.Class): DotStatement[] {
        const statements: DotEdge[] = [];
        const incomers = node.getIncomers;
        const outgoers = node.getOutgoers;

        for (const inEdge of incomers) {
            const source = inEdge.source;
            const edgeAttrs = this.getEdgeAttrs(inEdge);
            for (const outEdge of outgoers) {
                statements.push(this.createDotEdge(source.id, outEdge.target.id, edgeAttrs));
            }
        }

        return statements;
    }

    /** Suppresses raw edges connected to short-circuited intermediate tensor nodes. */
    public specialEdgeToDot(edge: BaseEdge.Class): DotStatement[] | null {
        if (
            this.tryAsIntermediateTensor(edge.source) !== undefined ||
            this.tryAsIntermediateTensor(edge.target) !== undefined
        ) {
            return [];
        }
        return null;
    }
}
