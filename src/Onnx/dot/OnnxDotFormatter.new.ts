import DefaultDotFormatter from "@specs-feup/flow/graph/dot/DefaultDotFormatter";
import OnnxGraph from "../OnnxGraph.js";
import Dot, { DotEdge, DotGraph, DotNode } from "@specs-feup/flow/graph/dot/dot";
import BaseNode from "@specs-feup/flow/graph/BaseNode";
import BaseEdge from "@specs-feup/flow/graph/BaseEdge";

export default class OnnxDotFormatter<
    G extends OnnxGraph.Class = OnnxGraph.Class,
> extends DefaultDotFormatter<G> {
    private idPrefix: string;

    nodeToDot(node: BaseNode.Class): DotNode {
        const attrs = this.getNodeAttrs(node);
        return Dot.node(node.id, attrs);
    }

    constructor(
        getNodeAttrs?: (node: BaseNode.Class) => Record<string, string>,
        getEdgeAttrs?: (edge: BaseEdge.Class) => Record<string, string>,
        getContainer?: (node: BaseNode.Class) => BaseNode.Class | undefined,
        getGraphAttrs?: () => Record<string, string>,
        idPrefix: string = ""
    ) {
        super(getNodeAttrs, getEdgeAttrs, getContainer, getGraphAttrs);

        this.idPrefix = idPrefix;
    }

    getExtraEdges(node: BaseNode.Class): DotEdge[] {
        return [];
    }

    override toDot(graph: G): DotGraph {
        const dot = Dot.graph().graphAttrs(this.getGraphAttrs());
        const nodes = graph.nodes;

        for (const node of nodes.filter(node => !this.isContained(node))) {
            if (this.isContainer(node)) {
                dot.statements(this.clusterNodeToDot(node));
            } else {
                dot.statements(this.nodeToDot(node));
            }
        }

        for (const edge of graph.edges) {
            dot.statements(this.edgeToDot(edge));
        }

        for (const node of nodes) {
            dot.statements(...this.getExtraEdges(node));
        }

        return dot;
    }
}