import DefaultDotFormatter from "@specs-feup/flow/graph/dot/DefaultDotFormatter";
import OnnxGraph from "../OnnxGraph.js";
import Dot, { DotEdge, DotGraph, DotNode } from "@specs-feup/flow/graph/dot/dot";
import BaseNode from "@specs-feup/flow/graph/BaseNode";
import BaseEdge from "@specs-feup/flow/graph/BaseEdge";
import Node from "@specs-feup/flow/graph/Node";
import TensorNode from "../TensorNode.js";
import VariableNode from "../VariableNode.js";
import ConstantNode from "../ConstantNode.js";
import OperationNode from "../OperationNode.js";

type ClusterInfo = {
    idPrefix: string;
    nodeLabels: string[];
};

export default class OnnxDotFormatter<
    G extends OnnxGraph.Class = OnnxGraph.Class,
> extends DefaultDotFormatter<G> {
    private idPrefix: string;
    private clusterNodes: Record<string, ClusterInfo> = {};

    static defaultGetNodeAttrs(node: BaseNode.Class): Record<string, string> {
        const attrs = super.defaultGetNodeAttrs(node);

        node.switch(
            Node.Case(TensorNode, node => {
                attrs.shape = 'ellipse';

                if (node.type === 'input') {
                    attrs.color = '#00ff00';
                } else if (node.type === 'output') {
                    attrs.color = '#ff0000';
                } else {
                    attrs.color = '#ff00ff';
                }
            }),
            Node.Case(VariableNode, node => {
                attrs.label = node.name;
                attrs.shape = 'ellipse';
                attrs.color = node.type === 'input' ? '#00ff00' : '#ff0000';
            }),
            Node.Case(ConstantNode, node => {
                attrs.label = node.value.toString();
                attrs.shape = 'box';
                attrs.color = '#a52a2a';
            }),
            Node.Case(OperationNode, node => {
                attrs.label = node.type;
                attrs.color = '#0000ff';
            }),
        );

        return attrs;
    }

    nodeToDot(node: BaseNode.Class): DotNode {
        const id = this.idPrefix + node.id;
        const attrs = this.getNodeAttrs(node);

        return Dot.node(id, attrs);
    }

    createDotEdge(sourceId: string, targetId: string, attrs: Record<string, string> = {}): DotEdge {
        let source = this.idPrefix + sourceId;
        let target = this.idPrefix + targetId;

        if (sourceId in this.clusterNodes) {
            const sourceCluster = this.clusterNodes[sourceId];

            attrs.ltail = sourceCluster.nodeLabels[0];  // TODO(Process-ing): See if this is correct
            source = sourceCluster.idPrefix + source;
        }

        if (targetId in this.clusterNodes) {
            const targetCluster = this.clusterNodes[targetId];

            attrs.lhead = targetCluster.nodeLabels[0];  // TODO(Process-ing): See if this is correct
            target = targetCluster.idPrefix + target;
        }

        return Dot.edge(source, target, attrs);
    }

    edgeToDot(edge: BaseEdge.Class): DotEdge {
        const sourceId = edge.source.id;
        const targetId = edge.target.id;
        const attrs = this.getEdgeAttrs(edge);

        return this.createDotEdge(sourceId, targetId, attrs);
    }

    constructor(
        getNodeAttrs?: (node: BaseNode.Class) => Record<string, string>,
        getEdgeAttrs?: (edge: BaseEdge.Class) => Record<string, string>,
        getContainer?: (node: BaseNode.Class) => BaseNode.Class | undefined,
        getGraphAttrs?: () => Record<string, string>,
        idPrefix: string = ""
    ) {
        getNodeAttrs ??= OnnxDotFormatter.defaultGetNodeAttrs;
        getEdgeAttrs ??= DefaultDotFormatter.defaultGetEdgeAttrs;
        getContainer ??= DefaultDotFormatter.defaultGetContainer;
        getGraphAttrs ??= DefaultDotFormatter.defaultGetGraphAttrs;

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