import DefaultDotFormatter from "@specs-feup/flow/graph/dot/DefaultDotFormatter";
import OnnxGraph from "../OnnxGraph.js";
import Dot, { DotEdge, DotGraph, DotNode, DotStatement, DotSubgraph } from "@specs-feup/flow/graph/dot/dot";
import BaseNode from "@specs-feup/flow/graph/BaseNode";
import BaseEdge from "@specs-feup/flow/graph/BaseEdge";
import Node from "@specs-feup/flow/graph/Node";
import TensorNode from "../TensorNode.js";
import VariableNode from "../VariableNode.js";
import ConstantNode from "../ConstantNode.js";
import OperationNode from "../OperationNode.js";
import OnnxEdge from "../OnnxEdge.js";

type ClusterInfo = {
    idPrefix: string;
    subgraphLabels: string[];
};

export default class OnnxDotFormatter<
    G extends OnnxGraph.Class = OnnxGraph.Class,
> extends DefaultDotFormatter<G> {
    private idPrefix: string;
    private clusterInfos: Record<string, ClusterInfo> = {};

    static defaultGetNodeAttrs(node: BaseNode.Class): Record<string, string> {
        const attrs = super.defaultGetNodeAttrs(node);

        node.switch(
            Node.Case(TensorNode, node => {
                if (node.type === 'input') {
                    attrs.shape = 'ellipse';
                    attrs.color = '#00ff00';
                } else if (node.type === 'output') {
                    attrs.shape = 'ellipse';
                    attrs.color = '#ff0000';
                } else if (['index', 'index_aux'].includes(node.type)) {
                    attrs.shape = 'ellipse';
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

    static defaultGetEdgeAttrs(edge: BaseEdge.Class): Record<string, string> {
        const attrs = super.defaultGetEdgeAttrs(edge);

        const onnxEdge = edge.tryAs(OnnxEdge);
        if (onnxEdge !== undefined) {
            const shapeString = `{${onnxEdge.shape.join(',')}}`;
            attrs.label = shapeString === '{}' ? 'sc' : shapeString;
        }

        return attrs;
    }

    static defaultGetGraphAttrs(): Record<string, string> {
        const attrs = super.defaultGetGraphAttrs();
        attrs.rankdir = 'LR';  // Due to an oversight, this had no effect before the refactor

        return attrs;
    }

    constructor(
        idPrefix: string = "",
        getNodeAttrs?: (node: BaseNode.Class) => Record<string, string>,
        getEdgeAttrs?: (edge: BaseEdge.Class) => Record<string, string>,
        getContainer?: (node: BaseNode.Class) => BaseNode.Class | undefined,
        getGraphAttrs?: () => Record<string, string>
    ) {
        getNodeAttrs ??= OnnxDotFormatter.defaultGetNodeAttrs;
        getEdgeAttrs ??= OnnxDotFormatter.defaultGetEdgeAttrs;
        getContainer ??= OnnxDotFormatter.defaultGetContainer;  // Method not implemented, add if needed
        getGraphAttrs ??= OnnxDotFormatter.defaultGetGraphAttrs;

        super(getNodeAttrs, getEdgeAttrs, getContainer, getGraphAttrs);

        this.idPrefix = idPrefix;
    }

    nodeToDot(node: BaseNode.Class): DotNode {
        const id = this.idPrefix + node.id;
        const attrs = this.getNodeAttrs(node);

        return Dot.node(id, attrs);
    }

    createDotEdge(sourceId: string, targetId: string, attrs: Record<string, string> = {}): DotEdge {
        let source = this.idPrefix + sourceId;
        let target = this.idPrefix + targetId;

        if (sourceId in this.clusterInfos) {
            const sourceCluster = this.clusterInfos[sourceId];

            attrs.ltail = sourceCluster.subgraphLabels[0];  // TODO(Process-ing): See if this is correct
            source = sourceCluster.idPrefix + source;
        }

        if (targetId in this.clusterInfos) {
            const targetCluster = this.clusterInfos[targetId];

            attrs.lhead = targetCluster.subgraphLabels[0];  // TODO(Process-ing): See if this is correct
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

    loopToDot(node: OperationNode.Class): DotStatement[] {
        const idPrefix = `loop${node.id}_`;
        const statements = [];

        const body = node.getBodySubgraph();
        if (body === undefined) {
            const subFormatter = new OnnxDotFormatter(idPrefix);
            const bodyDot = subFormatter.toDot(body);

            const bodySubdot = new DotSubgraph(`cluster_loop_${node.id}`, bodyDot.statementList)
                .graphAttr('label', `Loop ${node.id}`)
                .graphAttr('style', 'dashed')
                .graphAttr('color', 'gray');
            statements.push(bodySubdot);

            this.clusterInfos[node.id] = {
                idPrefix,
                subgraphLabels: [bodySubdot.label],
            };
        }

        return statements;
    }

    ifToDot(node: OperationNode.Class): DotStatement[] {
        const idPrefix = `if${node.id}_`;
        const statements = [];
        const subgraphLabels = [];

        const thenBranch = node.getThenBranch();
        if (thenBranch !== undefined) {
            const thenFormatter = new OnnxDotFormatter(`${idPrefix}then_`);
            const thenDot = thenFormatter.toDot(thenBranch);

            const thenGraph = new DotSubgraph(`cluster_if_then_${node.id}`, thenDot.statementList)
                .graphAttr('label', `If-Then ${node.id}`)
                .graphAttr('style', 'dashed')
                .graphAttr('color', 'green');

            const thenEdge = Dot.edge(this.idPrefix + node.id, `${idPrefix}then_condition`)
                .attr('label', 'then')
                .attr('style', 'dashed')
                .attr('color', 'green');

            statements.push(thenGraph);
            statements.push(thenEdge);

            subgraphLabels.push(thenGraph.label);
        }

        const elseBranch = node.getElseBranch();
        if (elseBranch !== undefined) {
            const elseFormatter = new OnnxDotFormatter(`${idPrefix}else_`);
            const elseDot = elseFormatter.toDot(elseBranch);

            const elseGraph = new DotSubgraph(`cluster_if_else_${node.id}`, elseDot.statementList)
                .graphAttr('label', `If-Else ${node.id}`)
                .graphAttr('style', 'dashed')
                .graphAttr('color', '#00ff00');

            const elseEdge = Dot.edge(this.idPrefix + node.id, `${idPrefix}else_condition`)
                .attr('label', 'else')
                .attr('style', 'dashed')
                .attr('color', '#ff0000');

            statements.push(elseGraph);
            statements.push(elseEdge);

            subgraphLabels.push(elseGraph.label);
        }

        this.clusterInfos[node.id] = {
            idPrefix,
            subgraphLabels,
        };

        return statements;
    }

    /**
     * @brief Tries to turn a node as an intermediate tensor.
     *
     * @param node The node to convert.
     * @returns The intermediate tensor node if compatible, otherwise undefined.
     */
    tryAsIntermediateTensor(node: BaseNode.Class): TensorNode.Class | undefined {
        const tensorNode = node.tryAs(TensorNode);
        if (tensorNode === undefined)
            return undefined;

        if (!['intermediate', 'constant'].includes(tensorNode.type))
            return undefined;

        return tensorNode;
    }

    /**
     * @brief Converts an intermediate tensor node into DOT statements.
     * This method short-circuits edges that connect through intermediate
     * tensors, to hide the respective tensor nodes from the graph.
     *
     * @param node The intermediate tensor node to convert.
     * @returns The resulting DOT statements.
     */
    intermediateTensorToDot(node: TensorNode.Class): DotStatement[] {
        const statements = [];

        const incomers = node.getIncomers;
        const outgoers = node.getOutgoers;

        for (const inEdge of incomers) {
            const source = inEdge.source;
            const edgeAttrs = this.getEdgeAttrs(inEdge);

            for (const outEdge of outgoers) {
                const target = outEdge.target;
                const newEdge = this.createDotEdge(source.id, target.id, edgeAttrs);
                statements.push(newEdge);
            }
        }

        return statements;
    }

    /**
     * @brief Handles special cases in the conversion from node to DOT.
     *
     * @param node The node to convert.
     * @returns The resulting DOT statements.
     */
    specialNodeToDot(node: BaseNode.Class): DotStatement[] | null {
        const tensorNode = this.tryAsIntermediateTensor(node);
        if (tensorNode !== undefined) {
            return this.intermediateTensorToDot(tensorNode);
        }

        const opNode = node.tryAs(OperationNode);
        if (opNode !== undefined) {
            switch (opNode.type) {
                case 'Loop':
                    return this.loopToDot(opNode);
                case 'If':
                    return this.ifToDot(opNode);
            }
        }

        return null;
    }

    /**
     * @brief Handles special cases in the conversion from edge to DOT.
     *
     * @param edge The edge to convert.
     * @returns The resulting DOT statements.
     */
    specialEdgeToDot(edge: BaseEdge.Class): DotStatement[] | null {
        // Ignore original edges from and to intermediate tensors
        if (this.tryAsIntermediateTensor(edge.source) !== undefined
            || this.tryAsIntermediateTensor(edge.target) !== undefined) {
            return [];
        }

        return null;
    }

    override toDot(graph: G): DotGraph {
        const dot = Dot.graph().graphAttrs(this.getGraphAttrs());
        const nodes = graph.nodes;

        for (const node of nodes.filter(node => !this.isContained(node))) {
            const statements = this.specialNodeToDot(node);
            if (statements !== null) {
                dot.statements(...statements);
                continue;
            }

            if (this.isContainer(node)) {
                dot.statements(this.clusterNodeToDot(node));
            } else {
                dot.statements(this.nodeToDot(node));
            }
        }

        for (const edge of graph.edges) {
            const statements = this.specialEdgeToDot(edge);
            if (statements !== null) {
                dot.statements(...statements);
                continue;
            }

            dot.statements(this.edgeToDot(edge));
        }

        return dot;
    }
}