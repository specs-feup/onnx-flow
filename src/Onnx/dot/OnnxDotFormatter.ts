import DefaultDotFormatter from "@specs-feup/flow/graph/dot/DefaultDotFormatter";
import OnnxGraph from "../OnnxGraph.js";
import Dot, {
    DotEdge,
    DotGraph,
    DotNode,
    DotStatement,
    DotSubgraph,
} from "@specs-feup/flow/graph/dot/dot";
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
    subgraphLabel: string;
};

export default class OnnxDotFormatter<
    G extends OnnxGraph.Class = OnnxGraph.Class,
> extends DefaultDotFormatter<G> {
    private idPrefix: string;
    private clusterInfos: Record<string, ClusterInfo> = {};

    static defaultGetNodeAttrs(node: BaseNode.Class): Record<string, string> {
        const attrs = super.defaultGetNodeAttrs(node);

        node.switch(
            Node.Case(TensorNode, (node) => {
                if (node.type === "input") {
                    attrs.shape = "ellipse";
                    attrs.color = "lime";
                } else if (node.type === "output") {
                    attrs.shape = "ellipse";
                    attrs.color = "red";
                } else if (["index", "index_aux"].includes(node.type)) {
                    attrs.shape = "ellipse";
                    attrs.color = "magenta";
                }
            }),
            Node.Case(VariableNode, (node) => {
                attrs.label = node.name;
                attrs.shape = "ellipse";
                attrs.color = node.type === "input" ? "lime" : "red";
            }),
            Node.Case(ConstantNode, (node) => {
                attrs.label = node.value.toString();
                attrs.shape = "box";
                attrs.color = "maroon";
            }),
            Node.Case(OperationNode, (node) => {
                attrs.label = node.type;
                attrs.color = "blue";
            }),
        );

        return attrs;
    }

    static shapeToLabel(shape: (number | string)[]): string {
        const shapeString = `{${shape.join(",")}}`;
        return shapeString === "{}" ? "sc" : shapeString;
    }

    static defaultGetEdgeAttrs(edge: BaseEdge.Class): Record<string, string> {
        const attrs = super.defaultGetEdgeAttrs(edge);
        const onnxEdge = edge.as(OnnxEdge);
        attrs.label = OnnxDotFormatter.shapeToLabel(onnxEdge.shape);

        return attrs;
    }

    static defaultGetGraphAttrs(): Record<string, string> {
        const attrs = super.defaultGetGraphAttrs();
        attrs.rankdir = "LR"; // Due to an oversight, this had no effect before the refactor

        return attrs;
    }

    constructor(
        idPrefix: string = "",
        getNodeAttrs?: (node: BaseNode.Class) => Record<string, string>,
        getEdgeAttrs?: (edge: BaseEdge.Class) => Record<string, string>,
        getContainer?: (node: BaseNode.Class) => BaseNode.Class | undefined,
        getGraphAttrs?: () => Record<string, string>,
    ) {
        getNodeAttrs ??= OnnxDotFormatter.defaultGetNodeAttrs;
        getEdgeAttrs ??= OnnxDotFormatter.defaultGetEdgeAttrs;
        getContainer ??= OnnxDotFormatter.defaultGetContainer; // Method not implemented, add if needed
        getGraphAttrs ??= OnnxDotFormatter.defaultGetGraphAttrs;

        super(getNodeAttrs, getEdgeAttrs, getContainer, getGraphAttrs);

        this.idPrefix = idPrefix;
    }

    override nodeToDot(node: BaseNode.Class): DotNode {
        const id = this.idPrefix + node.id;
        const attrs = this.getNodeAttrs(node);

        return Dot.node(id, attrs);
    }

    createDotEdge(sourceId: string, targetId: string, attrs: Record<string, string> = {}): DotEdge {
        let source = this.idPrefix + sourceId;
        let target = this.idPrefix + targetId;

        if (sourceId in this.clusterInfos) {
            const sourceCluster = this.clusterInfos[sourceId];

            attrs.ltail = sourceCluster.subgraphLabel;
            source = sourceCluster.idPrefix + source;
        }

        if (targetId in this.clusterInfos) {
            const targetCluster = this.clusterInfos[targetId];

            attrs.lhead = targetCluster.subgraphLabel;
            target = targetCluster.idPrefix + target;
        }

        return Dot.edge(source, target, attrs);
    }

    override edgeToDot(edge: BaseEdge.Class): DotEdge {
        const sourceId = edge.source.id;
        const targetId = edge.target.id;
        const attrs = this.getEdgeAttrs(edge);

        return this.createDotEdge(sourceId, targetId, attrs);
    }

    loopToDot(node: OperationNode.Class): DotStatement[] {
        const idPrefix = `loop${node.id}_`;
        const statements = [];

        const body = node.getBodySubgraph();
        if (body !== undefined) {
            const subFormatter = new OnnxDotFormatter(idPrefix);
            const bodyDot = subFormatter.toDot(body);

            const bodySubdot = new DotSubgraph(`cluster_loop_${node.id}`, bodyDot.statementList)
                .graphAttr("label", `Loop ${node.id}`)
                .graphAttr("style", "dashed")
                .graphAttr("color", "gray");

            statements.push(bodySubdot);

            this.clusterInfos[node.id] = {
                idPrefix,
                subgraphLabel: bodySubdot.label,
            };
        }

        return statements;
    }

    ifToDot(node: OperationNode.Class): DotStatement[] {
        const idPrefix = `if${node.id}_`;
        const statements = [];

        const ifDot = this.nodeToDot(node);
        statements.push(ifDot);

        const thenBranch = node.getThenBranch();
        if (thenBranch !== undefined) {
            const thenIdPrefix = `${idPrefix}then_`;
            const thenFormatter = new OnnxDotFormatter(thenIdPrefix);
            const thenDot = thenFormatter.toDot(thenBranch);

            const thenGraph = new DotSubgraph(`cluster_if_then_${node.id}`, thenDot.statementList)
                .graphAttr("label", `If-Then ${node.id}`)
                .graphAttr("style", "dashed")
                .graphAttr("color", "lime");

            const firstThenNode = thenBranch.nodes[0];
            const thenEdge = Dot.edge(
                this.idPrefix + node.id,
                thenFormatter.idPrefix + firstThenNode.id,
            )
                .attr("lhead", thenGraph.label)
                .attr("label", "then")
                .attr("style", "dashed")
                .attr("color", "lime");

            statements.push(thenGraph);
            statements.push(thenEdge);
        }

        const elseBranch = node.getElseBranch();
        if (elseBranch !== undefined) {
            const elseIdPrefix = `${idPrefix}else_`;
            const elseFormatter = new OnnxDotFormatter(elseIdPrefix);
            const elseDot = elseFormatter.toDot(elseBranch);

            const elseGraph = new DotSubgraph(`cluster_if_else_${node.id}`, elseDot.statementList)
                .graphAttr("label", `If-Else ${node.id}`)
                .graphAttr("style", "dashed")
                .graphAttr("color", "red");

            const firstElseNode = elseBranch.nodes[0];
            const elseEdge = Dot.edge(
                this.idPrefix + node.id,
                elseFormatter.idPrefix + firstElseNode.id,
            )
                .attr("lhead", elseGraph.label)
                .attr("label", "else")
                .attr("style", "dashed")
                .attr("color", "red");

            statements.push(elseGraph);
            statements.push(elseEdge);
        }

        return statements;
    }

    /**
     * @brief Tries to convert a node to an intermediate tensor.
     *
     * @param node The node to convert.
     * @returns The intermediate tensor node if compatible, otherwise undefined.
     */
    tryAsIntermediateTensor(node: BaseNode.Class): TensorNode.Class | undefined {
        const tensorNode = node.tryAs(TensorNode);
        if (tensorNode === undefined) return undefined;

        if (!["intermediate", "constant"].includes(tensorNode.type)) return undefined;

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
     * @brief Adds edges from external inputs to a given node.
     *
     * @param node The node to which to add external input edges.
     * @returns The resulting DOT statements.
     */
    externalInputsToDot(node: OperationNode.Class): DotStatement[] {
        const statements = [];

        const extInputs = node
            .getInputs()
            .filter((input) => !node.graph.as(OnnxGraph).hasNode(input.id));

        for (const input of extInputs) {
            const tensor = input.tryAs(TensorNode);
            if (tensor === undefined) continue;

            const targetId = this.idPrefix + node.id;
            const attrs = {
                label: OnnxDotFormatter.shapeToLabel(tensor.shape),
                style: "dashed",
                color: "gray",
            };

            const edge = Dot.edge(tensor.id, targetId, attrs);
            statements.push(edge);
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
                case "Loop":
                    return this.loopToDot(opNode);
                case "If":
                    return this.ifToDot(opNode);
                case "Gather":
                case "Scatter":
                    return [this.nodeToDot(opNode), ...this.externalInputsToDot(opNode)];
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
        if (
            this.tryAsIntermediateTensor(edge.source) !== undefined ||
            this.tryAsIntermediateTensor(edge.target) !== undefined
        ) {
            return [];
        }

        return null;
    }

    override toDot(graph: G): DotGraph {
        // Reset state
        this.clusterInfos = {};

        const dot = Dot.graph().graphAttrs(this.getGraphAttrs());
        const nodes = graph.nodes;

        for (const node of nodes.filter((node) => !this.isContained(node))) {
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

        // Extra: Add missing edges for operations like Gather, Scatter with external inputs
        for (const opNode of graph.getOperationNodes()) {
            if (
                ["Reshape", "Gather", "GatherElements", "Scatter", "ScatterElements"].includes(
                    opNode.type,
                )
            ) {
                const inputTensors = opNode.getInputs().filter((n) => !graph.hasNode(n.id));
                for (const ext of inputTensors) {
                    dot.statements(
                        Dot.edge(ext.id, this.idPrefix + opNode.id, {
                            style: "dashed",
                            color: "gray",
                            label:
                                ext.is(TensorNode) && ext.as(TensorNode).shape
                                    ? `{${ext.as(TensorNode).shape.join(",")}}`
                                    : "",
                        }),
                    );
                }
            }
        }

        return dot;
    }
}
