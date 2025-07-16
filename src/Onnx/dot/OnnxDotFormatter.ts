import DefaultDotFormatter from "@specs-feup/flow/graph/dot/DefaultDotFormatter";
import BaseEdge from "@specs-feup/flow/graph/BaseEdge";
import BaseNode from "@specs-feup/flow/graph/BaseNode";
import Dot, { DotEdge, DotGraph, DotNode, DotSubgraph } from "@specs-feup/flow/graph/dot/dot";
import TensorNode from "../TensorNode.js";
import OperationNode from "../OperationNode.js";
import ConstantNode from "../ConstantNode.js";
import VariableNode from "../VariableNode.js";
import OnnxEdge from "../OnnxEdge.js";
import OnnxGraph from "../OnnxGraph.js"
import Edge from "@specs-feup/flow/graph/Edge";
import Node from "@specs-feup/flow/graph/Node";

export default class OnnxDotFormatter<
    G extends OnnxGraph.Class = OnnxGraph.Class,
> extends DefaultDotFormatter<G> {
    private nodesInCluster: Record<string, string[]> = {};

    static defaultGetNodeAttrs(node: BaseNode.Class): Record<string, string> {
        let result: Record<string, string> | DotSubgraph = { label: node.id, shape: "box" }; 
        node.switch(
            Node.Case(TensorNode, (n) => {
                if (n.type === "input") {
                    result.color = "#00FF00";
                    result.shape = "ellipse";
                } else if (n.type === "output") {
                    result.color = "#FF0000";
                    result.shape = "ellipse";
                } else if (n.type === "index" || n.type === "index_aux") {
                    result.color = "#FF00FF";
                    result.shape = "ellipse";
                }
            }),
            Node.Case(VariableNode, (n) => {
                result.shape = "ellipse";
                result.label = n.name;
                result.color = n.type === "input" ? "#00FF00" : "#FF0000";
            }),
            Node.Case(ConstantNode, (n) => {
                result.label = n.value.toString();
                result.color = "#A52A2A";
                result.shape = "ellipse";
            }),
            Node.Case(OperationNode, (n) => {
                result.label = n.type;
                result.color = "#0000FF";
            }),
        );
        return result;
    }

    static defaultGetEdgeAttrs(edge: BaseEdge.Class): Record<string, string> {
        const result: Record<string, string> = {};
        edge.switch(
            Edge.Case(OnnxEdge, (e) => {
                const shapeString = `{${e.shape.join(',')}}`;
                result.label = shapeString === "{}" ? "" : `{${e.shape.join(',')}}`;
            }),
        );
        return result;
    }

    static defaultGetGraphAttrs(): Record<string, string> {
        return {
            rankdir: "LR",
            ...DefaultDotFormatter.defaultGetGraphAttrs(),
        };
    }

    constructor(
        getNodeAttrs?: (node: BaseNode.Class) => Record<string, string>,
        getEdgeAttrs?: (edge: BaseEdge.Class) => Record<string, string>,
        getContainer?: (node: BaseNode.Class) => BaseNode.Class | undefined,
        getGraphAttrs?: () => Record<string, string>,
        private idPrefix: string = "",
    ) {
        super(getNodeAttrs, getEdgeAttrs, getContainer, getGraphAttrs);
    }

    nodeToDot(node: BaseNode.Class): DotNode {
        return Dot.node(this.idPrefix + node.id, OnnxDotFormatter.defaultGetNodeAttrs(node));
    }

    edgeToDotIDs(sourceID: string, targetID: string, attrs: Record<string, string> = {}): DotEdge{
        let source = this.idPrefix + sourceID;
        let target = this.idPrefix + targetID;
        if (source in this.nodesInCluster){
            attrs["ltail"] = this.nodesInCluster[source][1];
            source = this.nodesInCluster[source][0] + source;
        }
        if (target in this.nodesInCluster){
            attrs["lhead"] = this.nodesInCluster[target][1];
            target = this.nodesInCluster[target][0] + target;
        }
        const dot = Dot.edge(
            source,
            target, 
            attrs
        );
        return dot;
    }

    edgeToDot(edge: BaseEdge.Class): DotEdge {
        let attrs = {};
        let source = this.idPrefix + edge.source.id;
        let target = this.idPrefix + edge.target.id;
        if (source in this.nodesInCluster){
            attrs["ltail"] = this.nodesInCluster[source][1];
            source = this.nodesInCluster[source][0] + source;
        }
        if (target in this.nodesInCluster){
            attrs["lhead"] = this.nodesInCluster[target][1];
            target = this.nodesInCluster[target][0] + target;
        }
        const dot = Dot.edge(
            source,
            target, 
            {...OnnxDotFormatter.defaultGetEdgeAttrs(edge), ...attrs}
        );
        return dot;
    }

    loopBodyToDot(node: OperationNode.Class): {DotSubgraph : DotSubgraph, idPrefix : string} | null {
        let idPrefix = `loop${node.id}_`;
        const body = node.getBodySubgraph?.();
        if (body && body instanceof OnnxGraph.Class) {
            // Format the body subgraph
            const subFormatter = new OnnxDotFormatter(
                OnnxDotFormatter.defaultGetNodeAttrs,
                OnnxDotFormatter.defaultGetEdgeAttrs,
                DefaultDotFormatter.defaultGetContainer,
                OnnxDotFormatter.defaultGetGraphAttrs,
                idPrefix
            );
            const bodyDot = subFormatter.toDot(body);

            // Wrap in cluster
            const cluster = new DotSubgraph(`cluster_loop_${node.id}`, bodyDot.statementList)
                .graphAttr("label", `Loop ${node.id}`)
                .graphAttr("style", "dashed")
                .graphAttr("color", "gray");

            return {DotSubgraph: cluster, idPrefix};
        }

        return null;
    }

    override toDot(graph: G): DotGraph {
        const dot = Dot.graph().graphAttrs(this.getGraphAttrs());
        const operationNodes = graph.getOperationNodes();

        for (const node of graph.nodes) {
            const tensorNode = node.tryAs(TensorNode)
            const isIntermediateTensor = tensorNode && tensorNode.type === "intermediate";

            if (isIntermediateTensor) {
                const incomers = tensorNode.getIncomers;
                const outgoers = tensorNode.getOutgoers;

                for (const inEdge of incomers) {
                    const src = inEdge.source;
                    const edgeAttrs = OnnxDotFormatter.defaultGetEdgeAttrs(inEdge);

                    for (const outEdge of outgoers) {
                        const tgt = outEdge.target;
                        dot.statements(this.edgeToDotIDs(src.id, tgt.id, edgeAttrs));
                    }
                }
                continue;
            }

            if (operationNodes.contains(node) && node.as(OperationNode).type === "Loop") {
                const result = this.loopBodyToDot(node.as(OperationNode));
                if (result) {
                    const { DotSubgraph: bodyCluster, idPrefix } = result;
                    dot.statements(bodyCluster);
                    this.nodesInCluster[node.id] = [idPrefix, bodyCluster.label];
                }
            } else if (!this.isContained(node)) {
                if (this.isContainer(node)) {
                    dot.statements(this.clusterNodeToDot(node));
                } else {
                    dot.statements(this.nodeToDot(node));
                }
            }
        }

        for (const edge of graph.edges) {
            const src = edge.source.id;
            const tgt = edge.target.id;

            // Skip edges involving intermediate TensorNodes (already handled above)
            if (
                (graph.getNodeById(src).tryAs(TensorNode)?.type === "intermediate") ||
                (graph.getNodeById(tgt).tryAs(TensorNode)?.type === "intermediate")
            ) continue;

            dot.statements(this.edgeToDot(edge));
        }

        // Extra: Add missing edges for operations like Gather, Scatter with external inputs
        for (const opNode of graph.getOperationNodes()) {
            if (["Gather", "Scatter"].includes(opNode.type)) {
                const inputTensors = opNode.getInputs().filter(n => !graph.hasNode(n.id));
                for (const ext of inputTensors) {
                    dot.statements(Dot.edge(
                        ext.id,
                        this.idPrefix + opNode.id,
                        { style: "dashed", color: "gray", label: "ext" }
                    ));
                }
            }
        }

        return dot;
    }
   
}

