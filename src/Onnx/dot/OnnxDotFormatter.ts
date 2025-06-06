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

    static defaultGetNodeAttrs(node: BaseNode.Class): Record<string, string> {
        const result: Record<string, string> = { label: node.id, shape: "box" };
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

    constructor() {
        super(
            OnnxDotFormatter.defaultGetNodeAttrs,
            OnnxDotFormatter.defaultGetEdgeAttrs,
            DefaultDotFormatter.defaultGetContainer,
            OnnxDotFormatter.defaultGetGraphAttrs,
        );
    }

    toDot(graph: G): DotGraph {
        const dot = super.toDot(graph); // get the base DOT graph from DefaultDotFormatter

        // Inject subgraphs for Loop nodes
        for (const loopNode of graph.getOperationNodes()) {
            if (loopNode.type === "Loop") {
                const body = loopNode.getBodySubgraph?.();
                if (body && body instanceof OnnxGraph.Class) {
                    // Format the body subgraph
                    const subFormatter = new OnnxDotFormatter();
                    const bodyDot = subFormatter.toDot(body); // <-- use toDot, not format/generate

                    // Wrap in cluster
                    const cluster = new DotSubgraph(`cluster_loop_${loopNode.id}`, bodyDot.statementList)
                        .graphAttr("label", `Loop ${loopNode.id}`)
                        .graphAttr("style", "dashed")
                        .graphAttr("color", "gray");

                    // Append to the main DOT graph
                    dot.statements(cluster);
                }
            }
        }

        return dot;
    }

}

