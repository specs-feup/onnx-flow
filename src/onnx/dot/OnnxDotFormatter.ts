import OnnxGraph from "../ir/OnnxGraph.js";
import type { DotEdge, DotGraph, DotNode, DotStatement } from "@specs-feup/flow/graph/dot/dot";
import Dot, { DotSubgraph } from "@specs-feup/flow/graph/dot/dot";
import type BaseNode from "@specs-feup/flow/graph/BaseNode";
import type BaseEdge from "@specs-feup/flow/graph/BaseEdge";
import Node from "@specs-feup/flow/graph/Node";
import TensorNode from "../ir/TensorNode.js";
import ConstantNode from "../ir/ConstantNode.js";
import OperationNode from "../ir/OperationNode.js";
import OnnxEdge from "../ir/OnnxEdge.js";
import { readTensorData } from "../utils/index.js";
import BaseOnnxDotFormatter from "./BaseOnnxDotFormatter.js";

type ClusterInfo = {
    idPrefix: string;
    subgraphLabel: string;
};

/**
 * Standard Graphviz DOT formatter for `OnnxGraph`, rendering nested `Loop` and
 * `If` subgraphs as dashed clusters and annotating edges with tensor shapes.
 */
export default class OnnxDotFormatter<
    G extends OnnxGraph.Class = OnnxGraph.Class,
> extends BaseOnnxDotFormatter<G> {
    private clusterInfos: Record<string, ClusterInfo> = {};

    static override defaultGetNodeAttrs(node: BaseNode.Class): Record<string, string> {
        const attrs = super.defaultGetNodeAttrs(node);

        node.switch(
            Node.Case(TensorNode, (tn) => {
                if (tn.type === "input") {
                    attrs["shape"] = "ellipse";
                    attrs["color"] = "lime";
                } else if (tn.type === "output") {
                    attrs["shape"] = "ellipse";
                    attrs["color"] = "red";
                } else if (["index", "index_aux"].includes(tn.type)) {
                    attrs["shape"] = "ellipse";
                    attrs["color"] = "magenta";
                }
            }),
            Node.Case(ConstantNode, (cn) => {
                const val = readTensorData(cn);
                attrs["label"] = val ? val.slice(0, 5).toString() : "empty";
                attrs["shape"] = "box";
                attrs["color"] = "maroon";
            }),
            Node.Case(OperationNode, (op) => {
                attrs["label"] = op.type;
                attrs["color"] = "blue";
            }),
        );

        return attrs;
    }

    static override defaultGetEdgeAttrs(edge: BaseEdge.Class): Record<string, string> {
        const attrs = super.defaultGetEdgeAttrs(edge);
        const onnxEdge = edge.as(OnnxEdge);
        attrs["label"] = OnnxDotFormatter.shapeToLabel(onnxEdge.shape);
        return attrs;
    }

    static override defaultGetGraphAttrs(): Record<string, string> {
        const attrs = super.defaultGetGraphAttrs();
        attrs["rankdir"] = "LR";
        return attrs;
    }

    constructor(
        idPrefix: string = "",
        getNodeAttrs?: (node: BaseNode.Class) => Record<string, string>,
        getEdgeAttrs?: (edge: BaseEdge.Class) => Record<string, string>,
        getContainer?: (node: BaseNode.Class) => BaseNode.Class | undefined,
        getGraphAttrs?: () => Record<string, string>,
    ) {
        super(
            idPrefix,
            getNodeAttrs ?? OnnxDotFormatter.defaultGetNodeAttrs,
            getEdgeAttrs ?? OnnxDotFormatter.defaultGetEdgeAttrs,
            getContainer ?? OnnxDotFormatter.defaultGetContainer,
            getGraphAttrs ?? OnnxDotFormatter.defaultGetGraphAttrs,
        );
    }

    public override createDotEdge(
        sourceId: string,
        targetId: string,
        attrs: Record<string, string> = {},
    ): DotEdge {
        let source = this.idPrefix + sourceId;
        let target = this.idPrefix + targetId;

        if (sourceId in this.clusterInfos) {
            const sourceCluster = this.clusterInfos[sourceId];
            attrs["ltail"] = sourceCluster.subgraphLabel;
            source = sourceCluster.idPrefix + source;
        }

        if (targetId in this.clusterInfos) {
            const targetCluster = this.clusterInfos[targetId];
            attrs["lhead"] = targetCluster.subgraphLabel;
            target = targetCluster.idPrefix + target;
        }

        return Dot.edge(source, target, attrs);
    }

    public loopToDot(node: OperationNode.Class): DotStatement[] {
        const idPrefix = `loop${node.id}_`;
        const body = node.regions[0];
        const subFormatter = new OnnxDotFormatter(idPrefix);
        const bodyDot = subFormatter.toDot(body);

        const bodySubdot = new DotSubgraph(`cluster_loop_${node.id}`, bodyDot.statementList)
            .graphAttr("label", `Loop ${node.id}`)
            .graphAttr("style", "dashed")
            .graphAttr("color", "gray");

        this.clusterInfos[node.id] = {
            idPrefix,
            subgraphLabel: bodySubdot.label ?? "",
        };

        return [bodySubdot];
    }

    public ifToDot(node: OperationNode.Class): DotStatement[] {
        const idPrefix = `if${node.id}_`;
        const statements: (DotNode | DotSubgraph | DotEdge)[] = [this.nodeToDot(node)];

        const formatBranch = (
            branch: OnnxGraph.Class,
            branchName: "then" | "else",
            labelTitle: string,
            color: string,
        ): void => {
            const branchFormatter = new OnnxDotFormatter(`${idPrefix}${branchName}_`);
            const branchDot = branchFormatter.toDot(branch);
            const branchCluster = new DotSubgraph(
                `cluster_if_${branchName}_${node.id}`,
                branchDot.statementList,
            )
                .graphAttr("label", `${labelTitle} ${node.id}`)
                .graphAttr("style", "dashed")
                .graphAttr("color", color);

            const firstNode = branch.nodes[0];
            const branchEdge = Dot.edge(
                this.idPrefix + node.id,
                branchFormatter.idPrefix + firstNode.id,
            )
                .attr("lhead", branchCluster.label!)
                .attr("label", branchName)
                .attr("style", "dashed")
                .attr("color", color);

            statements.push(branchCluster, branchEdge);
        };

        formatBranch(node.regions[0], "then", "If-Then", "lime");
        formatBranch(node.regions[1], "else", "If-Else", "red");

        return statements;
    }

    /** Adds dashed edges from captured external inputs into a given node. */
    public externalInputsToDot(node: OperationNode.Class): DotStatement[] {
        const statements: DotEdge[] = [];
        const extInputs = node
            .getInputs()!
            .filter((input) => !node.graph.as(OnnxGraph).hasNode(input.id));

        for (const input of extInputs) {
            const tensor = input.tryAs(TensorNode) ?? input.tryAs(ConstantNode);
            if (tensor === undefined) continue;

            statements.push(
                Dot.edge(tensor.id, this.idPrefix + node.id, {
                    label: OnnxDotFormatter.shapeToLabel(tensor.shape),
                    style: "dashed",
                    color: "gray",
                }),
            );
        }

        return statements;
    }

    public specialNodeToDot(node: BaseNode.Class): DotStatement[] | null {
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

    public override toDot(graph: G): DotGraph {
        this.clusterInfos = {};

        const dot = Dot.graph().graphAttrs(this.getGraphAttrs());
        const nodes = graph.nodes;

        for (const node of nodes.filter((n) => !this.isContained(n))) {
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

        // Add dashed/labeled edges for captured external inputs on data-movement ops
        for (const opNode of graph.getOperationNodes()) {
            if (
                opNode.getInputs() !== undefined &&
                ["Reshape", "Gather", "GatherElements", "Scatter", "ScatterElements"].includes(
                    opNode.type,
                )
            ) {
                const inputNodes = opNode.getInputs()!.filter((n) => !graph.hasNode(n.id));
                for (const ext of inputNodes) {
                    const shape = ext.is(TensorNode)
                        ? ext.as(TensorNode).shape
                        : ext.is(ConstantNode)
                          ? ext.as(ConstantNode).shape
                          : undefined;
                    dot.statements(
                        Dot.edge(ext.id, this.idPrefix + opNode.id, {
                            label: `{${shape?.join(",")}}`,
                        }),
                    );
                }
            }
        }

        return dot;
    }
}
