import type OnnxGraph from "../ir/OnnxGraph.js";
import type { DotGraph, DotStatement } from "@specs-feup/flow/graph/dot/dot";
import Dot, { DotEdge, DotNode } from "@specs-feup/flow/graph/dot/dot";
import type BaseNode from "@specs-feup/flow/graph/BaseNode";
import type BaseEdge from "@specs-feup/flow/graph/BaseEdge";
import Node from "@specs-feup/flow/graph/Node";
import TensorNode from "../ir/TensorNode.js";
import ConstantNode from "../ir/ConstantNode.js";
import OperationNode from "../ir/OperationNode.js";
import {
    readConstIntegerVectorFromTensorNode,
    readTensorData,
    TYPE_SIZE_MAP,
} from "../utils/index.js";
import BaseOnnxDotFormatter from "./BaseOnnxDotFormatter.js";

type ClusterInfo = {
    idPrefix: string;
    subgraphLabel: string;
    sourceMapping: Record<string, string>;
    targetMapping: Record<string, string>;
};

/**
 * DOT formatter specialized for exporting 1D CGRA-compatible ONNX graphs to the
 * DOT specification expected by the STRELA CGRA mapper.
 */
export default class CgraDotFormatter<
    G extends OnnxGraph.Class = OnnxGraph.Class,
> extends BaseOnnxDotFormatter<G> {
    private clusterInfos: Record<string, ClusterInfo> = {};

    static override defaultGetNodeAttrs(node: BaseNode.Class): Record<string, string> {
        const attrs = super.defaultGetNodeAttrs(node);
        delete attrs["shape"];

        node.switch(
            Node.Case(TensorNode, (tn) => {
                const size = (tn.shape as number[]).reduce((a, b) => a * b, 1);

                if (tn.type === "input") {
                    attrs["size"] = size.toString();
                    attrs["stride"] = TYPE_SIZE_MAP[tn.literalType as number]!.toString();
                } else if (tn.type === "output") {
                    attrs["size"] = size.toString();
                }
            }),
            Node.Case(ConstantNode, (cn) => {
                const val = readTensorData(cn);
                attrs["label"] = val!.slice(0, 5).toString();
            }),
            Node.Case(OperationNode, (op) => {
                attrs["label"] = op.type;
                attrs["type"] = op.type.toLowerCase();
                attrs["feedback"] = "0";
            }),
        );

        return attrs;
    }

    static override defaultGetEdgeAttrs(_edge: BaseEdge.Class): Record<string, string> {
        return {};
    }

    static override defaultGetGraphAttrs(): Record<string, string> {
        return super.defaultGetGraphAttrs();
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
            getNodeAttrs ?? CgraDotFormatter.defaultGetNodeAttrs,
            getEdgeAttrs ?? CgraDotFormatter.defaultGetEdgeAttrs,
            getContainer ?? CgraDotFormatter.defaultGetContainer,
            getGraphAttrs ?? CgraDotFormatter.defaultGetGraphAttrs,
        );
    }

    public override createDotEdge(
        sourceId: string,
        targetId: string,
        attrs: Record<string, string> = {},
        escape: boolean = true,
    ): DotEdge {
        let source = escape ? this.idPrefix + sourceId : sourceId;
        let target = escape ? this.idPrefix + targetId : targetId;

        if (sourceId in this.clusterInfos) {
            const sourceCluster = this.clusterInfos[sourceId];
            for (const [prefix, newSource] of Object.entries(sourceCluster.sourceMapping)) {
                if (target.startsWith(prefix)) {
                    source = sourceCluster.idPrefix + newSource;
                    break;
                }
            }
        }

        if (targetId in this.clusterInfos) {
            const targetCluster = this.clusterInfos[targetId];
            for (const [prefix, newTarget] of Object.entries(targetCluster.targetMapping)) {
                if (source.startsWith(prefix)) {
                    target = targetCluster.idPrefix + newTarget;
                    break;
                }
            }
        }

        return Dot.edge(source, target, attrs);
    }

    public ifToDot(_node: OperationNode.Class): DotStatement[] {
        throw new Error("If node conversion to DOT is not implemented.");
    }

    public validateTensorNode(node: TensorNode.Class): void {
        if (node.shape.length > 1) {
            throw new Error("CGRA supports only 1D tensors.");
        }
    }

    public addToDot(node: OperationNode.Class): DotStatement[] {
        return [this.nodeToDot(node).attr("label", "Add").attr("type", "add")];
    }

    public mulToDot(node: OperationNode.Class): DotStatement[] {
        return [this.nodeToDot(node).attr("label", "Mul").attr("type", "mul")];
    }

    public reduceSumToDot(node: OperationNode.Class): DotStatement[] {
        const dotNode = this.nodeToDot(node)
            .attr("label", "Add")
            .attr("type", "add")
            .attr("feedback", "1");
        const loopEdge = this.createDotEdge(node.id, node.id, {});
        return [dotNode, loopEdge];
    }

    public greaterToDot(node: OperationNode.Class): DotStatement[] {
        const inputs = node.getInputs();
        if (!inputs || inputs.length !== 2) {
            throw new Error("Greater node must have two inputs.");
        }

        const isZeroConstVector = (tensor: BaseNode.Class): boolean => {
            if (!tensor.is(ConstantNode)) return false;
            return readConstIntegerVectorFromTensorNode(tensor)!.every((val) => val === 0);
        };

        if (!isZeroConstVector(inputs[1])) {
            throw new Error(
                "All Greater nodes must compare against a zero constant vector on the left-hand side.",
            );
        }

        return [this.nodeToDot(node).attr("label", ">0").attr("type", ">0")];
    }

    public whereToDot(node: OperationNode.Class): DotStatement[] {
        return [this.nodeToDot(node).attr("label", "Mux").attr("type", "mux")];
    }

    public toSkipNode(node: OperationNode.Class): DotStatement[] {
        return [this.nodeToDot(node).attr("type", "skip")];
    }

    public safeNodeToDot(node: BaseNode.Class): DotStatement[] {
        if (node.is(ConstantNode)) {
            return [this.nodeToDot(node.as(ConstantNode))];
        }

        const tensorNode = node.tryAs(TensorNode);
        if (tensorNode !== undefined) {
            this.validateTensorNode(tensorNode);
            if (tensorNode.type === "intermediate") {
                return this.intermediateTensorToDot(tensorNode);
            }
            return [this.nodeToDot(tensorNode)];
        }

        const opNode = node.tryAs(OperationNode);
        if (opNode !== undefined) {
            switch (opNode.type) {
                case "Add":
                    return this.addToDot(opNode);
                case "Mul":
                    return this.mulToDot(opNode);
                case "ReduceSum":
                    return this.reduceSumToDot(opNode);
                case "Where":
                    return this.whereToDot(opNode);
                case "Greater":
                    return this.greaterToDot(opNode);
                case "If":
                    return this.ifToDot(opNode);
                case "Unsqueeze":
                    return this.toSkipNode(opNode);
                case "Concat":
                    return [this.nodeToDot(opNode)];
                default:
                    throw new Error(
                        `Operation node of type "${opNode.type}" is not supported in CGRA DOT formatter.`,
                    );
            }
        }

        return [];
    }

    public override toDot(graph: G): DotGraph {
        this.clusterInfos = {};

        const dot = Dot.graph().graphAttrs(this.getGraphAttrs());
        const dotNodes: DotNode[] = [];
        const dotEdges = new Map<string, DotEdge>();

        const addNodeStatements = (...statements: (DotStatement | DotNode | DotEdge)[]): void => {
            const edges = statements.filter((s) => s instanceof DotEdge) as DotEdge[];
            const nodes = statements.filter((s) => s instanceof DotNode) as DotNode[];
            const others = statements.filter(
                (s) => !(s instanceof DotNode) && !(s instanceof DotEdge),
            );

            dotNodes.push(...nodes);
            for (const edge of edges) {
                dotEdges.set(`${edge.source as string}:${edge.target as string}`, edge);
            }
            dot.statements(...others);
        };

        for (const node of graph.nodes.filter((n) => !this.isContained(n))) {
            if (this.isContainer(node)) {
                addNodeStatements(this.clusterNodeToDot(node));
            } else {
                addNodeStatements(...this.safeNodeToDot(node));
            }
        }

        for (const edge of graph.edges) {
            const statements = this.specialEdgeToDot(edge);
            if (statements !== null) {
                addNodeStatements(...statements);
                continue;
            }
            addNodeStatements(this.edgeToDot(edge));
        }

        // Collapse "skip" nodes (e.g., Unsqueeze) by bridging incoming edges to outgoing targets
        const nextTargets = new Map<string, string[]>();
        for (const node of dotNodes) {
            if (node.attrList["type"] === "skip") {
                nextTargets.set(node.id as string, []);
            }
        }

        for (const edge of dotEdges.values()) {
            nextTargets.get(edge.source as string)?.push(edge.target as string);
        }

        for (const edge of dotEdges.values()) {
            const targetSkip = nextTargets.get(edge.target as string);
            if (targetSkip !== undefined) {
                for (const nextTarget of targetSkip) {
                    const newEdge = this.createDotEdge(
                        edge.source as string,
                        nextTarget,
                        edge.attrList,
                        false,
                    );
                    dotEdges.set(
                        `${newEdge.source as string}:${newEdge.target as string}`,
                        newEdge,
                    );
                }
            }
        }

        dot.statements(...dotNodes.filter((node) => node.attrList["type"] !== "skip"));
        for (const edge of dotEdges.values()) {
            if (
                !nextTargets.has(edge.source as string) &&
                !nextTargets.has(edge.target as string)
            ) {
                dot.statements(edge);
            }
        }

        return dot;
    }
}
