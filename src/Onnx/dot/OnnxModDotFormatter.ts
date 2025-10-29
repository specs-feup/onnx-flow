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

type ClusterInfo = {
    idPrefix: string;
    subgraphLabel: string;
    sourceMapping: Record<string, string>,
    targetMapping: Record<string, string>,
};

export default class OnnxDotFormatter<
    G extends OnnxGraph.Class = OnnxGraph.Class,
> extends DefaultDotFormatter<G> {
    private idPrefix: string;
    private clusterInfos: Record<string, ClusterInfo> = {};

    static defaultGetNodeAttrs(node: BaseNode.Class): Record<string, string> {
        const attrs = super.defaultGetNodeAttrs(node);
        delete attrs.shape;  // Remove default shape attribute

        node.switch(
            Node.Case(TensorNode, node => {
                // attrs.address = node.address.toString();
                // attrs.stride = node.literalType.toString();
                // attrs.stride = '4';
                // attrs.size = node.shape[0]?.toString() ?? '1';
            }),
            Node.Case(VariableNode, node => {
                attrs.label = node.name;
                // attrs.address = '0';
                // attrs.size = '1';
            }),
            Node.Case(ConstantNode, node => {
                attrs.label = node.value.toString();
                // attrs.size = node;
            }),
            Node.Case(OperationNode, node => {
                // attrs.label = node.type;

                // TODO(Process-ing): Improve on this
                attrs.label = node.type != 'ReduceSum' ? node.type : 'Add';

                // attrs.feedback = '0';
                // attrs.constant = '0';
                // attrs.constant_fu_input = '0';
                // attrs.initial_value = '0';
                // attrs.initial_valid = '0';
                // attrs.delay_value = '0';
            }),
        );

        return attrs;
    }

    static shapeToLabel(shape: (number | String)[]): string {
        const shapeString = `{${shape.join(',')}}`;
        return shapeString === '{}' ? 'sc' : shapeString;
    }

    static defaultGetEdgeAttrs(edge: BaseEdge.Class): Record<string, string> {
        const attrs = {};

        return attrs;
    }

    static defaultGetGraphAttrs(): Record<string, string> {
        const attrs = super.defaultGetGraphAttrs();

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

    override nodeToDot(node: BaseNode.Class): DotNode {
        const id = this.idPrefix + node.id;
        const attrs = this.getNodeAttrs(node);

        return Dot.node(id, attrs);
    }

    createDotEdge(sourceId: string, targetId: string, attrs: Record<string, string> = {}, escape: boolean = true): DotEdge {
        let source = escape ? this.idPrefix + sourceId : sourceId;
        let target = escape ? this.idPrefix + targetId : targetId;

        if (sourceId in this.clusterInfos) {
            const sourceCluster = this.clusterInfos[sourceId];

            // attrs.ltail = sourceCluster.subgraphLabel;
            // source = sourceCluster.idPrefix + source;

            // Map all references to the cluster to an appropriate inner node
            for (const [prefix, newSource] of Object.entries(sourceCluster.sourceMapping)) {
                if (target.startsWith(prefix)) {
                    source = sourceCluster.idPrefix + newSource;
                    break;
                }
            }
        }

        if (targetId in this.clusterInfos) {
            const targetCluster = this.clusterInfos[targetId];

            // attrs.lhead = targetCluster.subgraphLabel;
            // target = targetCluster.idPrefix + target;

            // Map all references to the cluster to an appropriate inner node
            for (const [prefix, newTarget] of Object.entries(targetCluster.targetMapping)) {
                if (source.startsWith(prefix)) {
                    target = targetCluster.idPrefix + newTarget;
                    break;
                }
            }
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

        // Uncomment this if you need the loop as a node as well in the DOT
        // const loopNode = this.nodeToDot(node);
        // loopNode.attr('label', `Loop ${node.id}`);
        // statements.push(loopNode);

        const body = node.getBodySubgraph();
        if (body !== undefined) {
            const subFormatter = new OnnxDotFormatter(idPrefix);
            const bodyDot = subFormatter.toDot(body);

            const bodySubdot = new DotSubgraph(`cluster_loop_${node.id}`, bodyDot.statementList)
                .graphAttr('label', `Loop ${node.id}`);

            const carry = body.nodes.filter((node, _, __) => node.id.startsWith('carry')).first();
            const carryOut = body.nodes.filter((node, _, __) => node.id.startsWith('carry_out')).first();

            statements.push(...bodySubdot.statementList.filter(s => s instanceof DotNode || s instanceof DotEdge));

            // TODO(Process-ing): Only for demonstration, remove/improve later
            statements.push(this.createDotEdge(idPrefix + carryOut.id, idPrefix + carry.id));

            this.clusterInfos[node.id] = {
                idPrefix,
                subgraphLabel: bodySubdot.label,
                targetMapping: { 'init_carry': carry.id },
                sourceMapping: { '': carryOut.id },
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
                .graphAttr('label', `If-Then ${node.id}`);

            const thenEdge = Dot.edge(this.idPrefix + node.id, thenIdPrefix + '0')  // Linked to first node in subgraph
                .attr('lhead', thenGraph.label)
                .attr('label', 'then');

            statements.push(thenGraph);
            statements.push(thenEdge);
        }

        const elseBranch = node.getElseBranch();
        if (elseBranch !== undefined) {
            const elseIdPrefix = `${idPrefix}else_`;
            const elseFormatter = new OnnxDotFormatter(elseIdPrefix);
            const elseDot = elseFormatter.toDot(elseBranch);

            const elseGraph = new DotSubgraph(`cluster_if_else_${node.id}`, elseDot.statementList)
                .graphAttr('label', `If-Else ${node.id}`);

            const elseEdge = Dot.edge(this.idPrefix + node.id, elseIdPrefix + '0')  // Linked to first node in subgraph
                .attr('lhead', elseGraph.label)
                .attr('label', 'else');

            statements.push(elseGraph);
            statements.push(elseEdge);
        }

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
     * @brief Adds edges from external inputs to a given node.
     *
     * @param node The node to which to add external input edges.
     * @returns The resulting DOT statements.
     */
    externalInputsToDot(node: OperationNode.Class): DotStatement[] {
        const statements = [];

        const extInputs = node.getInputs().filter(input => !node.graph.as(OnnxGraph).hasNode(input.id));

        for (const input of extInputs) {
            const tensor = input.tryAs(TensorNode);
            if (tensor === undefined)
                continue;

            const targetId = this.idPrefix + node.id;
            const attrs = {};

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
        const intermediateTensorNode = this.tryAsIntermediateTensor(node);
        if (intermediateTensorNode !== undefined) {
            return this.intermediateTensorToDot(intermediateTensorNode);
        }

        const opNode = node.tryAs(OperationNode);
        if (opNode !== undefined) {
            switch (opNode.type) {
                case 'Loop':
                    return this.loopToDot(opNode);
                case 'If':
                    return this.ifToDot(opNode);
                case 'Gather':
                case 'Scatter':
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
        if (this.tryAsIntermediateTensor(edge.source) !== undefined
            || this.tryAsIntermediateTensor(edge.target) !== undefined) {
            return [];
        }

        return null;
    }

    toIgnore(node: DotNode): boolean {
        // TODO(Process-ing): Make these conditions robust to weird input names
        if (['Gather', 'Scatter', 'ScatterElements', 'Squeeze', 'Unsqueeze'].includes(node.attrList.label)) {
            return true;
        }

        const id = node.id as string;
        const label = node.attrList.label as string;
        if (id.startsWith('loop') && (id.includes('_cond_in') || id.includes('_cond_out') || label === 'Identity')) {
            return true;
        }

        // TODO(Process-ing): Only for demonstration, remove/improve later
        if (label.startsWith('init_carry')) {
            return true;
        }

        return false;
    }

    override toDot(graph: G): DotGraph {
        // Reset state
        this.clusterInfos = {};

        const dot = Dot.graph().graphAttrs(this.getGraphAttrs());
        const nodes = graph.nodes;
        const dotNodes: DotNode[] = [];
        const dotEdges: Map<string, DotEdge> = new Map<string, DotEdge>();

        function addNodeStatements(...statements: DotStatement[]) {
            const edges = statements?.filter(s => s instanceof DotEdge) as DotEdge[] || [];
            const nodes = statements?.filter(s => s instanceof DotNode) as DotNode[] || [];
            const others = statements?.filter(s => !(s instanceof DotNode) && !(s instanceof DotEdge)) || [];

            dotNodes.push(...nodes);
            edges.forEach(edge => dotEdges.set(edge.source as string + ':' + edge.target as string, edge));
            dot.statements(...others);
        }

        for (const node of nodes.filter(node => !this.isContained(node))) {
            const statements = this.specialNodeToDot(node);
            if (statements !== null) {
                addNodeStatements(...statements);
                continue;
            }

            if (this.isContainer(node)) {
                addNodeStatements(this.clusterNodeToDot(node));
            } else {
                addNodeStatements(this.nodeToDot(node));
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

        const nextTargets = new Map<string, string[]>();

        for (const node of dotNodes) {
            if (this.toIgnore(node)) {
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
                        false
                    );

                    dotEdges.set(newEdge.source as string + ':' + newEdge.target as string, newEdge);
                }
            }
        }

        dot.statements(...dotNodes.filter(node => !this.toIgnore(node)));

        for (const edge of dotEdges.values()) {
            if (!nextTargets.has(edge.source as string) && !nextTargets.has(edge.target as string)) {
                dot.statements(edge);
            }
        }

        return dot;
    }
}
