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
import { typeSizeMap } from "../Utils.js";

type ClusterInfo = {
  idPrefix: string;
  subgraphLabel: string;
  sourceMapping: Record<string, string>;
  targetMapping: Record<string, string>;
};

export default class CgraDotFormatter<
  G extends OnnxGraph.Class = OnnxGraph.Class,
> extends DefaultDotFormatter<G> {
  private idPrefix: string;
  private clusterInfos: Record<string, ClusterInfo> = {};

  static defaultGetNodeAttrs(node: BaseNode.Class): Record<string, string> {
    const attrs = super.defaultGetNodeAttrs(node);
    delete attrs.shape; // Remove default shape attribute

    node.switch(
      Node.Case(TensorNode, (node) => {
        // In case the node has more than one dimension (which it
        // shouldn't!), use the combined element count as the size
        const size = (node.shape as number[]).reduce((a, b) => a * b, 1);

        if (node.type === "input") {
          attrs.size = size.toString();
          attrs.stride = typeSizeMap[node.literalType as number]!.toString();
        } else if (node.type === "output") {
          attrs.size = size.toString();
        }
      }),
      Node.Case(VariableNode, (node) => {
        attrs.label = node.name;
        // attrs.address = '0';
        // attrs.size = '1';
      }),
      Node.Case(ConstantNode, (node) => {
        attrs.label = node.value.toString();
        // attrs.size = node;
      }),
      Node.Case(OperationNode, (node) => {
        attrs.label = node.type;
        attrs.type = node.type.toLowerCase();

        attrs.feedback = "0";
        // attrs.constant = '0';
        // attrs.constant_fu_input = '0';
        // attrs.initial_value = '0';
        // attrs.initial_valid = '0';
        // attrs.delay_value = '0';
      }),
    );

    return attrs;
  }

  static shapeToLabel(shape: (number | string)[]): string {
    const shapeString = `{${shape.join(",")}}`;
    return shapeString === "{}" ? "sc" : shapeString;
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
    getGraphAttrs?: () => Record<string, string>,
  ) {
    getNodeAttrs ??= CgraDotFormatter.defaultGetNodeAttrs;
    getEdgeAttrs ??= CgraDotFormatter.defaultGetEdgeAttrs;
    getContainer ??= CgraDotFormatter.defaultGetContainer; // Method not implemented, add if needed
    getGraphAttrs ??= CgraDotFormatter.defaultGetGraphAttrs;

    super(getNodeAttrs, getEdgeAttrs, getContainer, getGraphAttrs);

    this.idPrefix = idPrefix;
  }

  override nodeToDot(node: BaseNode.Class): DotNode {
    const id = this.idPrefix + node.id;
    const attrs = this.getNodeAttrs(node);

    return Dot.node(id, attrs);
  }

  createDotEdge(
    sourceId: string,
    targetId: string,
    attrs: Record<string, string> = {},
    escape: boolean = true,
  ): DotEdge {
    let source = escape ? this.idPrefix + sourceId : sourceId;
    let target = escape ? this.idPrefix + targetId : targetId;

    if (sourceId in this.clusterInfos) {
      const sourceCluster = this.clusterInfos[sourceId];

      // Map all references to the cluster to an appropriate inner node
      for (const [prefix, newSource] of Object.entries(
        sourceCluster.sourceMapping,
      )) {
        if (target.startsWith(prefix)) {
          source = sourceCluster.idPrefix + newSource;
          break;
        }
      }
    }

    if (targetId in this.clusterInfos) {
      const targetCluster = this.clusterInfos[targetId];

      // Map all references to the cluster to an appropriate inner node
      for (const [prefix, newTarget] of Object.entries(
        targetCluster.targetMapping,
      )) {
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

  ifToDot(node: OperationNode.Class): DotStatement[] {
    // Implementing the transformation of If is possible using a "branch" and
    // a "merge" node, but since it is not generated for the tested examples
    // and the representation in ONNX is non-trivial, we leave it unimplemented for now
    throw new Error("If node conversion to DOT is not implemented.");
  }

  /**
   * @brief Tries to turn a node as an intermediate tensor.
   *
   * @param node The node to convert.
   * @returns The intermediate tensor node if compatible, otherwise undefined.
   */
  tryAsIntermediateTensor(node: BaseNode.Class): TensorNode.Class | undefined {
    const tensorNode = node.tryAs(TensorNode);
    if (tensorNode === undefined) return undefined;

    if (!["intermediate", "constant"].includes(tensorNode.type))
      return undefined;

    return tensorNode;
  }

  validateTensorNode(node: TensorNode.Class): void {
    // TODO(Process-ing): Remove this bypass, once split-concat skip is implemented
    if (node.type === "intermediate") {
      return;
    }

    if (node.shape.length > 1) {
      throw new Error(
        "CGRA supports only 1D tensors.",
      );
    }
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

  addToDot(node: OperationNode.Class): DotStatement[] {
    const dotNode = this.nodeToDot(node)
      .attr("label", "Add")
      .attr("type", "add");

    return [dotNode];
  }

  mulToDot(node: OperationNode.Class): DotStatement[] {
    const dotNode = this.nodeToDot(node)
      .attr("label", "Mul")
      .attr("type", "mul");

    return [dotNode];
  }

  reduceSumToDot(node: OperationNode.Class): DotStatement[] {
    const dotNode = this.nodeToDot(node)
      .attr("label", "Add")
      .attr("type", "add")
      .attr("feedback", "1");

    const loopEdge = this.createDotEdge(node.id, node.id, {});

    return [dotNode, loopEdge];
  }

  greaterToDot(node: OperationNode.Class): DotStatement[] {
    if (node.getInputs().length !== 2) {
      throw new Error("Greater node must have two inputs.");
    }

    const isZeroConstVector = (tensor: TensorNode.Class): boolean => {
      if (tensor.type !== "constant") return false;

      // TODO(Process-ing): STRELA only supports operations on integers, but add support to other types if needed
      return tensor.constantValue.int32Data?.every((val) => val === 0);
    }

    const secondInput = node.getInputs()[1].as(TensorNode);
    if (!isZeroConstVector(secondInput)) {
      throw new Error("All Greater nodes must compare against a zero constant vector on the left-hand side.");
    }

    const dotNode = this.nodeToDot(node)
      .attr("label", ">0")
      .attr("type", ">0");

    return [dotNode];
  }

  whereToDot(node: OperationNode.Class): DotStatement[] {
    const dotNode = this.nodeToDot(node)
      .attr("label", "Mux")
      .attr("type", "mux");

    return [dotNode];
  }

  toSkipNode(node: OperationNode.Class): DotStatement[] {
    const dotNode = this.nodeToDot(node)
      .attr("type", "skip");

    return [dotNode];
  }

  safeNodeToDot(node: BaseNode.Class): DotStatement[] {
    const tensorNode = node.tryAs(TensorNode);
    if (tensorNode !== undefined) {
      this.validateTensorNode(tensorNode);

      if (["intermediate", "constant"].includes(tensorNode.type)) {
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
        case "Squeeze":
        case "Unsqueeze":
        case "Reshape":
          return this.toSkipNode(opNode);

        // Temporary case to visualize Split and Concat nodes
        case "Split":
        case "Concat":
          return [this.nodeToDot(opNode)];

        default:
          throw new Error(`Operation node of type "${opNode.type}" is not supported in CGRA DOT formatter.`);
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
    const dotNodes: DotNode[] = [];
    const dotEdges: Map<string, DotEdge> = new Map<string, DotEdge>();

    function addNodeStatements(...statements: DotStatement[]) {
      const edges =
        (statements?.filter((s) => s instanceof DotEdge) as DotEdge[]) || [];
      const nodes =
        (statements?.filter((s) => s instanceof DotNode) as DotNode[]) || [];
      const others =
        statements?.filter(
          (s) => !(s instanceof DotNode) && !(s instanceof DotEdge),
        ) || [];

      dotNodes.push(...nodes);
      edges.forEach((edge) =>
        dotEdges.set(
          ((edge.source as string) + ":" + edge.target) as string,
          edge,
        ),
      );
      dot.statements(...others);
    }

    for (const node of nodes.filter((node) => !this.isContained(node))) {
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

    const nextTargets = new Map<string, string[]>();

    for (const node of dotNodes) {
      if (node.attrList.type === "skip") {
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
            ((newEdge.source as string) + ":" + newEdge.target) as string,
            newEdge,
          );
        }
      }
    }

    dot.statements(...dotNodes.filter((node) => node.attrList.type !== "skip"));

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
