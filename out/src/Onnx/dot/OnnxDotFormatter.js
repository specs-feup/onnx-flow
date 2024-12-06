import DefaultDotFormatter from "@specs-feup/flow/graph/dot/DefaultDotFormatter";
import TensorNode from "../TensorNode.js";
import OperationNode from "../OperationNode.js";
import ConstantNode from "../ConstantNode.js";
import VariableNode from "../VariableNode.js";
import OnnxEdge from "../OnnxEdge.js";
import Edge from "@specs-feup/flow/graph/Edge";
import Node from "@specs-feup/flow/graph/Node";
export default class OnnxDotFormatter extends DefaultDotFormatter {
    /**
     * @param node The node to get the attributes for.
     * @returns The attributes of the node.
     */
    static defaultGetNodeAttrs(node) {
        const result = { label: node.id, shape: "box" };
        node.switch(Node.Case(TensorNode, (n) => {
            if (n.type === "input") {
                result.color = "#00FF00"; // Green
                result.shape = "ellipse";
            }
            else if (n.type === "output") {
                result.color = "#FF0000"; // Red
                result.shape = "ellipse";
            }
            else if (n.type === "index" || n.type === "index_aux") {
                result.color = "#FF00FF"; // Red
                result.shape = "ellipse";
            }
        }), Node.Case(VariableNode, (n) => {
            if (n.type === "input") {
                result.color = "#00FF00"; // Green
            }
            else if (n.type === "output") {
                result.color = "#FF0000"; // Red
            }
            result.shape = "ellipse";
            result.label = n.name;
        }), Node.Case(ConstantNode, (n) => {
            result.label = n.value.toString();
            result.color = "#A52A2A"; // Brown
            result.shape = "ellipse";
        }), Node.Case(OperationNode, (n) => {
            result.label = n.type;
            result.color = "#0000FF"; // Blue
        }));
        return result;
    }
    /**
     * @param edge The edge to get the attributes for.
     * @returns The attributes of the edge.
     */
    static defaultGetEdgeAttrs(edge) {
        const result = {};
        edge.switch(Edge.Case(OnnxEdge, (e) => {
            result.label = `{${e.shape.join(',')}}`;
        }));
        return result;
    }
    /**
     * Creates a new ONNX DOT formatter.
     */
    constructor() {
        super(OnnxDotFormatter.defaultGetNodeAttrs, OnnxDotFormatter.defaultGetEdgeAttrs, DefaultDotFormatter.defaultGetContainer);
    }
}
//# sourceMappingURL=OnnxDotFormatter.js.map