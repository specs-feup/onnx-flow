import DefaultDotFormatter from "@specs-feup/flow/graph/dot/DefaultDotFormatter";
import BaseEdge from "@specs-feup/flow/graph/BaseEdge";
import BaseNode from "@specs-feup/flow/graph/BaseNode";
import OnnxGraph from "../OnnxGraph.js";
export default class OnnxDotFormatter<G extends OnnxGraph.Class = OnnxGraph.Class> extends DefaultDotFormatter<G> {
    /**
     * @param node The node to get the attributes for.
     * @returns The attributes of the node.
     */
    static defaultGetNodeAttrs(node: BaseNode.Class): Record<string, string>;
    /**
     * @param edge The edge to get the attributes for.
     * @returns The attributes of the edge.
     */
    static defaultGetEdgeAttrs(edge: BaseEdge.Class): Record<string, string>;
    /**
     * Creates a new ONNX DOT formatter.
     */
    constructor();
}
//# sourceMappingURL=OnnxDotFormatter.d.ts.map