import OnnxGraph from "../../OnnxGraph.js";
import OperationNode from "../../OperationNode.js";

export type Handler = (graph: OnnxGraph.Class, op: OperationNode.Class) => boolean;
// Registry by op type
export type HandlersRegistry = Record<string, Handler>;

export interface PreDecomposeOptions {
  maxPasses?: number;
  handlers?: HandlersRegistry;
}
