import BaseNode from "@specs-feup/flow/graph/BaseNode";
import Node from "@specs-feup/flow/graph/Node";
import { EdgeCollection } from "@specs-feup/flow/graph/EdgeCollection";
import OnnxEdge from "./OnnxEdge.js";
import OnnxGraph from "./OnnxGraph.js";

namespace OperationNode {

  export const TAG = "__specs-onnx__operation_node";
  export const VERSION = "2";

  export class Class<
    D extends Data = Data,
    S extends ScratchData = ScratchData,
  > extends BaseNode.Class<D, S> {

    get type(): string {
      return this.data[TAG].type;
    }

    set type(newType: string) {
      this.data[TAG].type = newType;
    }

    get attributes(): Record<string, any> {
      return this.data[TAG].attributes || {};
    }

    set attributes(attrs: Record<string, any>) {
      this.data[TAG].attributes = attrs;
    }

    setAttributes(attrs: Record<string, any>): void {
      this.attributes = attrs;
    }

    getAttributes(): Record<string, any> {
      return this.attributes;
    }

    get getIncomers(): EdgeCollection<OnnxEdge.Class> {
      return this.incomers.filterIs(OnnxEdge);
    }

    get getOutgoers(): EdgeCollection<OnnxEdge.Class> {
      return this.outgoers.filterIs(OnnxEdge);
    }

    getInputs(): BaseNode.Class[] | undefined {
      return this.data[TAG].inputs;
    }

    getBodySubgraph(): OnnxGraph.Class | undefined {
      return this.data[TAG].bodyGraph ?? this.data[TAG].subgraphs?.body;
    }

    getSubgraph(name: string): OnnxGraph.Class | undefined {
      return this.data[TAG].subgraphs?.[name];
    }

    getThenBranch(): OnnxGraph.Class | undefined {
      return this.getSubgraph("thenBranch");
    }

    getElseBranch(): OnnxGraph.Class | undefined {
      return this.getSubgraph("elseBranch");
    }

    getSubgraphs(): Record<string, OnnxGraph.Class> {
      return this.data[TAG].subgraphs ?? {};
    }
  }

  export class Builder implements Node.Builder<Data, ScratchData> {
    private type: string;
    private attributes?: Record<string, any>;
    private inputs?: BaseNode.Class[];
    private bodyGraph?: OnnxGraph.Class;
    private subgraphs?: Record<string, OnnxGraph.Class>;

    constructor(
      type: string,
      inputs?: BaseNode.Class[],
      attributes?: Record<string, any>,
      bodyGraphOrSubgraphs?: OnnxGraph.Class | Record<string, OnnxGraph.Class>
    ) {
      this.type = type;
      this.attributes = attributes;
      this.inputs = inputs;

      if (bodyGraphOrSubgraphs instanceof OnnxGraph.Class) {
        this.bodyGraph = bodyGraphOrSubgraphs;
      } else if (bodyGraphOrSubgraphs && typeof bodyGraphOrSubgraphs === "object") {
        this.subgraphs = bodyGraphOrSubgraphs;
      }
    }

    buildData(data: BaseNode.Data): Data {
      return {
        ...data,
        [TAG]: {
          version: VERSION,
          type: this.type,
          inputs: this.inputs || [],
          attributes: this.attributes || {},
          bodyGraph: this.bodyGraph,
          subgraphs: this.subgraphs,
        },
      };
    }

    buildScratchData(scratchData: BaseNode.ScratchData): ScratchData {
      return {
        ...scratchData,
      };
    }
  }

  export const TypeGuard = Node.TagTypeGuard<Data, ScratchData>(TAG, VERSION);

  export interface Data extends BaseNode.Data {
    [TAG]: {
      version: typeof VERSION;
      type: string;
      inputs?: BaseNode.Class[];
      attributes?: Record<string, any>;
      bodyGraph?: OnnxGraph.Class;
      subgraphs?: Record<string, OnnxGraph.Class>;
    };
  }

  export interface ScratchData extends BaseNode.ScratchData {}

}

export default OperationNode;
