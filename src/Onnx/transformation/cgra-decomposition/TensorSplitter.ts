import OnnxGraph from "../../OnnxGraph.js";
import TensorNode from "../../TensorNode.js";

export type TensorSplit = {
  splits: TensorNode.Class[];
  columnWise: boolean;
}

export default class TensorSplitter {
  tensorSplits: Map<TensorNode.Class, TensorSplit>;
  graph: OnnxGraph.Class;

  constructor(graph: OnnxGraph.Class) {
    this.tensorSplits = new Map();
    this.graph = graph;
  }

  getSplit(tensor: TensorNode.Class, columnWise: boolean): TensorSplit {
    const existingSplit = this.tensorSplits.get(tensor);
    if (existingSplit !== undefined) {
      if (existingSplit.columnWise !== columnWise) {
        throw new Error("Tensor has already been split in a different orientation.");
      }

      return existingSplit;
    }

    const [numSplits, splitShape] = columnWise ? [tensor.shape[1] as number, [tensor.shape[0]]] : [tensor.shape[0] as number, [tensor.shape[1]]];
    const splitBuilder = new TensorNode.Builder(tensor.literalType, splitShape, tensor.type);

    const splits = numSplits == 1
      ? [tensor.init(splitBuilder).as(TensorNode)]
      : Array.from({ length: numSplits }, (_, i) => {
        const split = this.graph.addNode(`${tensor.id}${i}`, tensor.parent).init(splitBuilder).as(TensorNode);
        return split;
      });

    const tensorSplit: TensorSplit = {
      splits: splits,
      columnWise,
    };

    this.tensorSplits.set(tensor, tensorSplit);
    return tensorSplit;
  }

  /**
   * @brief Removes all unremoved splitted tensors from the graph.
   */
  clearTensors(): void {
    this.tensorSplits.forEach((split, oldTensor) => {
      if (split.splits.every(split => split.id != oldTensor.id) && this.graph.hasNode(oldTensor.id)) {
        oldTensor.remove();
      }
    });
  }
}