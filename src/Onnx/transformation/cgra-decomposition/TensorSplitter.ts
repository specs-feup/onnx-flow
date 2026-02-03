import OnnxGraph from "../../OnnxGraph.js";
import TensorNode from "../../TensorNode.js";

export type TensorSplit = {
  splits: TensorNode.Class[];
  columnWise: boolean;
}

/**
 * @class TensorSplitter
 * @brief Manages the splitting of tensors into smaller tensors for CGRA mapping.
 *
 * The TensorSplitter class provides functionality to split tensors, which is
 * useful for converting
 */
export default class TensorSplitter {
  tensorSplits: Map<string, TensorSplit>;
  graph: OnnxGraph.Class;

  constructor(graph: OnnxGraph.Class) {
    this.tensorSplits = new Map();
    this.graph = graph;
  }

  /**
   * @brief Gives a split of the given tensor, creating it if it does not already exist.
   *
   * @param tensor The tensor to be split.
   * @param columnWise Whether to split the tensor column-wise or row-wise.
   * @returns The TensorSplit object containing the splits.
   */
  getSplit(tensor: TensorNode.Class, columnWise: boolean): TensorSplit {
    const existingSplit = this.tensorSplits.get(tensor.id);
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

    this.tensorSplits.set(tensor.id, tensorSplit);
    return tensorSplit;
  }

  /**
   * @brief Removes all unremoved splitted tensors from the graph.
   */
  clearTensors(): void {
    this.tensorSplits.forEach((split, oldTensorId) => {
      if (split.splits.every(split => split.id != oldTensorId) && this.graph.hasNode(oldTensorId)) {
        this.graph.getNodeById(oldTensorId).remove();
      }
    });
  }
}