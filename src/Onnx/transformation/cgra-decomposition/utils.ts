import TensorNode from "../../TensorNode.js";

/**
 * @brief Checks if two tensor nodes have the same shape.
 *
 * @param tensor1 The first tensor node to compare.
 * @param tensor2 The second tensor node to compare.
 * @returns True if the shapes are equal, false otherwise.
 */
export function shapesEqual(tensor1: TensorNode.Class, tensor2: TensorNode.Class): boolean {
  if (tensor1.shape.length !== tensor2.shape.length) {
    return false;
  }

  for (let i = 0; i < tensor1.shape.length; i++) {
    if (tensor1.shape[i] !== tensor2.shape[i]) {
      return false;
    }
  }

  return true;
}
