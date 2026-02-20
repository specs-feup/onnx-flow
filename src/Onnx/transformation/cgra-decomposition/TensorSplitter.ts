import type OnnxGraph from "../../OnnxGraph.js";
import TensorNode from "../../TensorNode.js";
import ConstantNode from "../../ConstantNode.js";
import type { TensorProto } from "../../OnnxTypes.js";
import { makeTensorProto, readTensorData } from "../../Utils.js";

export type TensorSplit = {
    // Phase 3: splits can contain either TensorNodes or ConstantNodes
    splits: (TensorNode.Class | ConstantNode.Class)[];
    columnWise: boolean;
};

/**
 * @class TensorSplitter
 * @brief Manages the splitting of tensors into smaller tensors for CGRA mapping.
 */
export default class TensorSplitter {
    tensorSplits: Map<string, TensorSplit>;
    graph: OnnxGraph.Class;

    constructor(graph: OnnxGraph.Class) {
        this.tensorSplits = new Map();
        this.graph = graph;
    }

    /**
     * @brief Slices the underlying data of a ConstantNode.
     */
    private splitConstantData(
        node: ConstantNode.Class,
        splitIdx: number,
        numSplits: number,
        columnWise: boolean,
    ): TensorProto {
        const data = readTensorData(node) ?? [];
        const shape = node.shape as number[];
        const [rows, cols] = shape.length === 2 ? shape : [1, shape[0]];

        let subData: number[] = [];
        let newShape: number[] = [];

        if (columnWise) {
            // Split [R, C] into C tensors of [R]
            // Each split gets one column
            for (let r = 0; r < rows; r++) {
                subData.push(data[r * cols + splitIdx]);
            }
            newShape = [rows];
        } else {
            // Split [R, C] into R tensors of [C]
            // Each split gets one row
            const start = splitIdx * cols;
            subData = data.slice(start, start + cols);
            newShape = [cols];
        }

        return makeTensorProto(node.literalType, newShape, subData);
    }

    /**
     * @brief Gives a split of the given tensor, creating it if it does not already exist.
     * Supports both TensorNode and ConstantNode.
     */
    getSplit(tensor: TensorNode.Class | ConstantNode.Class, columnWise: boolean): TensorSplit {
        const existingSplit = this.tensorSplits.get(tensor.id);
        if (existingSplit !== undefined) {
            if (existingSplit.columnWise !== columnWise) {
                throw new Error(`Tensor ${tensor.id} already split in a different orientation.`);
            }
            return existingSplit;
        }

        const shape = tensor.shape as number[];
        // Determine number of resulting nodes and their internal shapes
        const [numSplits, splitShape] = columnWise
            ? [shape[1] ?? 1, [shape[0]]]
            : [shape[0] ?? 1, [shape[1]]];

        const splits: (TensorNode.Class | ConstantNode.Class)[] = [];

        for (let i = 0; i < numSplits; i++) {
            const splitId = `${tensor.id}_split${i}`;

            if (tensor.is(ConstantNode)) {
                // Phase 3: Create a new ConstantNode with sliced data
                const slicedProto = this.splitConstantData(
                    tensor.as(ConstantNode),
                    i,
                    numSplits,
                    columnWise,
                );
                const split = this.graph
                    .addNode(splitId, tensor.parent)
                    .init(new ConstantNode.Builder(slicedProto))
                    .as(ConstantNode);
                splits.push(split);
            } else {
                // Handle standard TensorNode (intermediate/input/output)
                const tNode = tensor.as(TensorNode);
                const splitBuilder = new TensorNode.Builder(
                    tNode.literalType,
                    splitShape,
                    tNode.type === "input" ? "input" : "intermediate",
                );
                const split = this.graph
                    .addNode(splitId, tensor.parent)
                    .init(splitBuilder)
                    .as(TensorNode);
                splits.push(split);
            }
        }

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
            // Only remove if the original node wasn't reused as one of the splits
            if (
                split.splits.every((s) => s.id !== oldTensorId) &&
                this.graph.hasNode(oldTensorId)
            ) {
                this.graph.getNodeById(oldTensorId).remove();
            }
        });
    }
}
