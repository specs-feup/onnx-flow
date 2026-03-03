import { makeTensorProto } from "@specs-feup/onnx-flow/Onnx/Utils";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { ConcreteValueNode, ValueNode } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";

/**
 * Gathers a scalar from a tensor using an iterator, respecting ONNX broadcasting rules.
 * Calculates: FlatIndex = sum( (iterator / OutStride[i] % OutDim[i]) * InStride[i] )
 * where InStride[i] is 0 if the input dimension is 1 (broadcasting).
 */
export function buildBroadcastGather(
    builder: GraphBuilder,
    input: ConcreteValueNode,
    iterator: ValueNode,
    outShape: number[],
    inShape: number[],
    tag: string,
): ConcreteValueNode {
    if (inShape.length === 0) return input; // True scalar

    const rankOut = outShape.length;
    const rankIn = inShape.length;

    // 1. Compute Output Strides
    const outStrides = new Array(rankOut);
    let acc = 1;
    for (let i = rankOut - 1; i >= 0; i--) {
        outStrides[i] = acc;
        acc *= outShape[i];
    }

    // 2. Compute Input Strides
    const inStrides = new Array(rankIn);
    acc = 1;
    for (let i = rankIn - 1; i >= 0; i--) {
        inStrides[i] = acc;
        acc *= inShape[i];
    }

    // 3. Build the index calculation mathematically
    let flatInIdx: ConcreteValueNode = builder.createConstant(
        `${tag}_zero`,
        makeTensorProto(DataType.INT64, [], [0]),
    );

    for (let i = 0; i < rankIn; i++) {
        const inDim = inShape[i];
        if (inDim === 1) continue; // Broadcast dim: index is always 0, contributes 0 to flat offset

        const outPos = rankOut - rankIn + i; // Right-aligned matching
        const outStride = outStrides[outPos];
        const outDim = outShape[outPos];

        // Calculate the N-dimensional coordinate for this axis
        let dimIdx = iterator;
        if (outStride > 1) {
            const outStrideConst = builder.createConstant(
                `${tag}_ostride_${i}`,
                makeTensorProto(DataType.INT64, [], [outStride]),
            );
            dimIdx = builder.createOp("Div", [dimIdx, outStrideConst])[0];
        }

        const outDimConst = builder.createConstant(
            `${tag}_odim_${i}`,
            makeTensorProto(DataType.INT64, [], [outDim]),
        );
        dimIdx = builder.createOp("Mod", [dimIdx, outDimConst])[0];

        // Multiply by the input's stride and add to total flat index
        const inStrideConst = builder.createConstant(
            `${tag}_istride_${i}`,
            makeTensorProto(DataType.INT64, [], [inStrides[i]]),
        );
        const offset = builder.createOp("Mul", [dimIdx, inStrideConst])[0];
        flatInIdx = builder.createOp("Add", [flatInIdx, offset])[0];
    }

    // 4. Flatten the input and Gather
    const flatShape = builder.createConstant(
        `${tag}_flat_shape`,
        makeTensorProto(DataType.INT64, [1], [-1]),
    );
    const flatInput = builder.createOp("Reshape", [input, flatShape])[0];

    const flatAxes = builder.createConstant(
        `${tag}_flat_axes`,
        makeTensorProto(DataType.INT64, [1], [0]),
    );
    const idxUnsq = builder.createOp("Unsqueeze", [flatInIdx, flatAxes])[0];

    const gathered = builder.createOp("Gather", [flatInput, idxUnsq], { axis: 0 })[0];
    return builder.createOp("Squeeze", [gathered, flatAxes])[0];
}
