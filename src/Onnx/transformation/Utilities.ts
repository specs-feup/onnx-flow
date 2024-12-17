export const typeSizeMap: Record<number, number> = {
    0: 0,    // onnx.TensorProto.UNDEFINED
    1: 4,    // onnx.TensorProto.FLOAT
    2: 1,    // onnx.TensorProto.UINT8
    3: 1,    // onnx.TensorProto.INT8
    4: 2,    // onnx.TensorProto.UINT16
    5: 2,    // onnx.TensorProto.INT16
    6: 4,    // onnx.TensorProto.INT32
    7: 8,    // onnx.TensorProto.INT64
    8: -1,   // onnx.TensorProto.STRING (Variable size)
    9: 1,    // onnx.TensorProto.BOOL
    10: 2,   // onnx.TensorProto.FLOAT16
    11: 8,   // onnx.TensorProto.DOUBLE
    12: 4,   // onnx.TensorProto.UINT32
    13: 8,   // onnx.TensorProto.UINT64
    14: 8,   // onnx.TensorProto.COMPLEX64
    15: 16,  // onnx.TensorProto.COMPLEX128
    16: 2,   // onnx.TensorProto.BFLOAT16
    17: 1,   // onnx.TensorProto.FLOAT8E4M3FN
    18: 1,   // onnx.TensorProto.FLOAT8E4M3FNUZ
    19: 2,   // onnx.TensorProto.FLOAT8E5M2
    20: 2,   // onnx.TensorProto.FLOAT8E5M2FNUZ
    21: 1,   // onnx.TensorProto.UINT4
    22: 1    // onnx.TensorProto.INT4
};


export function formatId(name : string, nodeId : string) : string {
    return `${name}_${nodeId}`;
}

