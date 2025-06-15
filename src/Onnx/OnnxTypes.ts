// enums for ONNX attribute and tensor types

export enum AttributeType {
  UNDEFINED = 0,
  FLOAT = 1,
  INT = 2,
  STRING = 3,
  TENSOR = 4,
  GRAPH = 5,
  FLOATS = 6,
  INTS = 7,
  STRINGS = 8,
  TENSORS = 9,
  GRAPHS = 10,
  SPARSE_TENSOR = 11,
  SPARSE_TENSORS = 12,
}

export enum DataType {
  UNDEFINED = 0,
  FLOAT = 1,
  UINT8 = 2,
  INT8 = 3,
  UINT16 = 4,
  INT16 = 5,
  INT32 = 6,
  INT64 = 7,
  STRING = 8,
  BOOL = 9,
  FLOAT16 = 10,
  DOUBLE = 11,
  UINT32 = 12,
  UINT64 = 13,
  COMPLEX64 = 14,
  COMPLEX128 = 15,
  BFLOAT16 = 16,
}

// ONNX-compatible TensorProto definition
export type TensorProto = {
  name?: string;
  dataType?: DataType;
  dims?: number[];
  rawData?: { type: string; data: number[] }; // Buffer
  floatData?: number[];
  int32Data?: number[];
  int64Data?: (number | bigint)[];
  stringData?: string[];
  doubleData?: number[];
  uint64Data?: number[];
  externalData?: any; // Should be ExternalDataProto if needed
};

// ONNX-compatible AttributeProto definition
export type AttributeProto = {
  name: string;
  type: AttributeType;
  i?: number;
  f?: number;
  s?: string;
  ints?: number[];
  floats?: number[];
  t?: TensorProto;
  g?: any; // Should be GraphProto if needed
};
