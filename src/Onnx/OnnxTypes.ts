import type OnnxGraph from "./OnnxGraph.js";

// =====================================================================================
// RAW ONNX JSON INTERFACES (For parsing external JSON without 'any')
// =====================================================================================

export interface RawOnnxDim {
    dimValue?: string | number;
    dim_value?: string | number;
    dimParam?: string;
    dim_param?: string;
}

export interface RawOnnxTensorType {
    elemType?: number | string;
    elem_type?: number | string;
    shape?: {
        dim?: RawOnnxDim[];
    };
}

export interface RawOnnxTypeProto {
    tensorType?: RawOnnxTensorType;
    tensor_type?: RawOnnxTensorType;
}

export interface RawOnnxValueInfo {
    name: string;
    type?: RawOnnxTypeProto;
}

export interface RawOnnxAttribute {
    name: string;
    // Type can come in as an integer enum or a string like "INTS"
    type?: string | number;
    f?: number;
    i?: number | string;
    s?: string;
    floats?: number[];
    // JSON often exports integer arrays as string arrays to prevent 64-bit precision loss
    ints?: (number | string)[];
    strings?: string[];
    t?: TensorProto;
    tensors?: TensorProto[];
    // Note: Raw attributes contain RAW graphs, not instantiated OnnxGraph.Class objects
    g?: RawOnnxGraph;
    graphs?: RawOnnxGraph[];
}

export interface RawOnnxNode {
    name?: string;
    opType?: string;
    op_type?: string; // snake_case fallback
    input?: string[];
    output?: string[];
    attribute?: RawOnnxAttribute[];
    domain?: string;
    docString?: string;
    doc_string?: string;
}

export interface RawOnnxGraph {
    name?: string;
    node?: RawOnnxNode[];
    initializer?: TensorProto[];
    sparseInitializer?: unknown[];
    sparse_initializer?: unknown[];
    input?: RawOnnxValueInfo[];
    output?: RawOnnxValueInfo[];
    valueInfo?: RawOnnxValueInfo[];
    value_info?: RawOnnxValueInfo[];
    docString?: string;
    doc_string?: string;
}

export interface RawOnnxModel {
    irVersion?: number | string;
    ir_version?: number | string;
    opsetImport?: { domain?: string; version?: number | string }[];
    opset_import?: { domain?: string; version?: number | string }[];
    producerName?: string;
    producer_name?: string;
    producerVersion?: string;
    producer_version?: string;
    domain?: string;
    modelVersion?: number | string;
    model_version?: number | string;
    docString?: string;
    doc_string?: string;
    graph?: RawOnnxGraph;
}

// =====================================================================================
// ACTUAL ONNX TYPES
// =====================================================================================

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
    // --- Newer ONNX Types (Opset 19+) ---
    FLOAT8E4M3FN = 17,
    FLOAT8E4M3FNUZ = 18,
    FLOAT8E5M2 = 19,
    FLOAT8E5M2FNUZ = 20,
    UINT4 = 21,
    INT4 = 22,
}

// ONNX-compatible TensorProto definition
export type TensorProto = {
    name?: string;
    dataType?: DataType;
    dims?: (number | string)[];
    rawData?: { type: string; data: number[] | Buffer | bigint[] };

    // Field mapping for specific types:
    // FLOAT -> floatData
    // DOUBLE, COMPLEX128 -> doubleData
    // INT64 -> int64Data
    // UINT64 -> uint64Data
    // STRING -> stringData
    // INT32, INT16, UINT16, INT8, UINT8, BOOL, FLOAT16, BFLOAT16, FLOAT8*, INT4* -> int32Data
    floatData?: number[];
    int32Data?: number[];
    int64Data?: (number | bigint)[];
    stringData?: string[];
    doubleData?: number[];
    uint64Data?: number[];

    externalData?: unknown;
};

// ONNX-compatible AttributeProto definition
export type AttributeProto = {
    name: string;
    type: AttributeType;
    i?: number; // INT
    f?: number; // FLOAT
    s?: string; // STRING
    ints?: number[]; // INTS
    floats?: number[]; // FLOATS
    strings?: string[]; // STRINGS
    t?: TensorProto; // TENSOR
    tensors?: TensorProto[]; // TENSORS
    g?: OnnxGraph.Class; // GRAPH
    graphs?: OnnxGraph.Class[]; // GRAPHS
};

// =====================================================================================
// OTHER USEFUL TYPES
// =====================================================================================

export type AttributeValue =
    | boolean
    | number
    | string
    | boolean[]
    | number[]
    | string[]
    | object
    | TensorProto
    | TensorProto[]
    | OnnxGraph.Class
    | OnnxGraph.Class[];
export type AttributeMap = Record<string, AttributeValue>;

export type Dim = number | string;
export type Shape = Dim[];
