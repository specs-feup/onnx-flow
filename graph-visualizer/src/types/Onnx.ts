/**
 * @file Onnx.ts
 * @description Frontend TypeScript type definitions and interfaces representing ONNX graph elements,
 * tensor data types, operator payloads, constant tensors, and select options for the visualizer.
 */

import type OnnxGraph from "../../../src/Onnx/OnnxGraph";
import { DataType, type AttributeMap, type Shape, type TensorProto, type ValueNode } from "../../../src/Onnx/OnnxTypes";
import type TensorNode from "../../../src/Onnx/TensorNode";

/**
 * Represents a selectable option item for ONNX DataType dropdown menus.
 */
export interface DataTypeOption {
    /** The ONNX DataType enum value */
    value: DataType;
    /** The human-readable string label for the data type */
    label: string;
    /** Optional UI display color */
    color?: string;
}

/**
 * Array of available ONNX DataType options derived from the DataType enum for UI select components.
 */
export const dataTypeOptions: DataTypeOption[] = Object.entries(DataType)
    .filter(([key]) => isNaN(Number(key)))
    .map(([key, val]) => ({
        value: val as DataType,
        label: key,
        color: 'white',
    })
);

/**
 * Represents the ONNX-specific data payload attached to a Cytoscape TensorNode.
 */
export interface TensorOnnxData {
    /** Unique identifier for the tensor node */
    id: string;
    /** Discriminator kind indicating this is a TensorNode */
    kind: string;
    /** Role/category of the tensor (e.g. 'input', 'output', 'intermediate', 'index', 'index_aux') */
    tensorType: TensorNode.TensorKind;
    /** Literal element data type of the tensor (e.g. FLOAT, INT32) */
    literalType: DataType;
    /** Shape dimensions of the tensor */
    shape: Shape;
    /** Key-value metadata dictionary attached to the tensor */
    metadata: AttributeMap;
}

/**
 * Represents the ONNX-specific data payload attached to a Cytoscape OperationNode.
 */
export interface OperationOnnxData {
    /** Unique identifier for the operation node */
    id: string;
    /** Discriminator kind indicating this is an OperationNode */
    kind: string;
    /** ONNX operator schema type (e.g. 'MatMul', 'Add', 'If', 'Loop') */
    opType: string;
    /** Operator attributes mapping attribute names to their values */
    attributes: AttributeMap | undefined;
    /** Array of input node IDs consumed by this operation */
    inputs: string[];
    /** Nested subgraph regions for control flow operations (e.g. Loop body, If branches) */
    regions: OnnxGraph.Class[] | undefined;
    /** Key-value metadata dictionary attached to the operation */
    metadata: AttributeMap;
}

/**
 * Represents the ONNX-specific data payload attached to a Cytoscape ConstantNode.
 */
export interface ConstantOnnxData {
    /** Unique identifier for the constant node */
    id: string;
    /** Discriminator kind indicating this is a ConstantNode */
    kind: string;
    /** Flag indicating whether this constant is an explicit graph input */
    isInput: boolean;
    /** TensorProto structure containing constant shape, data type, and raw data buffers */
    proto: TensorProto;
    /** Key-value metadata dictionary attached to the constant */
    metadata: AttributeMap;
}

/**
 * Union type representing all supported ONNX node data structures.
 */
export type OnnxData = TensorOnnxData | OperationOnnxData | ConstantOnnxData;