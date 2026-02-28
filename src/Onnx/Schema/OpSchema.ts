import type { AttributeType, AttributeValue } from "../OnnxTypes.js";

export enum OpCategory {
    ElementWise = "ElementWise",
    Reduction = "Reduction",
    Spatial = "Spatial",
    DataMovement = "DataMovement",
    ControlFlow = "ControlFlow",
    Normalization = "Normalization",
    Other = "Other",
}

/**
 * Defines a single attribute for an operator (e.g., 'kernel_shape' for Conv).
 */
export interface AttributeDefinition {
    name: string;
    type: AttributeType;
    required: boolean;
    defaultValue?: AttributeValue;
    description?: string;
    structural?: boolean;
}

/**
 * Defines an input or output for an operator.
 */
export interface IOInterface {
    name: string;
    typeConstraint?: string; // e.g., "T" (matches a defined type constraint)
    variadic?: boolean; // If true, this input accepts multiple tensors (e.g., Concat)
    optional?: boolean; // If true, this input can be omitted
}

/**
 * The full definition of an ONNX Operator.
 */
export interface OpSchema {
    opType: string;
    domain?: string; // Default is 'ai.onnx' (empty string)
    sinceVersion: number; // The opset version where this definition became valid

    category: OpCategory;
    broadcastable: boolean;
    hasState: boolean;

    inputs: IOInterface[];
    outputs: IOInterface[];
    attributes: Record<string, AttributeDefinition>;
    typeConstraints?: Record<string, string[]>;

    /**
     * Optional: Logic to infer output shapes based on inputs and attributes.
     * Can move the logic from `InferShapes.ts` here eventually.
     */
    inferShape?: (
        inputShapes: number[][],
        attributes: Record<string, AttributeValue>,
    ) => number[][];
}
