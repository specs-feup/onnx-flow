import { AttributeType, DataType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import type { OpSchema } from "../../OpSchema.js";
import { OpCategory, T_ANY } from "../../OpSchema.js";

export const OptionalOp: OpSchema = {
    opType: "Optional",
    sinceVersion: 15,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "input", typeConstraint: T_ANY, optional: true }],
    outputs: [{ name: "output", typeConstraint: "optional(T)" }],
    attributes: {
        type: { name: "type", type: AttributeType.TENSOR, required: false },
    },
    inferShape: () => [{ shape: [], dtype: DataType.UNDEFINED }],
};

export const OptionalHasElement: OpSchema = {
    opType: "OptionalHasElement",
    sinceVersion: 15,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "input", typeConstraint: "optional(T)" }],
    outputs: [{ name: "output", typeConstraint: "tensor(bool)" }],
    attributes: {},
    inferShape: () => [{ shape: [], dtype: DataType.BOOL }], // Always a 0D scalar boolean
};

export const OptionalGetElement: OpSchema = {
    opType: "OptionalGetElement",
    sinceVersion: 15,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "input", typeConstraint: "optional(T)" }],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {},
    inferShape: () => [{ shape: [], dtype: DataType.UNDEFINED }], // Inherits from the wrapped tensor
};

export const OptionalOps: OpSchema[] = [OptionalOp, OptionalHasElement, OptionalGetElement];
