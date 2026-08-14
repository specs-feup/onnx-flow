/**
 * @file TensorNodeAdder.tsx
 * @description Sub-form component for configuring an ONNX TensorNode.
 * Manages tensor literal data type (FLOAT, INT32, etc.), shape dimensions
 * via DimensionBuilder, and tensor role category ('input', 'output', 'intermediate', etc.).
 */

import Select from "react-select";
import { DataType, type Shape } from "@specs-feup/onnx-flow/Onnx/OnnxTypes.ts";
import { DimensionBuilder } from "./DimensionBuilder.tsx";
import type TensorNode from "@specs-feup/onnx-flow/Onnx/TensorNode.ts";
import { dataTypeOptions } from "@/types/Onnx.ts";
import { getReactSelectStyles } from "@/styles/ReactSelectStyle.ts";

/**
 * Dropdown options for selecting the TensorKind category.
 */
const tensorTypes = ["Input", "Output", "Intermediate", "Index", "Index_Aux"].map((e) => ({
    value: e.toLowerCase(),
    label: e,
}));

/**
 * Properties for the TensorNodeAdder component.
 */
interface TensorNodeAdderProps {
    /** Optional custom styles for ReactSelect */
    reactSelectStyles?: any;
    /** State setter for the tensor's literal DataType */
    setTensorDataType: (value: DataType) => void;
    /** State setter for the tensor's Shape */
    setTensorShapeValue: (value: Shape) => void;
    /** Current shape array of the tensor */
    tensorShapeValue: Shape;
    /** Optional state setter for the TensorKind */
    setTensorKind?: (value: TensorNode.TensorKind) => void;
    /** Current DataType value of the tensor */
    tensorDataType: DataType;
    /** Current TensorKind category value */
    tensorKind?: TensorNode.TensorKind;
    /** Flag to control visibility of the Tensor Kind selection field */
    showTensorKind?: boolean;
    /** Flag indicating whether the entire form is disabled (e.g. inside an operation attribute) */
    disabled?: boolean;
    /** Object containing form validation error messages keyed by field name */
    errors?: Record<string, string>;
}

/**
 * Form component for creating or editing an ONNX TensorNode.
 *
 * @param props - TensorNodeAdder properties
 * @returns JSX element containing tensor configuration inputs
 */
export default function TensorNodeAdder({

    reactSelectStyles,
    setTensorDataType,
    setTensorShapeValue,
    tensorShapeValue,
    setTensorKind,
    tensorDataType,
    tensorKind,
    showTensorKind = true,
    disabled = false,
    errors = {},
}: TensorNodeAdderProps) {
    const dataTypeSelectStyles = getReactSelectStyles(Boolean(errors.tensorDataType));
    const tensorKindSelectStyles = getReactSelectStyles(Boolean(errors.tensorKind));

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: "6px", opacity: disabled ? 0.6 : 1, pointerEvents: disabled ? "none" : "auto" }}>
            <label htmlFor="dataType">Literal Type:</label>
            <Select
                isDisabled={disabled}
                isClearable
                name="dataType"
                styles={dataTypeSelectStyles}
                options={dataTypeOptions}
                onChange={(e: any) => setTensorDataType(e?.value ?? DataType.UNDEFINED)}
                value={dataTypeOptions.find(opt => opt.value === tensorDataType) || null}
            />
            {errors.tensorDataType && (
                <span style={{ color: "#ff7875", fontSize: "12px", marginTop: "-2px" }}>{errors.tensorDataType}</span>
            )}

            <label htmlFor="shape">Shape:</label>
            <DimensionBuilder
                value={tensorShapeValue || []}
                onChange={setTensorShapeValue}
                hasError={Boolean(errors.tensorShape)}
            />
            {errors.tensorShape && (
                <span style={{ color: "#ff7875", fontSize: "12px", marginTop: "-2px" }}>{errors.tensorShape}</span>
            )}

            {showTensorKind && setTensorKind && (
                <>
                    <label htmlFor="tensorKind">Tensor Kind: *</label>
                    <Select
                        isDisabled={disabled}
                        isClearable
                        name="tensorKind"
                        styles={tensorKindSelectStyles}
                        options={tensorTypes}
                        onChange={(e: any) => setTensorKind(e?.value ?? "intermediate")}
                        value={tensorTypes.find(opt => opt.value === tensorKind) || null}
                        defaultValue={tensorTypes[0]}
                    />
                    {errors.tensorKind && (
                        <span style={{ color: "#ff7875", fontSize: "12px", marginTop: "-2px" }}>{errors.tensorKind}</span>
                    )}
                </>
            )}
        </div>
    );
}
