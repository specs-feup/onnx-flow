/**
 * @file ConstantNodeAdder.tsx
 * @description Sub-form component for configuring an ONNX ConstantNode (TensorProto).
 * Manages tensor proto name, literal data type selection, shape dimension configuration,
 * and comma-separated raw constant values with count validation.
 */

import Select from "react-select";
import type { DataType, KnownShape } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import { dataTypeOptions } from "@/types/Onnx.ts";
import { DimensionBuilder } from "./DimensionBuilder.tsx";
import { getReactSelectStyles } from "@/styles/ReactSelectStyle.ts";

/**
 * Properties for the ConstantNodeAdder component.
 */
interface ConstantNodeAdderProps {
    /** Optional custom styles for ReactSelect */
    reactSelectStyles?: any;
    /** Current constant TensorProto name */
    constantProtoName: string;
    /** Selected ONNX DataType for the constant values */
    constantDataType: DataType;
    /** Array of fixed integer dimensions defining the constant tensor shape */
    constantShape: KnownShape;
    /** Array of raw parsed data values */
    protoData: (number | bigint | string)[];
    /** State setter for the constant TensorProto name */
    setConstantProtoName: (value: string) => void;
    /** State setter for the constant data type */
    setConstantDataType: (value: DataType) => void;
    /** State setter for the constant shape */
    setConstantShape: (value: KnownShape) => void;
    /** State setter for the raw data array */
    setProtoData: (value: (number | bigint | string)[]) => void;
    /** Object containing form validation error messages keyed by field name */
    errors?: Record<string, string>;
}

/**
 * Form component for creating or editing an ONNX ConstantNode.
 *
 * @param props - ConstantNodeAdder properties
 * @returns JSX element containing constant configuration inputs
 */
export default function ConstantNodeAdder({

    reactSelectStyles,
    constantProtoName,
    constantDataType,
    constantShape,
    protoData,
    setConstantProtoName,
    setConstantDataType,
    setConstantShape,
    setProtoData,
    errors = {},
}: ConstantNodeAdderProps) {
    const expectedValuesCount = constantShape.length === 0 ? 1 : constantShape.reduce((acc, v) => acc * (Number(v) || 1), 1);
    const dataTypeSelectStyles = getReactSelectStyles(Boolean(errors.constantDataType));

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            <label htmlFor="constantTensorProto" style={{ fontWeight: "bold" }}>Constant TensorProto</label>
            <label>Name: *</label>
            <input 
                type="text" 
                name="tensorName" 
                onChange={(e) => setConstantProtoName(e.target.value)}
                value={constantProtoName}
                style={{
                    border: errors.constantProtoName ? "2px solid #ff4d4f" : "2px solid rgb(95, 92, 102)",
                    background: errors.constantProtoName ? "#321d23" : "#2c2a30",
                    color: "white",
                    borderRadius: "6px",
                    padding: "8px",
                    outline: "none",
                }}
            />
            {errors.constantProtoName && (
                <span style={{ color: "#ff7875", fontSize: "12px", marginTop: "-2px" }}>{errors.constantProtoName}</span>
            )}

            <label>Data Type:</label>
            <Select 
                name="dataType"
                options={dataTypeOptions}
                styles={dataTypeSelectStyles}
                onChange={(e: any) => setConstantDataType(e?.value ?? 0)}
                value={dataTypeOptions.find(opt => opt.value === constantDataType) || null}
                defaultValue={dataTypeOptions[0]}
            />
            {errors.constantDataType && (
                <span style={{ color: "#ff7875", fontSize: "12px", marginTop: "-2px" }}>{errors.constantDataType}</span>
            )}

            <label>Shape:</label>
            <DimensionBuilder 
                value={constantShape}
                onChange={setConstantShape}
                hasError={Boolean(errors.constantShape)}
            />
            {errors.constantShape && (
                <span style={{ color: "#ff7875", fontSize: "12px", marginTop: "-2px" }}>{errors.constantShape}</span>
            )}

            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", color: "#bbb", marginTop: "2px" }}>
                <span>Expected Values: {expectedValuesCount}</span>
                <span style={{ color: protoData.length !== expectedValuesCount && errors.protoData ? "#ff7875" : "#bbb" }}>
                    Actual Values: {protoData.length}
                </span>
            </div>

            <label>Data: * (comma-separated)</label>
            <textarea 
                rows={5} 
                name="data" 
                style={{
                    flexShrink: 0, 
                    alignSelf: "stretch",
                    border: errors.protoData ? "2px solid #ff4d4f" : "2px solid rgb(95, 92, 102)",
                    background: errors.protoData ? "#321d23" : "#2c2a30",
                    color: "white",
                    borderRadius: "6px",
                    padding: "8px",
                    outline: "none",
                }}
                onChange={(e) => {setProtoData(e.target.value.split(",").map(s => s.trim()).filter(Boolean))}}
                value={protoData.join(",")}
            />
            {errors.protoData && (
                <span style={{ color: "#ff7875", fontSize: "12px", marginTop: "-2px" }}>{errors.protoData}</span>
            )}
        </div>
    );
}