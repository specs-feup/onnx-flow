import Select from "react-select";

import { DataType, type Shape } from "@specs-feup/onnx-flow/Onnx/OnnxTypes.ts";

import { DimensionBuilder } from "./DimensionBuilder.tsx";
import type TensorNode from "@specs-feup/onnx-flow/Onnx/TensorNode.ts";

import { dataTypeOptions } from "@/types/Onnx.ts";

/*--- TENSOR OPTIONS ---*/
const tensorTypes = ["Input", "Output", "Intermediate", "Index", "Index_Aux"].map((e) => ({
    value: e.toLowerCase(),
    label: e,
}));
/*---   ---*/

interface TensorNodeAdderProps {
    reactSelectStyles: any;
    setTensorDataType: (value: DataType) => void;
    setTensorShapeValue: (value: Shape) => void;
    tensorShapeValue: Shape;
    setTensorKind?: (value: TensorNode.TensorKind) => void;
    tensorDataType: DataType;
    tensorKind?: TensorNode.TensorKind;
    showTensorKind?: boolean;
    disabled?: boolean;
}

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
}: TensorNodeAdderProps) {
    /*
    -- literalType: DataType
    -- shape: Shape
    -- type: TensorKind
    ?? extraAttrs?: AttributeProto[] | undefined
    ?? metadata: AttributeMap
    */
    return (
        <div style={{ display: "flex", flexDirection: "column", gap: "6px", opacity: disabled ? 0.6 : 1, pointerEvents: disabled ? "none" : "auto" }}>
            <label htmlFor="dataType">Literal Type: </label>
            <Select
                isDisabled={disabled}
                isClearable
                name="dataType"
                styles={reactSelectStyles}
                options={dataTypeOptions}
                onChange={(e: any) => setTensorDataType(e?.value ?? DataType.UNDEFINED)}
                value={dataTypeOptions.find(opt => opt.value === tensorDataType) || null} // Node Editor
            />

            <label htmlFor="shape">Shape:</label>
            <DimensionBuilder value={tensorShapeValue || []} onChange={setTensorShapeValue} />

            {showTensorKind && setTensorKind && (
                <>
                    <label htmlFor="tensorKind">Tensor Kind:</label>
                    <Select
                        isDisabled={disabled}
                        isClearable
                        name="tensorKind"
                        styles={reactSelectStyles}
                        options={tensorTypes}
                        onChange={(e: any) => setTensorKind(e?.value ?? "intermediate")}
                        value={tensorTypes.find(opt => opt.value === tensorKind) || null} // Node Editor
                        defaultValue={tensorTypes[0]}
                    />
                </>
            )}
        </div>
    );
}
