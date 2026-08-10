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
    setTensorKind: (value: TensorNode.TensorKind) => void;
    tensorDataType: DataType;
    tensorKind: TensorNode.TensorKind;
}

export default function TensorNodeAdder({
    reactSelectStyles,
    setTensorDataType,
    setTensorShapeValue,
    tensorShapeValue,
    setTensorKind,
    tensorDataType,
    tensorKind,
}: TensorNodeAdderProps) {
    /*
    -- literalType: DataType
    -- shape: Shape
    -- type: TensorKind
    ?? extraAttrs?: AttributeProto[] | undefined
    ?? metadata: AttributeMap
    */
    return (
        <>
            <label htmlFor="dataType">Literal Type: </label>
            <Select
                isClearable
                name="dataType"
                styles={reactSelectStyles}
                options={dataTypeOptions}
                onChange={(e: any) => setTensorDataType(e.value)}
                value={dataTypeOptions.find(opt => opt.value === tensorDataType) || null} // Node Editor
            />

            <label htmlFor="shape">Shape:</label>
            <DimensionBuilder value={tensorShapeValue} onChange={setTensorShapeValue} />

            <label htmlFor="tensorKind">Tensor Kind:</label>
            <Select
                isClearable
                name="tensorKind"
                styles={reactSelectStyles}
                options={tensorTypes}
                onChange={(e: any) => setTensorKind(e.value)}
                value={tensorTypes.find(opt => opt.value === tensorKind) || null} // Node Editor
                defaultValue={tensorTypes[0]}
            />
        </>
    );
}
