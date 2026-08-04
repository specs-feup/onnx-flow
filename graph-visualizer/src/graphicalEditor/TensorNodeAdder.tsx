import Select from "react-select";

import { DataType, type Shape } from "../../../src/Onnx/OnnxTypes";

import { DimensionBuilder } from "./DimensionBuilder.tsx";
import type TensorNode from "../../../src/Onnx/TensorNode.ts";

import { dataTypeOptions } from "./Definitions.ts";

/*--- TENSOR OPTIONS ---*/
const tensorTypes = ['Input', 'Output', 'Intermediate', 'Index', 'Index_Aux'].map((e) => ({
    value: e.toLowerCase(),
    label: e
}));

/*---   ---*/

interface TensorNodeAdderProps {
    reactSelectStyles: any;
    setTensorDataType: (value: DataType) => void;
    setTensorShapeValue: (value: Shape) => void;
    tensorShapeValue: Shape;
    setTensorKind: (value: TensorNode.TensorKind) => void;
}

export default function TensorNodeAdder({    
    reactSelectStyles,
    setTensorDataType,
    setTensorShapeValue,
    tensorShapeValue,
    setTensorKind,
}: TensorNodeAdderProps) {
    /*
    -- literalType: DataType
    -- shape: Shape
    -- type: TensorKind
    ?? extraAttrs?: AttributeProto[] | undefined
    ?? metadata: AttributeMap
    */ 
    return(
    <>
        <label htmlFor="dataType">Literal Type: </label>
        <Select
            isClearable
            name="dataType"
            styles={reactSelectStyles}
            options={dataTypeOptions}
            onChange={(e: any) => setTensorDataType(e.value)}
        />

        <label htmlFor="shape">Shape:</label>
        <DimensionBuilder 
            value={tensorShapeValue}
            onChange={setTensorShapeValue}
        />

        <label htmlFor="tensorKind">Tensor Kind:</label>
        <Select
            isClearable
            name="tensorKind"
            styles={reactSelectStyles}
            options={tensorTypes}
            onChange={(e: any) => setTensorKind(e.value)}
        />
    </> 
    )
}