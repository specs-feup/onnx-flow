import Select from "react-select";

import type { KnownShape, TensorProto } from "../../../src/Onnx/OnnxTypes";

import { dataTypeOptions } from "./Definitions.ts";
import { DimensionBuilder } from "./DimensionBuilder.tsx";

interface ConstantNodeAdderProps {
    reactSelectStyles: any;
    constantTensorProto: TensorProto;
    setConstantTensorProto: (value: TensorProto) => void;
}

export default function ConstantNodeAdder({
    reactSelectStyles,
    constantTensorProto,
    setConstantTensorProto,
}: ConstantNodeAdderProps) {

    return (
        <>
        <label htmlFor="constantTensorProto">Constant TensorProto</label>
        <label>Name:</label>
        <input type="text" name="tensorName" />

        <label>Data Type:</label>
        <Select 
            name="dataType"
            options={dataTypeOptions}
            styles={reactSelectStyles} 
        />

        <label>Shape:</label>
        <DimensionBuilder 
            value={constantTensorProto.dims ?? []}
            onChange={(knownShape: KnownShape) => setConstantTensorProto({
                ...constantTensorProto,
                dims: knownShape,
            })}
        />


        </>
    )
}