import Select from "react-select";

import type { DataType, KnownShape } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";

import { dataTypeOptions } from "@/types/Onnx.ts";
import { DimensionBuilder } from "./DimensionBuilder.tsx";

interface ConstantNodeAdderProps {
    reactSelectStyles: any;
    constantProtoName: string;
    constantDataType: DataType;
    constantShape: KnownShape;
    protoData: (number | bigint | string)[];
    setConstantProtoName: (value: string) => void;
    setConstantDataType: (value: DataType) => void;
    setConstantShape: (value: KnownShape) => void;
    setProtoData: (value: (number | bigint | string)[]) => void;        
}

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
}: ConstantNodeAdderProps) {

    return (
        <>
        <label htmlFor="constantTensorProto">Constant TensorProto</label>
        <label>Name:</label>
        <input 
            type="text" 
            name="tensorName" 
            onChange={(e) => setConstantProtoName(e.target.value)}
            value={constantProtoName}
        />

        <label>Data Type:</label>
        <Select 
            name="dataType"
            options={dataTypeOptions}
            styles={reactSelectStyles}
            onChange={(e: any) => setConstantDataType(e.value)}
            value={dataTypeOptions.find(opt => opt.value === constantDataType) || null}
            defaultValue={dataTypeOptions[0]}
        />

        <label>Shape:</label>
        <DimensionBuilder 
            value={constantShape}
            onChange={setConstantShape}
        />
        {constantShape.length !== 0 && 
                <p>Expected Value: {constantShape.reduce((total, v) => v * total)}</p>}
        <p>Actual Values: {protoData.length }</p>
        <label>Data:</label>
        <textarea 
            rows={5} 
            name="data" 
            style={{
                flexShrink: 0, 
                alignSelf: "stretch"
            }}
            onChange={(e) => {setProtoData(e.target.value.split(",").filter(Boolean))}}
            value={protoData.join(",")}
        >    
        </textarea>

        </>
    )
}