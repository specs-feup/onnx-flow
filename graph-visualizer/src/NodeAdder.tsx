import { useState } from "react";
import Select from "react-select";
import { StandardOps } from '../../src/Onnx/Schema/definitions/StandardOps/index.ts'
import TensorNode from '../../src/Onnx/TensorNode.ts';
import { DimensionBuilder} from "./DimensionBuilder.tsx";
import { DataType, type Shape } from '../../src/Onnx/OnnxTypes.ts'


const operationsTypes: Array<{ value: string; label: string }> = StandardOps.map((op) => ({
    value: op.opType,
    label: op.opType,
})).sort((a, b) => a.label.localeCompare(b.label));

/*Tensor Options*/
const tensorTypes = ['Input', 'Output', 'Intermediate', 'Index', 'Index_Aux'].map((e) => ({
    value: e.toLowerCase(),
    label: e
}));

interface DataTypeOption {
    value: DataType;
    label: string;
}


const dataTypeOptions: DataTypeOption[] = Object.entries(DataType)
    .filter(([key]) => isNaN(Number(key)))
    .map(([key, val]) => ({
        value: val as DataType,
        label: key,
    }));

const customStyles = {
  option: (provided: any, state: any) => ({
    ...provided,
    color: 'black', 
  }),
};


export default function NodeAdder() {
    const [nodeKind, setNodeKind] = useState<string>("constant");

    /*Tensor Attributes*/
    const [tensorDataType, setTensorDataType] = useState<DataType>();
    const [tensorShapeValue, setTensorShapeValue] = useState<Shape>();
    const [tensorKind, setTensorKind] = useState<TensorNode.TensorKind>();


    /*Operation Attributes*/
    const [operationType, setOperationType] = useState<string>("");



    return (
        <aside style={{display: 'flex', flexDirection: 'column'}}>
            <label htmlFor="kind">Node Kind: </label>
            <Select 
                id="kind"
                onChange={(e: any) => setNodeKind(e.value)}
                options={[
                    {value: "constant", label: "Constant Node"},
                    {value: "tensor", label: "Tensor Node"},
                    {value: "operation", label: "Operation Node"},
                ]}
                styles={customStyles}
                />

            {nodeKind === 'operation' && (
                <>
                <label htmlFor="operation-type">Operation Type:</label>
                <Select 
                    isSearchable
                    isClearable
                    name="operation-type"
                    options={operationsTypes}
                    styles={customStyles}
                />
                </>
            )}

            {nodeKind === 'tensor' && (
                /*
                -- literalType: DataType
                -- shape: Shape
                -- type: TensorKind
                ?? extraAttrs?: AttributeProto[] | undefined
                ?? metadata: AttributeMap
                */ 

                <>
                <label htmlFor="dataType">Literal Type: </label>
                <Select
                    isClearable
                    name="dataType"
                    styles={customStyles}
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
                    styles={customStyles}
                    options={tensorTypes}
                    onChange={(e: any) => setTensorKind(e.value)}
                />
                </>
            )
            }
        </aside>
    );
}
