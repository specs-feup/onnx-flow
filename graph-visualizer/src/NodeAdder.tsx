import { useState } from "react";
import Select from "react-select";
import TensorNode from '../../src/Onnx/TensorNode.ts';
import { DataType, type Shape } from '../../src/Onnx/OnnxTypes.ts'

import TensorNodeAdder from "./graphicalEditor/TensorNodeAdder.tsx";
import OperationNodeAdder from "./graphicalEditor/OperationNodeAdder.tsx";

/*--- REACT SELECT STYLE---*/
const customStyles = {
    singleValue: (provided: any, state: any) => ({
        ...provided,
        color: 'white',
    }),
    color: 'white',
    menu: (provided: any, state: any) => ({
        ...provided,
        backgroundColor: '#2c2a30',
        border: '2px solid rgb(95, 92, 102)',
        '&:hover': {
            backgroundColor: '#3e3c46',
            border: '2px solid rgb(132, 124, 150)',
            },
        }),

    control: (provided: any, state: any) => ({
      ...provided,
      color: '#ffc400',
      backgroundColor: '#2c2a30',
      border: '2px solid rgb(95, 92, 102)',

      '&:hover': {
        backgroundColor: '#3e3c46',
        border: '2px solid rgb(132, 124, 150)',
      },
    }),
    
    option: (provided: any, state: any) => ({
        ...provided,
        color: 'white', 
        backgroundColor: '#2c2a30',
        '&:hover': {
        backgroundColor: '#3e3c46',

        },
    margin: '0px',
  }),
};
/*---   ---*/

interface NodeAdderProps {
    position?: { x: number; y: number } | null;
    onSubmit?: (nodePayload: any, pos: { x: number; y: number } | null) => void;
    valueNodes: Array<unknown>;
}

export default function NodeAdder({ position, onSubmit, valueNodes }: NodeAdderProps) {
    /* General Attributes*/
    const [nodeId, setNodeId] = useState<string>("")
    const [nodeKind, setNodeKind] = useState<string>("constant");

    /*Constant Attributes*/

    /*Tensor Attributes*/
    const [tensorDataType, setTensorDataType] = useState<DataType>();
    const [tensorShapeValue, setTensorShapeValue] = useState<Shape>([]);
    const [tensorKind, setTensorKind] = useState<TensorNode.TensorKind>();


    /*Operation Attributes*/
    const [operationType, setOperationType] = useState<string>("");
    const [operationInputs, setOperationInputs] = useState<string[]>([]);


    const handleCreateClick = () => {
        let onnxData: Record<string, any> = {};

        if (nodeKind === "tensor") {
            onnxData = {
                id: nodeId,
                kind: "TensorNode",
                tensorType: tensorKind,
                literalType: tensorDataType,
                shape: tensorShapeValue,
                metadata: {}
            };
        } else if (nodeKind === "operation") {
            onnxData = {
                id: nodeId,
                kind: "OperationNode",
                opType: operationType,
                inputs: operationInputs,
                regions: [],
                attributes: {},
                metadata: {}
            };
        } else {
            onnxData = {
                id: nodeId,
                kind: "Constant",
                proto: { dataType: "Constant" }
            };
        }

        const nodePayload = {
            onnxData,
            label: nodeId,
        };

        if (onSubmit) {
            onSubmit(nodePayload, position || null);
        }
    };

    return (
        
        <aside style={{display: 'flex', flexDirection: 'column'}}>
            <label htmlFor="id">Node ID: </label>
            <input  name="id" type="text" value={nodeId} color="white"/>
            <button type="button" onClick={() => setNodeId(`node_${Math.random().toString(36).substr(2, 9)}`)}>Generate Random ID</button>

            <label  color="white"  htmlFor="kind">Node Kind: </label>
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

            {nodeKind === 'operation' && 
                <OperationNodeAdder
                    reactSelectStyles={customStyles}
                    setOperationType={setOperationType}
                    setOperationInputs={setOperationInputs}
                    valueNodes={valueNodes}
                />
            }

            {nodeKind === 'tensor' &&
                <TensorNodeAdder
                    reactSelectStyles={customStyles}
                    setTensorDataType={setTensorDataType}
                    setTensorShapeValue={setTensorShapeValue}
                    tensorShapeValue={tensorShapeValue}
                    setTensorKind={setTensorKind}
                />
            }
            {/* BOTÃO ADICIONADO NO FINAL DO FORMULÁRIO PARA CRIAR O NÓ NO GRAFO */}
            <button 
                type="button" 
                onClick={handleCreateClick}
                style={{ marginTop: '15px' }}
            >
                Create Node
            </button>
        </aside>
    );
}
