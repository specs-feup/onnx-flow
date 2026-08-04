import { useState } from "react";
import Select from "react-select";
import TensorNode from '../../../src/Onnx/TensorNode.ts';
import { DataType, type KnownShape, type Shape } from '../../../src/Onnx/OnnxTypes.ts'

import TensorNodeAdder from "./TensorNodeAdder.tsx";
import OperationNodeAdder from "./OperationNodeAdder.tsx";
import ConstantNodeAdder from "./ConstantNodeAdder.tsx";

import { reactSelectCustomStyles } from "./Style.ts";

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
    const [constantProtoName, setConstantProtoName] = useState<string>("");
    const [constantDataType, setConstantDataType] = useState<DataType>(DataType.UNDEFINED);
    const [constantShape, setConstantShape] = useState<KnownShape>([]);
    const [stringData, setStringData] = useState<string[]>([]);
    const [numberData, setNumberData] = useState<number[]>([]);

    /*Tensor Attributes*/
    const [tensorDataType, setTensorDataType] = useState<DataType>(DataType.UNDEFINED);
    const [tensorShapeValue, setTensorShapeValue] = useState<Shape>([]);
    const [tensorKind, setTensorKind] = useState<TensorNode.TensorKind>();

    /*Operation Attributes*/
    const [operationType, setOperationType] = useState<string>("");
    const [operationInputs, setOperationInputs] = useState<string[]>([]);


    const handleCreateClick = () => {
        // eslint-disable-next-line no-useless-assignment
        let onnxData = {};

        if (nodeKind === "tensor") {
            onnxData = {
                id: nodeId,
                kind: "TensorNode",
                tensorType: tensorKind,
                literalType: tensorDataType,
                shape: tensorShapeValue,
                metadata: {},
            };
        } else if (nodeKind === "operation") {
            onnxData = {
                id: nodeId,
                kind: "OperationNode",
                opType: operationType,
                inputs: operationInputs,
                regions: [],
                attributes: {},
                metadata: {},
            };
        } else {
            onnxData = {
                id: nodeId,
                kind: "ConstantNode",
                isInput: false,
                proto: {},
                metadata: {}
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
                styles={reactSelectCustomStyles}
                />
            
            {nodeKind === 'constant' && 
                <ConstantNodeAdder
                    reactSelectStyles={reactSelectCustomStyles}
                    constantTensorProto={constantTensorProto}
                    setConstantTensorProto={setConstantTensorProto}
                />
            }

            {nodeKind === 'operation' && 
                <OperationNodeAdder
                    reactSelectStyles={reactSelectCustomStyles}
                    setOperationType={setOperationType}
                    setOperationInputs={setOperationInputs}
                    valueNodes={valueNodes}
                />
            }

            {nodeKind === 'tensor' &&
                <TensorNodeAdder
                    reactSelectStyles={reactSelectCustomStyles}
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
