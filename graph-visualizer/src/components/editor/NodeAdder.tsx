import { useEffect, useState } from "react";
import Select from "react-select";
import TensorNode from "@specs-feup/onnx-flow/Onnx/TensorNode.ts";
import { DataType, type KnownShape, type Shape } from "@specs-feup/onnx-flow/Onnx/OnnxTypes.ts";

import TensorNodeAdder from "./TensorNodeAdder.tsx";
import OperationNodeAdder from "./OperationNodeAdder.tsx";
import ConstantNodeAdder from "./ConstantNodeAdder.tsx";

import { reactSelectCustomStyles } from "@/styles/ReactSelectStyle.ts";
import type { OnnxData } from "@/types/Onnx.ts";

interface NodeAdderProps {
    position?: { x: number; y: number } | null;
    onSubmit?: (nodePayload: any, pos: { x: number; y: number } | null) => void;
    valueNodes: Array<unknown>;
    nodeToEdit?: any;   
}

const nodeKindOptions = [
        { value: "constant", label: "Constant Node" },
        { value: "tensor", label: "Tensor Node" },
        { value: "operation", label: "Operation Node" },
    ];

export default function NodeAdder({ position, onSubmit, valueNodes, nodeToEdit }: NodeAdderProps) {
    /* General Attributes*/
    const [nodeId, setNodeId] = useState<string>("");
    const [nodeKind, setNodeKind] = useState<string>("constant");

    /*Constant Attributes*/
    const [constantProtoName, setConstantProtoName] = useState<string>("");
    const [constantDataType, setConstantDataType] = useState<DataType>(DataType.UNDEFINED);
    const [constantShape, setConstantShape] = useState<KnownShape>([]);
    const [protoData, setProtoData] = useState<(number | bigint | string)[]>([]);

    /*Tensor Attributes*/
    const [tensorDataType, setTensorDataType] = useState<DataType>(DataType.UNDEFINED);
    const [tensorShapeValue, setTensorShapeValue] = useState<Shape>([]);
    const [tensorKind, setTensorKind] = useState<TensorNode.TensorKind>("input");

    /*Operation Attributes*/
    const [operationType, setOperationType] = useState<string>("");
    const [operationInputs, setOperationInputs] = useState<string[]>([]);

    /* Node Editor */
    useEffect(() => {
        if (nodeToEdit) {
            const onnx = nodeToEdit.onnxData;
            setNodeId(onnx.id || "");
            
            if (onnx.kind === "TensorNode") {
                setNodeKind("tensor");
                setTensorDataType(onnx.literalType);
                setTensorShapeValue(onnx.shape || []);
                setTensorKind(onnx.tensorType);
            } else if (onnx.kind === "OperationNode") {
                setNodeKind("operation");
                setOperationType(onnx.opType || "");
                setOperationInputs(onnx.inputs || []);
            } else if (onnx.kind === "ConstantNode") {
                setNodeKind("constant");
                setConstantProtoName(onnx.proto?.name || "");
                const dType = onnx.proto?.dataType || DataType.UNDEFINED;
                setConstantDataType(dType);
                setConstantShape(onnx.proto?.dims || []);
                
                let selectedArray: any[] = [];
                switch (dType) {
                    case DataType.FLOAT: selectedArray = onnx.proto?.floatData || []; break;
                    case DataType.DOUBLE:
                    case DataType.COMPLEX128: selectedArray = onnx.proto?.doubleData || []; break;
                    case DataType.INT64: selectedArray = onnx.proto?.int64Data || []; break;
                    case DataType.UINT64: selectedArray = onnx.proto?.uint64Data || []; break;
                    case DataType.STRING: selectedArray = onnx.proto?.stringData || []; break;
                    case DataType.INT32:
                    case DataType.INT16:
                    case DataType.UINT16:
                    case DataType.INT8:
                    case DataType.UINT8:
                    case DataType.BOOL:
                    case DataType.FLOAT16:
                    case DataType.BFLOAT16:
                    case DataType.INT4: selectedArray = onnx.proto?.int32Data || []; break;
                    default: selectedArray = onnx.proto?.rawData || []; break;
                }
                setProtoData(selectedArray);
            }
        }
    }, [nodeToEdit]);

    const handleCreateClick = () => {
        // eslint-disable-next-line no-useless-assignment
        let onnxData: OnnxData;

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
                proto: {
                    name: constantProtoName,
                    dataType: constantDataType,
                    dims: constantShape,
                    rawData: undefined,
                    floatData: [],
                    int32Data: [],
                    int64Data: [],
                    stringData: [],
                    doubleData: [],
                    uint64Data: [],
                },
                metadata: {},
            };

            let selectedArray;
            switch (constantDataType) {
                case DataType.FLOAT:
                    selectedArray = "floatData";
                    break;
                case DataType.DOUBLE:
                case DataType.COMPLEX128:
                    selectedArray = "doubleData";
                    break;
                case DataType.INT64:
                    selectedArray = "int64Data";
                    break;
                case DataType.UINT64:
                    selectedArray = "uint64Data";
                    break;
                case DataType.STRING:
                    selectedArray = "stringData";
                    break;
                case DataType.INT32:
                case DataType.INT16:
                case DataType.UINT16:
                case DataType.INT8:
                case DataType.UINT8:
                case DataType.BOOL:
                case DataType.FLOAT16:
                case DataType.BFLOAT16:
                // case DataType.FLOAT8:
                case DataType.INT4:
                    selectedArray = "int32Data";
                    break;
                default:
                    selectedArray = "rawData";
                    break;
            }
            onnxData.proto[selectedArray] = protoData;
        }

        console.log(onnxData);

        const nodePayload = {
            onnxData,
            label: nodeId,
        };

        if (onSubmit) {
            onSubmit(nodePayload, position || null);
        }
    };

    return (
        <aside style={{ display: "flex", flexDirection: "column" }}>
            <label htmlFor="id">Node ID: </label>
            <input name="id" type="text" value={nodeId} color="white" />
            <button
                type="button"
                onClick={() => setNodeId(`node_${Math.random().toString(36).substr(2, 9)}`)}
            >
                Generate Random ID
            </button>

            <label color="white" htmlFor="kind">
                Node Kind:{" "}
            </label>
            <Select
                id="kind"
                onChange={(e: any) => setNodeKind(e.value)}
                options={nodeKindOptions}
                value={nodeKindOptions.find(opt => opt.value === nodeKind) || null} // Node Editor
                // defaultValue={{ value: "constant", label: "Constant Node" }}
                styles={reactSelectCustomStyles}
            />

            {nodeKind === "constant" && (
                <ConstantNodeAdder
                    reactSelectStyles={reactSelectCustomStyles}
                    constantProtoName={constantProtoName}
                    constantDataType={constantDataType}
                    constantShape={constantShape}
                    protoData={protoData}
                    setConstantProtoName={setConstantProtoName}
                    setConstantDataType={setConstantDataType}
                    setConstantShape={setConstantShape}
                    setProtoData={setProtoData}
                />
            )}

            {nodeKind === "operation" && (
                <OperationNodeAdder
                    reactSelectStyles={reactSelectCustomStyles}
                    setOperationType={setOperationType}
                    setOperationInputs={setOperationInputs}
                    valueNodes={valueNodes}
                    operationType={operationType} // Node Editor
                    operationInputs={operationInputs} // Node Editor
                />
            )}

            {nodeKind === "tensor" && (
                <TensorNodeAdder
                    reactSelectStyles={reactSelectCustomStyles}
                    setTensorDataType={setTensorDataType}
                    setTensorShapeValue={setTensorShapeValue}
                    tensorShapeValue={tensorShapeValue}
                    setTensorKind={setTensorKind}
                    tensorDataType={tensorDataType} // Node Editor
                    tensorKind={tensorKind} // Node Editor
                />
            )}
            <button type="button" onClick={handleCreateClick} style={{ marginTop: "15px" }}>
                {nodeToEdit ? "Edit Node" : "Create Node"}
            </button>
        </aside>
    );
}
