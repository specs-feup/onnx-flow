/**
 * @file NodeAdder.tsx
 * @description Master form component for adding new nodes or editing existing nodes in the graph.
 * Supports ConstantNode, TensorNode, and OperationNode kinds, pre-populating fields during edits,
 * extracting and constructing nested subgraph regions, generating random IDs, validating form inputs,
 * and synthesizing schema output tensor nodes.
 */

import { useEffect, useState } from "react";
import Select from "react-select";
import TensorNode from "@specs-feup/onnx-flow/Onnx/TensorNode.ts";
import { AttributeType, DataType, type AttributeValue, type KnownShape, type Shape } from "@specs-feup/onnx-flow/Onnx/OnnxTypes.ts";
import type { OpSchema } from "@specs-feup/onnx-flow/Onnx/Schema/OpSchema.ts";

import TensorNodeAdder from "./TensorNodeAdder.tsx";
import OperationNodeAdder, { operationsTypes, getPattern } from "./OperationNodeAdder.tsx";
import ConstantNodeAdder from "./ConstantNodeAdder.tsx";

import { reactSelectCustomStyles, getReactSelectStyles } from "@/styles/ReactSelectStyle.ts";
import type { OnnxData } from "@/types/Onnx.ts";

/**
 * Properties for the NodeAdder component.
 */
interface NodeAdderProps {
    /** Coordinates on the Cytoscape canvas where the new node should be placed */
    position?: { x: number; y: number } | null;
    /** Callback invoked with the completed node payload and coordinate position upon submission */
    onSubmit?: (nodePayload: any, pos: { x: number; y: number } | null) => void;
    /** List of value nodes available for operation input binding */
    valueNodes: Array<unknown>;
    /** Full list of graph nodes for region/parent mapping */
    graphNodes?: Array<unknown>;
    /** Node object to populate when editing an existing graph node */
    nodeToEdit?: any;   
}

/**
 * Options for selecting the high-level ONNX node kind.
 */
const nodeKindOptions = [
    { value: "constant", label: "Constant Node" },
    { value: "tensor", label: "Tensor Node" },
    { value: "operation", label: "Operation Node" },
];

/**
 * Main node adder and editor form component.
 *
 * @param props - NodeAdder properties
 * @returns JSX element containing the node creation/editing controls
 */
export default function NodeAdder({ position, onSubmit, valueNodes, graphNodes, nodeToEdit }: NodeAdderProps) {

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
    const [operationType, setOperationType] = useState<OpSchema>(operationsTypes[0].value);
    const [operationInputs, setOperationInputs] = useState<string[]>([]);
    const [operationAttributes, setOperationAttributes] = useState<AttributeValue[]>([]);

    /* Validation Errors State */
    const [errors, setErrors] = useState<Record<string, string>>({});

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
                
                const opMatch = operationsTypes.find(op => op.value.opType === onnx.opType);
                const matchedOp = opMatch ? opMatch.value : operationsTypes[0].value;
                setOperationType(matchedOp);
                
                const flatInputs = onnx.inputs || [];
                const formInputs: string[] = [];
                let flatIndex = 0;

                matchedOp.inputs.forEach((schemaInput, i) => {
                    if (schemaInput.variadic) {
                        formInputs[i] = flatInputs.slice(flatIndex).join(",");
                        flatIndex = flatInputs.length; 
                    } else {
                        formInputs[i] = flatInputs[flatIndex] || "";
                        flatIndex++;
                    }
                });

                setOperationInputs(formInputs);

                if (matchedOp.attributes) {
                    let regionIndex = 0;
                    const mappedAttributes = Object.values(matchedOp.attributes).map((attr) => {
                        const existingVal = onnx.attributes?.[attr.name];
                        if (existingVal !== undefined) {
                            if (attr.type === AttributeType.GRAPH || attr.type === AttributeType.GRAPHS) {
                                regionIndex++;
                                if (Array.isArray(existingVal)) return existingVal;
                                if (typeof existingVal === "string") return existingVal.split(",").map(s => s.trim()).filter(Boolean);
                                if (typeof existingVal === "object" && existingVal !== null) {
                                    if ("elements" in existingVal && (existingVal as any).elements?.nodes) {
                                        return (existingVal as any).elements.nodes.map((n: any) => n.data?.id || n.id);
                                    }
                                    return existingVal;
                                }
                            }
                            if (
                                attr.type === AttributeType.TENSOR ||
                                attr.type === AttributeType.SPARSE_TENSOR ||
                                attr.type === AttributeType.TENSORS ||
                                attr.type === AttributeType.SPARSE_TENSORS
                            ) {
                                return existingVal;
                            }
                            return Array.isArray(existingVal) ? existingVal.join(",") : existingVal;
                        }
                        if (attr.type === AttributeType.GRAPH || attr.type === AttributeType.GRAPHS) {
                            if (onnx.regions && onnx.regions[regionIndex]?.elements?.nodes) {
                                const ids = onnx.regions[regionIndex].elements.nodes.map((n: any) => n.data?.id || n.id).filter(Boolean);
                                regionIndex++;
                                return ids;
                            }
                            if (graphNodes && Array.isArray(graphNodes)) {
                                const childIds = graphNodes
                                    .filter((n: any) => (n as any)?.data?.parent === onnx.id && ((n as any)?.data?.regionIndex === regionIndex || (n as any)?.data?.regionIndex === undefined))
                                    .map((n: any) => (n as any)?.data?.id || (n as any)?.id)
                                    .filter(Boolean);
                                if (childIds.length > 0) {
                                    regionIndex++;
                                    return childIds;
                                }
                            }
                            regionIndex++;
                            return [];
                        }
                        if (
                            attr.type === AttributeType.TENSOR ||
                            attr.type === AttributeType.SPARSE_TENSOR ||
                            attr.type === AttributeType.TENSORS ||
                            attr.type === AttributeType.SPARSE_TENSORS
                        ) {
                            return {
                                dataType: DataType.UNDEFINED,
                                dims: [],
                            };
                        }
                        return attr.defaultValue !== undefined ? attr.defaultValue : "";
                    });
                    setOperationAttributes(mappedAttributes);
                }
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

    const validateForm = (): Record<string, string> => {
        const newErrors: Record<string, string> = {};

        // 1. Validate Node ID
        const trimmedId = nodeId.trim();
        if (!trimmedId) {
            newErrors["nodeId"] = "Node ID is required";
        } else {
            const allNodes = (graphNodes || valueNodes || []) as any[];
            const isDuplicate = allNodes.some((n: any) => {
                const existingId = n?.data?.id || n?.id;
                if (!existingId) return false;
                if (nodeToEdit && existingId === nodeToEdit.id) return false;
                return existingId === trimmedId;
            });
            if (isDuplicate) {
                newErrors["nodeId"] = `Node ID '${trimmedId}' already exists in graph`;
            }
        }

        // 2. Validate Node Kind specific fields
        if (nodeKind === "tensor") {
            if (tensorDataType === undefined || tensorDataType === null || typeof tensorDataType !== "number") {
                newErrors["tensorDataType"] = "Literal Type is required";
            }
            if (!tensorKind) {
                newErrors["tensorKind"] = "Tensor Kind is required";
            }
            if (Array.isArray(tensorShapeValue)) {
                for (let i = 0; i < tensorShapeValue.length; i++) {
                    const dim = tensorShapeValue[i];
                    if (dim !== undefined && dim !== null && dim !== "" && typeof dim === "number") {
                        if (dim < 0 || !Number.isInteger(dim)) {
                            newErrors["tensorShape"] = `Dimension D${i} must be a non-negative integer or symbol`;
                            break;
                        }
                    }
                }
            }
        } else if (nodeKind === "constant") {
            if (!constantProtoName.trim()) {
                newErrors["constantProtoName"] = "TensorProto Name is required";
            }
            if (constantDataType === undefined || constantDataType === null || typeof constantDataType !== "number") {
                newErrors["constantDataType"] = "Data Type is required";
            }
            let isShapeValid = true;
            for (let i = 0; i < constantShape.length; i++) {
                const dim = constantShape[i];
                if (typeof dim !== "number" || dim < 0 || !Number.isInteger(dim)) {
                    newErrors["constantShape"] = `Constant dimension D${i} must be a non-negative integer`;
                    isShapeValid = false;
                    break;
                }
            }
            if (isShapeValid) {
                const expectedCount = constantShape.length === 0 ? 1 : constantShape.reduce((acc, v) => acc * (Number(v) || 1), 1);
                if (protoData.length !== expectedCount) {
                    newErrors["protoData"] = `Expected ${expectedCount} value${expectedCount === 1 ? "" : "s"} for shape [${constantShape.join(",")}], but got ${protoData.length}`;
                } else {
                    for (let i = 0; i < protoData.length; i++) {
                        const item = protoData[i];
                        if (constantDataType === DataType.FLOAT || constantDataType === DataType.DOUBLE || constantDataType === DataType.FLOAT16 || constantDataType === DataType.BFLOAT16) {
                            if (isNaN(Number(item))) {
                                newErrors["protoData"] = `Value at index ${i} ('${item}') is not a valid float`;
                                break;
                            }
                        } else if (
                            constantDataType === DataType.INT32 ||
                            constantDataType === DataType.INT64 ||
                            constantDataType === DataType.INT16 ||
                            constantDataType === DataType.INT8 ||
                            constantDataType === DataType.UINT32 ||
                            constantDataType === DataType.UINT64 ||
                            constantDataType === DataType.UINT16 ||
                            constantDataType === DataType.UINT8
                        ) {
                            if (isNaN(Number(item)) || !Number.isInteger(Number(item))) {
                                newErrors["protoData"] = `Value at index ${i} ('${item}') is not a valid integer`;
                                break;
                            }
                        }
                    }
                }
            }
        } else if (nodeKind === "operation") {
            if (!operationType) {
                newErrors["operationType"] = "Operation Type is required";
            } else {
                operationType.inputs.forEach((input, index) => {
                    const isOptional = input.optional || false;
                    const isRequired = !isOptional;
                    const val = operationInputs[index];
                    const inputKey = `input_${input.name || index}`;

                    if (isRequired) {
                        if (!val || (typeof val === "string" && val.trim() === "")) {
                            newErrors[inputKey] = `Input '${input.name}' is required`;
                        }
                    }
                });

                if (operationType.attributes) {
                    Object.values(operationType.attributes).forEach((att, index) => {
                        const val = operationAttributes[index];
                        const attrKey = `attr_${att.name}`;
                        const isGraphAttr = att.type === AttributeType.GRAPH || att.type === AttributeType.GRAPHS;
                        const isTensorAttr =
                            att.type === AttributeType.TENSOR ||
                            att.type === AttributeType.SPARSE_TENSOR ||
                            att.type === AttributeType.TENSORS ||
                            att.type === AttributeType.SPARSE_TENSORS;

                        if (att.required) {
                            if (val === undefined || val === null || val === "") {
                                newErrors[attrKey] = `Attribute '${att.name}' is required`;
                            } else if (isGraphAttr && Array.isArray(val) && val.length === 0) {
                                newErrors[attrKey] = `Attribute '${att.name}' requires at least one selected node`;
                            } else if (isTensorAttr) {
                                const tVal = val as any;
                                if (!tVal || (tVal.dataType === undefined && tVal.literalType === undefined)) {
                                    newErrors[attrKey] = `Attribute '${att.name}' requires a valid tensor definition`;
                                }
                            }
                        }

                        if (val !== undefined && val !== null && val !== "" && !isGraphAttr && !isTensorAttr) {
                            const patternStr = getPattern(att.type);
                            if (patternStr) {
                                const regex = new RegExp(patternStr);
                                const strVal = Array.isArray(val) ? val.join(",") : String(val);
                                if (!regex.test(strVal)) {
                                    newErrors[attrKey] = `Value '${strVal}' does not match expected format for ${AttributeType[att.type] ?? att.type}`;
                                }
                            }
                        }
                    });
                }
            }
        }

        return newErrors;
    };

    const handleCreateClick = () => {
        const validationErrors = validateForm();
        setErrors(validationErrors);

        if (Object.keys(validationErrors).length > 0) {
            return;
        }

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
            const parsedAttributes: Record<string, AttributeValue> = {};
            const constructedRegions: any[] = [];
            
            if (operationType?.attributes) {
                let regionIdx = 0;
                Object.values(operationType.attributes).forEach((attr, index) => {
                    const val = operationAttributes[index];
                    
                    if (val !== "" && val !== undefined && val !== null) {
                        if (attr.type === AttributeType.INT || attr.type === AttributeType.FLOAT) {
                            parsedAttributes[attr.name] = Number(val);
                        } else if (attr.type === AttributeType.INTS || attr.type === AttributeType.FLOATS) {
                            parsedAttributes[attr.name] = Array.isArray(val)
                                ? val.map(Number)
                                : String(val).split(",").map(Number);
                        } else if (attr.type === AttributeType.STRINGS) {
                            parsedAttributes[attr.name] = Array.isArray(val)
                                ? val.map(s => String(s).trim())
                                : String(val).split(",").map(s => s.trim());
                        } else if (attr.type === AttributeType.GRAPH || attr.type === AttributeType.GRAPHS) {
                            const nodeIds: string[] = Array.isArray(val)
                                ? val.map((v: any) => (typeof v === "string" ? v : (v?.value || v?.id || v?.data?.id || String(v))))
                                : typeof val === "string" && val.length > 0
                                ? val.split(",").map(s => s.trim()).filter(Boolean)
                                : [];
                            parsedAttributes[attr.name] = nodeIds;

                            const selectedIdSet = new Set(nodeIds);
                            const allNodes = (graphNodes || valueNodes || []) as any[];
                            const regionNodes = allNodes
                                .filter((n: any) => selectedIdSet.has(n?.data?.id || n?.id))
                                .map((n: any) => ({
                                    ...JSON.parse(JSON.stringify(n)),
                                    data: {
                                        ...(n?.data || {}),
                                        parent: nodeId,
                                        regionIndex: regionIdx,
                                    },
                                }));
                            constructedRegions.push({
                                elements: {
                                    nodes: regionNodes,
                                    edges: [],
                                },
                            });
                            regionIdx++;
                        } else if (
                            attr.type === AttributeType.TENSOR ||
                            attr.type === AttributeType.SPARSE_TENSOR ||
                            attr.type === AttributeType.TENSORS ||
                            attr.type === AttributeType.SPARSE_TENSORS
                        ) {
                            if (typeof val === "object" && val !== null) {
                                const tVal = val as any;
                                parsedAttributes[attr.name] = {
                                    name: attr.name,
                                    dataType: tVal.dataType ?? tVal.literalType ?? DataType.UNDEFINED,
                                    dims: tVal.dims ?? tVal.shape ?? [],
                                    ...tVal,
                                };
                            } else {
                                parsedAttributes[attr.name] = val;
                            }
                        } else {
                            parsedAttributes[attr.name] = val;
                        }
                    }
                });
            }

            const parsedInputs = operationInputs
                .filter(Boolean)
                .flatMap(input => input.split(",").map(i => i.trim()))
                .filter(Boolean);

            onnxData = {
                id: nodeId,
                kind: "OperationNode",
                opType: operationType!.opType,
                inputs: parsedInputs,
                regions: constructedRegions,
                attributes: parsedAttributes,
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
                case DataType.INT4:
                    selectedArray = "int32Data";
                    break;
                default:
                    selectedArray = "rawData";
                    break;
            }
            let processedProtoData: any[];
            if (selectedArray === "floatData" || selectedArray === "doubleData" || selectedArray === "int32Data") {
                processedProtoData = protoData.map((item) => Number(item)).filter((n) => !isNaN(n));
            } else if (selectedArray === "int64Data" || selectedArray === "uint64Data") {
                processedProtoData = protoData.map((item) => {
                    try {
                        return BigInt(item);
                    } catch {
                        return Number(item) || 0;
                    }
                });
            } else if (selectedArray === "stringData") {
                processedProtoData = protoData.map((item) => String(item));
            } else {
                processedProtoData = protoData;
            }
            onnxData.proto[selectedArray] = processedProtoData;
        }

        const nodePayload = {
            onnxData,
            label: nodeId,
            schemaOutputs: nodeKind === "operation" ? (operationType?.outputs || [{ name: "output" }]) : [],
        };

        if (onSubmit) {
            onSubmit(nodePayload, position || null);
        }
    };

    const hasErrors = Object.keys(errors).length > 0;

    return (
        <>
            {hasErrors && (
                <div
                    style={{
                        background: "rgba(255, 77, 79, 0.15)",
                        border: "1px solid #ff4d4f",
                        borderRadius: "8px",
                        padding: "8px 12px",
                        color: "#ff7875",
                        fontSize: "12px",
                        marginBottom: "10px",
                    }}
                >
                    Please fix the highlighted fields in red before creating/updating the node.
                </div>
            )}

            <label htmlFor="id">Node ID: *</label>
            <input
                name="id"
                type="text"
                value={nodeId}
                onChange={(e) => {
                    setNodeId(e.target.value);
                    if (errors.nodeId) {
                        setErrors(prev => {
                            const next = { ...prev };
                            delete next.nodeId;
                            return next;
                        });
                    }
                }}
                style={{
                    border: errors.nodeId ? "2px solid #ff4d4f" : "2px solid rgb(95, 92, 102)",
                    background: errors.nodeId ? "#321d23" : "#2c2a30",
                    color: "white",
                    borderRadius: "6px",
                    padding: "8px",
                    outline: "none",
                }}
            />
            {errors.nodeId && (
                <span style={{ color: "#ff7875", fontSize: "12px", marginTop: "-2px" }}>{errors.nodeId}</span>
            )}
            <button
                type="button"
                onClick={() => {
                    const randomId = `node_${Math.random().toString(36).substr(2, 9)}`;
                    setNodeId(randomId);
                    if (errors.nodeId) {
                        setErrors(prev => {
                            const next = { ...prev };
                            delete next.nodeId;
                            return next;
                        });
                    }
                }}
                style={{ marginTop: "4px", marginBottom: "8px" }}
            >
                Generate Random ID
            </button>

            <label color="white" htmlFor="kind">
                Node Kind: *
            </label>
            <Select
                id="kind"
                onChange={(e: any) => {
                    setNodeKind(e.value);
                    setErrors({});
                }}
                options={nodeKindOptions}
                value={nodeKindOptions.find(opt => opt.value === nodeKind) || null}
                styles={getReactSelectStyles(Boolean(errors.nodeKind))}
            />

            {nodeKind === "constant" && (
                <ConstantNodeAdder
                    reactSelectStyles={reactSelectCustomStyles}
                    constantProtoName={constantProtoName}
                    constantDataType={constantDataType}
                    constantShape={constantShape}
                    protoData={protoData}
                    setConstantProtoName={(v) => {
                        setConstantProtoName(v);
                        if (errors.constantProtoName) setErrors(prev => { const n = { ...prev }; delete n.constantProtoName; return n; });
                    }}
                    setConstantDataType={(v) => {
                        setConstantDataType(v);
                        if (errors.constantDataType) setErrors(prev => { const n = { ...prev }; delete n.constantDataType; return n; });
                    }}
                    setConstantShape={(v) => {
                        setConstantShape(v);
                        if (errors.constantShape || errors.protoData) setErrors(prev => { const n = { ...prev }; delete n.constantShape; delete n.protoData; return n; });
                    }}
                    setProtoData={(v) => {
                        setProtoData(v);
                        if (errors.protoData) setErrors(prev => { const n = { ...prev }; delete n.protoData; return n; });
                    }}
                    errors={errors}
                />
            )}

            {nodeKind === "operation" && (
                <OperationNodeAdder
                    reactSelectStyles={reactSelectCustomStyles}
                    setOperationType={(v) => {
                        setOperationType(v);
                        if (errors.operationType) setErrors(prev => { const n = { ...prev }; delete n.operationType; return n; });
                    }}
                    setOperationInputs={(v) => {
                        setOperationInputs(v);
                        setErrors(prev => {
                            const next = { ...prev };
                            Object.keys(next).filter(k => k.startsWith("input_")).forEach(k => delete next[k]);
                            return next;
                        });
                    }}
                    setOperationAttributes={(v) => {
                        setOperationAttributes(v);
                        setErrors(prev => {
                            const next = { ...prev };
                            Object.keys(next).filter(k => k.startsWith("attr_")).forEach(k => delete next[k]);
                            return next;
                        });
                    }}
                    valueNodes={valueNodes}
                    graphNodes={graphNodes || valueNodes}
                    operationType={operationType}
                    operationInputs={operationInputs}
                    operationAttributes={operationAttributes}
                    errors={errors}
                />
            )}

            {nodeKind === "tensor" && (
                <TensorNodeAdder
                    reactSelectStyles={reactSelectCustomStyles}
                    setTensorDataType={(v) => {
                        setTensorDataType(v);
                        if (errors.tensorDataType) setErrors(prev => { const n = { ...prev }; delete n.tensorDataType; return n; });
                    }}
                    setTensorShapeValue={(v) => {
                        setTensorShapeValue(v);
                        if (errors.tensorShape) setErrors(prev => { const n = { ...prev }; delete n.tensorShape; return n; });
                    }}
                    tensorShapeValue={tensorShapeValue}
                    setTensorKind={(v) => {
                        setTensorKind(v);
                        if (errors.tensorKind) setErrors(prev => { const n = { ...prev }; delete n.tensorKind; return n; });
                    }}
                    tensorDataType={tensorDataType}
                    tensorKind={tensorKind}
                    errors={errors}
                />
            )}
            <button type="button" onClick={handleCreateClick} style={{ marginTop: "15px" }}>
                {nodeToEdit ? "Edit Node" : "Create Node"}
            </button>
        </>
    );
}
