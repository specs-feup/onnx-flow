import Select from "react-select";
import { StandardOps } from "@specs-feup/onnx-flow/Onnx/Schema/definitions/StandardOps";
import type { OpSchema, AttributeDefinition, IOInterface } from "@specs-feup/onnx-flow/Onnx/Schema/OpSchema";
import { AttributeType, DataType, type AttributeValue } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import { useEffect, useState } from "react";
import TensorNodeAdder from "./TensorNodeAdder.tsx";

/*--- OPERATION OPTIONS ---*/
export const operationsTypes: Array<{ value: OpSchema; label: string }> = StandardOps.map((op) => ({
    value: op,
    label: op.opType,
})).sort((a, b) => a.label.localeCompare(b.label));

function getPattern(type: AttributeType): string | undefined {
    switch (type) {
        default:
        case AttributeType.UNDEFINED:
            return ".*";
        case AttributeType.FLOAT:
            return "^-?\\d+(\\.\\d+)?$";
        case AttributeType.INT:
            return "^-?\\d+$";
        case AttributeType.STRING:
            return "^[a-zA-Z0-9_\\-]+$";
        case AttributeType.FLOATS:
            return "^-?\\d+(\\.\\d+)?(,-?\\d+(\\.\\d+)?)*$";
        case AttributeType.INTS:
            return "^-?\\d+(,-?\\d+)*$";
        case AttributeType.STRINGS:
            return "^[a-zA-Z0-9_\\-]+(,[a-zA-Z0-9_\\-]+)*$";
    }
}

/*---   ---*/
interface OperationNodeAdderProps {
    reactSelectStyles: any;
    setOperationType: (value: OpSchema) => void;
    setOperationInputs: (value: string[]) => void;
    setOperationAttributes: (value: AttributeValue[]) => void;
    valueNodes: Array<unknown>;
    graphNodes?: Array<unknown>;
    operationType: OpSchema;
    operationInputs: string[];
    operationAttributes: AttributeValue[];
}

export default function OperationNodeAdder({
    reactSelectStyles,
    setOperationType,
    setOperationInputs,
    setOperationAttributes,
    valueNodes,
    graphNodes,
    operationType,
    operationInputs,
    operationAttributes
}: OperationNodeAdderProps) {
    const [requiredInputs, setRequiredInputs] = useState<boolean[]>(operationType.inputs.map((input) => input.optional === undefined ? true : !input.optional));
    const [disabledAttr, setDisabledAttr] = useState<boolean[]>(operationType.attributes ? Object.values(operationType.attributes).map(() => {
            return false;
        }) : []);

    const allGraphNodes = (graphNodes && graphNodes.length > 0 ? graphNodes : valueNodes) || [];
    const graphNodeOptions = Array.from(
        new Map(
            allGraphNodes
                .filter((node: any) => Boolean(node?.data?.id || node?.id))
                .map((node: any) => {
                    const id = node?.data?.id || node?.id;
                    return [id, { value: id, label: id }];
                })
        ).values()
    );

    useEffect(() => {
        setRequiredInputs(operationType.inputs.map((input) => input.optional === undefined ? true : !input.optional));

        const requiredAttrArray = operationType.attributes ? Object.values(operationType.attributes).map(() => {
            return false;
        }) : [];
        setDisabledAttr(requiredAttrArray);
    }, [operationType]);

    return (
        <>
        <label htmlFor="operation-type">Operation Type:</label>
        <Select 
            isSearchable
            isClearable
            name="operation-type"
            options={operationsTypes}
            defaultValue={operationsTypes[0]}
            styles={reactSelectStyles}
            onChange={(op) => {
                 const newOp = op!.value;
                 setOperationType(newOp);
                 
                 setOperationInputs([]);
                 const newAttributes = newOp.attributes ? Object.values(newOp.attributes).map((att) => {
                     if (att.type === AttributeType.GRAPH || att.type === AttributeType.GRAPHS) {
                         return [];
                     }
                     if (
                         att.type === AttributeType.TENSOR ||
                         att.type === AttributeType.SPARSE_TENSOR ||
                         att.type === AttributeType.TENSORS ||
                         att.type === AttributeType.SPARSE_TENSORS
                     ) {
                         return {
                             dataType: DataType.UNDEFINED,
                             dims: [],
                         };
                     }
                     return att.defaultValue !== undefined ? att.defaultValue : "";
                 }) : [];
                 setOperationAttributes(newAttributes);
             }}
            value={operationsTypes.find(op => op.value === operationType) || null} 
        />
        
        <label htmlFor="inputs">Inputs:</label>
        {operationType.inputs.map((input: IOInterface, index: number) => {
            const isOptional = input.optional || false;
            const isRequired = requiredInputs[index];

            return (
                <div key={input.name || index} style={{display: "flex", flexDirection: "column", gap: "5px", backgroundColor: "rgb(77, 74, 85)", padding: "1em", borderRadius: "20px"}}>
                    <label>Name: {input.name}</label>
                    <label>Constraint: {input.typeConstraint}</label>
                    <label>Is Variadic: {input.variadic ? "Yes" : "No"}</label>
                    <label>Required?</label>
                    <input 
                        type="checkbox"
                        checked={!isOptional ? true : undefined}
                        disabled={!isOptional}
                        onChange={(e) => {
                            if (!e.target.checked) {
                                const newInputs = [...operationInputs];
                                newInputs[index] = "";
                                setOperationInputs(newInputs);
                            }
                            setRequiredInputs((prevRequiredInputs) => {
                                const newRequiredInputs = [...prevRequiredInputs];
                                newRequiredInputs[index] = e.target.checked;
                                return newRequiredInputs;
                            });
                        }}
                    />
                    {input.variadic ? 
                        <Select
                        isDisabled={!isRequired} 
                        isSearchable
                        isClearable
                        isMulti
                        options={valueNodes.map((node: any) => ({
                            value: node.data.id,
                            label: node.data.id,
                        }))}
                        onChange={(selectedInputs) => {
                            const newInputs = [...operationInputs];
                            newInputs[index] = selectedInputs ? selectedInputs.map((input) => input.value).join(",") : "";
                            setOperationInputs(newInputs);
                        }}
                        value={operationInputs[index] ? operationInputs[index].split(",").map((inputId) => ({ value: inputId, label: inputId })) : []}
                        styles={reactSelectStyles}
                        name={`input-${input.name}`}
                        /> 
                        : 
                        <Select
                        isDisabled={!isRequired}
                        isSearchable
                        isClearable
                        options={valueNodes.map((node: any) => ({
                            value: node.data.id,
                            label: node.data.id,
                        }))}
                        onChange={(selectedInput) => {
                            const newInputs = [...operationInputs];
                            newInputs[index] = selectedInput ? selectedInput.value : "";
                            setOperationInputs(newInputs);
                        }}
                        value={operationInputs[index] ? { value: operationInputs[index], label: operationInputs[index] } : null}
                        styles={reactSelectStyles}
                        name={`input-${input.name}`}
                    />}
                    <br/>
                </div>
            )})
        }
        
        {Object.values(operationType.attributes).length > 0 && (
            <label htmlFor="attributes" style={{ marginTop: "10px", fontWeight: "bold" }}>Attributes</label>
        )}
        {Object.values(operationType.attributes).map((att: AttributeDefinition, index: number) => {
            const isAttrDisabled = att.required ? false : disabledAttr[index];
            const currentVal = operationAttributes[index];

            const isGraphAttr = att.type === AttributeType.GRAPH || att.type === AttributeType.GRAPHS;
            const isTensorAttr =
                att.type === AttributeType.TENSOR ||
                att.type === AttributeType.SPARSE_TENSOR ||
                att.type === AttributeType.TENSORS ||
                att.type === AttributeType.SPARSE_TENSORS;

            return (
                <div
                    key={att.name}
                    title={att.description}
                    style={{
                        display: "flex",
                        flexDirection: "column",
                        gap: "6px",
                        backgroundColor: "rgb(60, 58, 68)",
                        padding: "12px",
                        borderRadius: "12px",
                        marginBottom: "10px",
                    }}
                >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <label style={{ fontWeight: "bold" }}>{att.name}</label>
                        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                            <label style={{ fontSize: "12px" }}>Enabled:</label>
                            <input
                                type="checkbox"
                                checked={att.required ? true : !disabledAttr[index]}
                                disabled={att.required ? true : false}
                                onChange={(e) => {
                                    const newAttributes = [...operationAttributes];
                                    const newDisabledAttr = [...disabledAttr];
                                    newDisabledAttr[index] = !e.target.checked;
                                    if (!e.target.checked) {
                                        newAttributes[index] = "";
                                    } else {
                                        if (isGraphAttr) {
                                            newAttributes[index] = [];
                                        } else if (isTensorAttr) {
                                            newAttributes[index] = {
                                                dataType: DataType.UNDEFINED,
                                                dims: [],
                                            };
                                        } else {
                                            newAttributes[index] = att.defaultValue !== undefined ? att.defaultValue : "";
                                        }
                                    }
                                    setOperationAttributes(newAttributes);
                                    setDisabledAttr(newDisabledAttr);
                                }}
                            />
                        </div>
                    </div>
                    <label style={{ fontSize: "12px", color: "#bbb" }}>Type: {AttributeType[att.type] ?? att.type}</label>

                    {isGraphAttr ? (
                        <Select
                            isMulti
                            isSearchable
                            isClearable
                            isDisabled={isAttrDisabled}
                            name={`attr-${att.name}`}
                            options={graphNodeOptions}
                            styles={reactSelectStyles}
                            placeholder="Select graph nodes..."
                            value={
                                Array.isArray(currentVal)
                                    ? (currentVal as any[]).map((val: any) => {
                                          const id = typeof val === "string" ? val : (val?.value || val?.id || val?.data?.id || String(val));
                                          return { value: id, label: id };
                                      })
                                    : typeof currentVal === "string" && currentVal !== ""
                                    ? (currentVal as string).split(",").filter(Boolean).map((id) => ({ value: id.trim(), label: id.trim() }))
                                    : []
                            }
                            onChange={(selectedOptions) => {
                                const newAttributes = [...operationAttributes];
                                newAttributes[index] = selectedOptions ? (selectedOptions as any[]).map((opt) => opt.value) : [];
                                setOperationAttributes(newAttributes);
                            }}
                        />
                    ) : isTensorAttr ? (
                        <TensorNodeAdder
                            reactSelectStyles={reactSelectStyles}
                            setTensorDataType={(newDataType) => {
                                const newAttributes = [...operationAttributes];
                                const prev = (typeof newAttributes[index] === "object" && newAttributes[index] !== null)
                                    ? newAttributes[index]
                                    : {};
                                newAttributes[index] = {
                                    ...prev,
                                    dataType: newDataType,
                                    literalType: newDataType,
                                    dims: prev.dims ?? prev.shape ?? [],
                                    shape: prev.shape ?? prev.dims ?? [],
                                };
                                setOperationAttributes(newAttributes);
                            }}
                            setTensorShapeValue={(newShape) => {
                                const newAttributes = [...operationAttributes];
                                const prev = (typeof newAttributes[index] === "object" && newAttributes[index] !== null)
                                    ? newAttributes[index]
                                    : {};
                                newAttributes[index] = {
                                    ...prev,
                                    dims: newShape,
                                    shape: newShape,
                                    dataType: prev.dataType ?? prev.literalType ?? DataType.UNDEFINED,
                                    literalType: prev.literalType ?? prev.dataType ?? DataType.UNDEFINED,
                                };
                                setOperationAttributes(newAttributes);
                            }}
                            tensorDataType={
                                (typeof currentVal === "object" && currentVal !== null)
                                    ? ((currentVal as any).dataType ?? (currentVal as any).literalType ?? DataType.UNDEFINED)
                                    : DataType.UNDEFINED
                            }
                            tensorShapeValue={
                                (typeof currentVal === "object" && currentVal !== null)
                                    ? ((currentVal as any).dims ?? ((currentVal as any).shape ?? []))
                                    : []
                            }
                            showTensorKind={false}
                            disabled={isAttrDisabled}
                        />
                    ) : (
                        <input 
                            type="text"
                            pattern={getPattern(att.type)}
                            disabled={isAttrDisabled}
                            name={att.name}
                            value={
                                typeof currentVal === "string" || typeof currentVal === "number"
                                    ? currentVal
                                    : Array.isArray(currentVal)
                                    ? currentVal.join(",")
                                    : ""
                            } 
                            onChange={(e) => {
                                const newAttributes = [...operationAttributes];
                                newAttributes[index] = e.target.value;
                                setOperationAttributes(newAttributes);
                            }}
                        />
                    )}
                </div>
            );
        })}
        </>
    )
}