/**
 * @file OperationNodeAdder.tsx
 * @description Sub-form component for configuring an ONNX OperationNode.
 * Allows selecting standard ONNX operator schemas (StandardOps), binding input slots
 * to existing value nodes (with support for optional flags and variadic inputs),
 * configuring typed attributes (graph subgraphs, tensors, scalars, arrays), and validation feedback.
 */

import Select from "react-select";
import { StandardOps } from "@specs-feup/onnx-flow/Onnx/Schema/definitions/StandardOps";
import type { OpSchema, AttributeDefinition, IOInterface } from "@specs-feup/onnx-flow/Onnx/Schema/OpSchema";
import { AttributeType, DataType, type AttributeValue } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import { useEffect, useState } from "react";
import TensorNodeAdder from "./TensorNodeAdder.tsx";
import { getReactSelectStyles } from "@/styles/ReactSelectStyle.ts";

/**
 * Array of sorted ONNX operator schema options for the operation type selector.
 */
export const operationsTypes: Array<{ value: OpSchema; label: string }> = StandardOps.map((op) => ({
    value: op,
    label: op.opType,
})).sort((a, b) => a.label.localeCompare(b.label));

/**
 * Returns a regular expression validation pattern string corresponding to an AttributeType.
 *
 * @param type - The ONNX AttributeType enum value
 * @returns Regex pattern string for validating textual input, or undefined
 */
export function getPattern(type: AttributeType): string | undefined {
    switch (type) {
        default:
        case AttributeType.UNDEFINED:
            return ".*";
        case AttributeType.FLOAT:
            return "^-?\\d+(\\.\\d+)?([eE][+-]?\\d+)?$";
        case AttributeType.INT:
            return "^-?\\d+$";
        case AttributeType.STRING:
            return "^[a-zA-Z0-9_\\-\\.\\/\\s]+$";
        case AttributeType.FLOATS:
            return "^-?\\d+(\\.\\d+)?([eE][+-]?\\d+)?(,-?\\d+(\\.\\d+)?([eE][+-]?\\d+)?)*$";
        case AttributeType.INTS:
            return "^-?\\d+(,-?\\d+)*$";
        case AttributeType.STRINGS:
            return "^[a-zA-Z0-9_\\-\\.\\/\\s]+(,[a-zA-Z0-9_\\-\\.\\/\\s]+)*$";
    }
}

/**
 * Properties for the OperationNodeAdder component.
 */
interface OperationNodeAdderProps {
    /** Optional custom styles for ReactSelect */
    reactSelectStyles?: any;
    /** State setter for the active ONNX operator schema */
    setOperationType: (value: OpSchema) => void;
    /** State setter for the array of operation input node IDs */
    setOperationInputs: (value: string[]) => void;
    /** State setter for the array of operation attribute values */
    setOperationAttributes: (value: AttributeValue[]) => void;
    /** List of value nodes available for input binding */
    valueNodes: Array<unknown>;
    /** Full list of graph nodes for GRAPH attribute sub-region selection */
    graphNodes?: Array<unknown>;
    /** Currently selected ONNX operator schema */
    operationType: OpSchema;
    /** Current input bindings array */
    operationInputs: string[];
    /** Current attribute values array */
    operationAttributes: AttributeValue[];
    /** Object containing form validation error messages keyed by field name */
    errors?: Record<string, string>;
}

/**
 * Form component for creating or editing an ONNX OperationNode.
 *
 * @param props - OperationNodeAdder properties
 * @returns JSX element containing operator configuration inputs
 */
export default function OperationNodeAdder({

    reactSelectStyles,
    setOperationType,
    setOperationInputs,
    setOperationAttributes,
    valueNodes,
    graphNodes,
    operationType,
    operationInputs,
    operationAttributes,
    errors = {},
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

    const opTypeSelectStyles = getReactSelectStyles(Boolean(errors.operationType));

    return (
        <>
        <label htmlFor="operation-type">Operation Type: *</label>
        <Select 
            isSearchable
            isClearable
            name="operation-type"
            options={operationsTypes}
            defaultValue={operationsTypes[0]}
            styles={opTypeSelectStyles}
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
        {errors.operationType && (
            <span style={{ color: "#ff7875", fontSize: "12px", marginTop: "-2px" }}>{errors.operationType}</span>
        )}
        
        <label htmlFor="inputs" style={{ marginTop: "8px", fontWeight: "bold" }}>Inputs:</label>
        {operationType.inputs.map((input: IOInterface, index: number) => {
            const isOptional = input.optional || false;
            const isRequired = requiredInputs[index];
            const inputErrorKey = `input_${input.name || index}`;
            const hasInputError = Boolean(errors[inputErrorKey]);
            const inputSelectStyles = getReactSelectStyles(hasInputError);

            return (
                <div key={input.name || index} style={{display: "flex", flexDirection: "column", gap: "5px", backgroundColor: "rgb(77, 74, 85)", padding: "1em", borderRadius: "20px", border: hasInputError ? "1px solid #ff4d4f" : "none"}}>
                    <label>Name: {input.name} {isRequired && "*"}</label>
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
                        styles={inputSelectStyles}
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
                        styles={inputSelectStyles}
                        name={`input-${input.name}`}
                    />}
                    {errors[inputErrorKey] && (
                        <span style={{ color: "#ff7875", fontSize: "12px", marginTop: "2px" }}>{errors[inputErrorKey]}</span>
                    )}
                </div>
            )})
        }
        
        {Object.values(operationType.attributes).length > 0 && (
            <label htmlFor="attributes" style={{ marginTop: "10px", fontWeight: "bold" }}>Attributes</label>
        )}
        {Object.values(operationType.attributes).map((att: AttributeDefinition, index: number) => {
            const isAttrDisabled = att.required ? false : disabledAttr[index];
            const currentVal = operationAttributes[index];
            const attrErrorKey = `attr_${att.name}`;
            const hasAttrError = Boolean(errors[attrErrorKey]);
            const attrSelectStyles = getReactSelectStyles(hasAttrError);

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
                        border: hasAttrError ? "1px solid #ff4d4f" : "none",
                    }}
                >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <label style={{ fontWeight: "bold" }}>{att.name} {att.required && "*"}</label>
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
                            styles={attrSelectStyles}
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
                            errors={{
                                tensorDataType: errors[`${attrErrorKey}_dataType`],
                                tensorShape: errors[`${attrErrorKey}_shape`],
                            }}
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
                            style={{
                                border: hasAttrError ? "2px solid #ff4d4f" : "2px solid rgb(95, 92, 102)",
                                background: hasAttrError ? "#321d23" : "#2c2a30",
                                color: "white",
                                borderRadius: "6px",
                                padding: "8px",
                                outline: "none",
                            }}
                            onChange={(e) => {
                                const newAttributes = [...operationAttributes];
                                newAttributes[index] = e.target.value;
                                setOperationAttributes(newAttributes);
                            }}
                        />
                    )}
                    {errors[attrErrorKey] && (
                        <span style={{ color: "#ff7875", fontSize: "12px", marginTop: "2px" }}>{errors[attrErrorKey]}</span>
                    )}
                </div>
            );
        })}
        </>
    )
}