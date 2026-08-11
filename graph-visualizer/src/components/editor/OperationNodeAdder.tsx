import Select from "react-select";
import { StandardOps } from "@specs-feup/onnx-flow/Onnx/Schema/definitions/StandardOps";
import type { OpSchema, AttributeDefinition, IOInterface } from "@specs-feup/onnx-flow/Onnx/Schema/OpSchema";
import { AttributeType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import { useState } from "react";

/*--- OPERATION OPTIONS ---*/
export const operationsTypes: Array<{ value: OpSchema; label: string }> = StandardOps.map((op) => ({
    value: op,
    label: op.opType,
})).sort((a, b) => a.label.localeCompare(b.label));

/*---   ---*/
interface OperationNodeAdderProps {
    reactSelectStyles: any;
    setOperationType: (value: OpSchema) => void;
    setOperationInputs: (value: string[]) => void;
    valueNodes: Array<unknown>;
    operationType: OpSchema;
    operationInputs: string[];
}

export default function OperationNodeAdder({
    reactSelectStyles,
    setOperationType,
    setOperationInputs,
    valueNodes,
    operationType,
    operationInputs,
}: OperationNodeAdderProps) {
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
            onChange={(op) => setOperationType(op!.value)}
            value={operationsTypes.find(op => op.value === operationType) || null} 
        />
        
        <label htmlFor="inputs">Inputs:</label>
        {operationType.inputs.map((input: IOInterface, index: number) => {
            const [isRequired, setIsRequired] = useState(!input.optional);
            return (
                <div key={input.name || index} style={{display: "flex", flexDirection: "column", gap: "5px", backgroundColor: "rgb(77, 74, 85)", padding: "1em", borderRadius: "20px"}}>
                    <label>Name: {input.name}</label>
                    <label>Constraint: {input.typeConstraint}</label>
                    <label>Is Variadic: {input.variadic ? "Yes" : "No"}</label>
                    <label>Required?</label>
                    <input 
                        type="checkbox"
                        checked={!input.optional ? true : undefined}
                        disabled={!input.optional}
                        onChange={(e) => {
                            if (!e.target.checked) {
                                const newInputs = [...operationInputs];
                                newInputs[index] = "";
                                setOperationInputs(newInputs);
                            }
                            setIsRequired(e.target.checked);
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
            <label htmlFor="attributes">Attributes</label>
        )}
        {Object.values(operationType.attributes).map((att: AttributeDefinition) => {
            let inputField;
            
            switch (att.type) {
                case AttributeType.UNDEFINED:
                    inputField = 
                        <input 
                            type="text"
                            name={att.name} 
                            defaultValue={att.defaultValue !== undefined ? att.defaultValue : ""} 
                        />;
                    break;
                case AttributeType.FLOAT:
                    inputField = 
                        <input 
                            type="text"
                            pattern="^-?[1-9]\d*\.\d+$" 
                            name={att.name} 
                            defaultValue={att.defaultValue !== undefined ? att.defaultValue : ""} 
                        />;
                    break;
                case AttributeType.INT:
                    inputField = 
                        <input 
                            type="text"
                            pattern="^-?[1-9]\d*$" 
                            name={att.name} 
                            defaultValue={att.defaultValue !== undefined ? att.defaultValue : ""} 
                        />;
                    break;
                case AttributeType.STRING:
                    inputField = 
                        <input 
                            type="text"
                            pattern="^[a-zA-Z0-9_\-]+$" 
                            name={att.name} 
                            defaultValue={att.defaultValue !== undefined ? att.defaultValue : ""} 
                        />;
                    break;
                case AttributeType.FLOATS:
                    inputField = 
                        <input 
                            type="text"
                            pattern="^-?[1-9]\d*\.\d+(,-?[1-9]\d*\.\d+)*$" 
                            name={att.name} 
                            defaultValue={att.defaultValue !== undefined ? att.defaultValue : ""} 
                        />;
                    break;
                case AttributeType.INTS:
                    inputField = 
                        <input 
                            type="text"
                            pattern="^-?[1-9]\d*(,-?[1-9]\d*)*$" 
                            name={att.name} 
                            defaultValue={att.defaultValue !== undefined ? att.defaultValue : ""} 
                        />;
                    break;
                case AttributeType.STRINGS:
                    inputField = 
                        <input 
                            type="text"
                            pattern="^[a-zA-Z0-9_\-]+(,[a-zA-Z0-9_\-]+)*$" 
                            name={att.name} 
                            defaultValue={att.defaultValue !== undefined ? att.defaultValue : ""} 
                        />;
                    break;
                default: // TODO: Handle TENSOR, GRAPH and any related types
                    inputField = <p>{AttributeType[att.type]}</p>;
                    break;
            }

            console.log(StandardOps)

            return (
                <div key={att.name} title={att.description}>
                    <label>{att.name}</label>
                    {inputField}
                </div>
            );
        })}
        </>
    )
}