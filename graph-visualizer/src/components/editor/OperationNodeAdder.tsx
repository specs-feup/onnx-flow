import Select from "react-select";
import { StandardOps } from "@specs-feup/onnx-flow/Onnx/Schema/definitions/StandardOps";
import type { OpSchema, AttributeDefinition, IOInterface } from "@specs-feup/onnx-flow/Onnx/Schema/OpSchema";
import { AttributeType, type AttributeValue } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import { useEffect, useState } from "react";

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
    setOperationAttributes: (value: AttributeValue[]) => void;
    valueNodes: Array<unknown>;
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
    operationType,
    operationInputs,
    operationAttributes
}: OperationNodeAdderProps) {
    const [requiredInputs, setRequiredInputs] = useState<boolean[]>(operationType.inputs.map((input) => input.optional === undefined ? true : !input.optional));
    const [disabledAttr, setDisabledAttr] = useState<boolean[]>(operationType.attributes ? Object.values(operationType.attributes).map(() => {
            return false;
        }) : [])

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
            <label htmlFor="attributes">Attributes</label>
        )}
        {Object.values(operationType.attributes).map((att: AttributeDefinition, index: number) => {
            let pattern;
            
            switch (att.type) {
                default:
                case AttributeType.UNDEFINED:
                    pattern = ".*";
                    break;
                case AttributeType.FLOAT:
                    pattern = "^-?[1-9]\d*\.\d+$"; 
                    break;
                case AttributeType.INT:
                    pattern = "^-?[1-9]\d*$";
                    break;
                case AttributeType.STRING:
                    pattern = "^[a-zA-Z0-9_\-]+$";
                    break;
                case AttributeType.FLOATS:
                    pattern = "^-?[1-9]\d*\.\d+(,-?[1-9]\d*\.\d+)*$";
                    break;
                case AttributeType.INTS:
                    pattern = "^-?[1-9]\d*(,-?[1-9]\d*)*$";
                    break;
                case AttributeType.STRINGS:
                    pattern = "^[a-zA-Z0-9_\-]+(,[a-zA-Z0-9_\-]+)*$" 
                    break;
            }

            console.log(StandardOps.find(op => op.opType === operationType.opType));

            return (
                <div key={att.name} title={att.description}>
                    <label>{att.name}</label>
                    <label>Type: {AttributeType[att.type]}</label>
                    <input 
                        type="text"
                        pattern={pattern}
                        disabled={att.required ? false : disabledAttr[index]}
                        name={att.name}
                        value={
                            operationAttributes[index]
                        } 
                        defaultValue={att.defaultValue !== undefined ? att.defaultValue : ""}
                        onChange={(e) => {
                            const newAttributes = [...operationAttributes];
                            newAttributes[index] = e.target.value;
                            setOperationAttributes(newAttributes);
                        }}
                    />
                    <input
                        type="checkbox"
                        defaultChecked={true}
                        disabled={att.required ? true : false}
                        onChange={(e) => {
                            const newAttributes = [...operationAttributes];
                            const newDisabledAttr = [...disabledAttr];
                            newDisabledAttr[index] = !e.target.checked;
                            if (!e.target.checked) {
                                newAttributes[index] = "";
                            } else {
                                newAttributes[index] = att.defaultValue !== undefined ? att.defaultValue : "";
                            }
                            setOperationAttributes(newAttributes);
                            setDisabledAttr(newDisabledAttr);
                        }}
                    />
                </div>
            );
        })}
        </>
    )
}