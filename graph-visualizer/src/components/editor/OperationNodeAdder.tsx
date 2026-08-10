import Select from "react-select";
import { StandardOps } from "@specs-feup/onnx-flow/Onnx/Schema/definitions/StandardOps";
import type { OpSchema, AttributeDefinition, IOInterface } from "@specs-feup/onnx-flow/Onnx/Schema/OpSchema";
import { AttributeType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";

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
            return (
                <div key={input.name || index} style={{display: "flex", flexDirection: "column", gap: "5px", backgroundColor: "rgb(77, 74, 85)", padding: "1em", borderRadius: "20px"}}>
                    <label>Name: {input.name}</label>
                    <label>Constraint: {input.typeConstraint}</label>
                    <label>Is Variadic: {input.variadic ? "Yes" : "No"}</label>
                    <label>Is Optional: {input.optional ? "Yes" : "No"}</label>
                    <Select
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
                    />
                    <br/>
                </div>
            )})
        }
        
        <label htmlFor="attributes">Attributes</label>
        {Object.values(operationType.attributes).map((att: AttributeDefinition) => {
            let inputField;
            
            switch (att.type) {
                case AttributeType.UNDEFINED:
                case AttributeType.STRING:
                    inputField = <input type="text" name={att.name} defaultValue={att.defaultValue !== undefined ? att.defaultValue : ""} />;
                    break;
                case AttributeType.FLOAT:
                case AttributeType.INT:
                    inputField = <input type="number" name={att.name} defaultValue={att.defaultValue !== undefined ? att.defaultValue : ""} />;
                    break;
                default:
                    inputField = <p>{AttributeType[att.type]}</p>;
                    break;
            }

            return (
                <div key={att.name}>
                    <label>{att.name}</label>
                    {inputField}
                </div>
            );
        })}
        </>
    )
}