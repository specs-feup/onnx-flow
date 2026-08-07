import Select from "react-select";
import { StandardOps } from "@specs-feup/onnx-flow/Onnx/Schema/definitions/StandardOps";

/*--- OPERATION OPTIONS ---*/
const operationsTypes: Array<{ value: string; label: string }> = StandardOps.map((op) => ({
    value: op.opType,
    label: op.opType,
})).sort((a, b) => a.label.localeCompare(b.label));
/*---   ---*/

interface OperationNodeAdderProps {
    reactSelectStyles: any;
    setOperationType: (value: string) => void;
    setOperationInputs: (value: string[]) => void;
    valueNodes: Array<unknown>;
}

export default function OperationNodeAdder({
    reactSelectStyles,
    setOperationType,
    setOperationInputs,
    valueNodes,
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
        />

        <label htmlFor="inputs">Inputs:</label>
        <Select
            isSearchable
            isClearable
            isMulti
            options={valueNodes.map((node: any) => ({
                value: node.data.id,
                label: node.data.id,
            }))}
            onChange={(inputs) => setOperationInputs(inputs.map((input: any) => input.value))}
            styles={reactSelectStyles}
            name="inputs"
        />

        </>  
    )
}