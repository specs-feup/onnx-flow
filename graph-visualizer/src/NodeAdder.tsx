import { useState } from "react";
import Select from "react-select";
import { StandardOps } from '../../src/Onnx/Schema/definitions/StandardOps/index.ts'

const operationsTypes: Array<{ value: string; label: string }> = StandardOps.map((op) => ({
    value: op.opType,
    label: op.opType,
})).sort((a, b) => a.label.localeCompare(b.label));

const customStyles = {
  option: (provided: any, state: any) => ({
    ...provided,
    color: 'black', 
  }),
};


export default function NodeAdder() {
    const [nodeKind, setNodeKind] = useState<string>("constant");

    const [operationType, setOperationType] = useState<string>("");


    return (
        <aside style={{display: 'flex', flexDirection: 'column'}}>
            <label htmlFor="kind">Node Kind: </label>
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

            {nodeKind === 'operation' && (
                <>
                <label htmlFor="operation-type">Operation Type:</label>
                <Select 
                    isSearchable
                    isClearable
                    name="operation-type"
                    options={operationsTypes}
                    styles={customStyles}
                />
                </>
            )}
        </aside>
    );
}
