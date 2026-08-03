import Select from "react-select";

import type { TensorProto } from "../../../src/Onnx/OnnxTypes";

interface ConstantNodeAdderProps {
    setConstantTensorProto: (value: TensorProto) => void;
    setIsConstantInput: (value: boolean) => void;
}

export default function ConstantNodeAdder({
    setConstantTensorProto,
    setIsConstantInput,
}: ConstantNodeAdderProps) {
    /*
    -- value: TensorProto;
    isInput: boolean;
    metadata: AttributeMap; 
    */
    return (
        <>
        <label htmlFor="isInput">Is Input?:</label>
        <input 
            type="checkbox" 
            name="isInput" 
            onChange={(e) => setIsConstantInput(e.target.checked)}
        />

        <label htmlFor="tensorProto">Tensor Proto</label>
        <label htmlFor="tensorName"> Tensor Name:</label>
        <input type="text" name="tensorName" />

        <Select />


        </>
    )
}