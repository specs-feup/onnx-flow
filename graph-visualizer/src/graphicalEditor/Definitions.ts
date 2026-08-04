import { DataType } from "../../../src/Onnx/OnnxTypes";

interface DataTypeOption {
    value: DataType;
    label: string;
}

export const dataTypeOptions: DataTypeOption[] = Object.entries(DataType)
    .filter(([key]) => isNaN(Number(key)))
    .map(([key, val]) => ({
        value: val as DataType,
        label: key,
        color: 'white',
    })
);