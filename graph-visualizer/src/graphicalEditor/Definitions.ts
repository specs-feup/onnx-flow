import type OnnxGraph from "../../../src/Onnx/OnnxGraph";
import { DataType, type AttributeMap, type Shape, type TensorProto, type ValueNode } from "../../../src/Onnx/OnnxTypes";
import type TensorNode from "../../../src/Onnx/TensorNode";

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

interface TensorOnnxData {
    id: string,
    kind: string,
    tensorType: TensorNode.TensorKind,
    literalType: DataType,
    shape: Shape,
    metadata: AttributeMap,
}

interface OperationOnnxData {
    id: string,
    kind: string,
    opType: string,
    attributes: AttributeMap | undefined,
    inputs: string[],
    regions: OnnxGraph.Class[] | undefined,
    metadata: AttributeMap,
}

interface ConstantOnnxData {
    id: string,
    kind: string,
    isInput: boolean,
    proto: TensorProto,
    metadata: AttributeMap,
}

export type OnnxData = TensorOnnxData | OperationOnnxData | ConstantOnnxData;