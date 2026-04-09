import onnx
from onnx import helper
from onnx import TensorProto

#Define I/O
A=helper.make_tensor_value_info('A', TensorProto.FLOAT, [2])
B=helper.make_tensor_value_info('B', TensorProto.FLOAT, [2])
C=helper.make_tensor_value_info('C', TensorProto.FLOAT, [2])

#Create Mul node
mul_node = helper.make_node(
    'Mul',
    ['A','B'],
    ['C'],
    name='MulNode'
)

#Create Graph
graph = helper.make_graph(
    [mul_node],
    'MulGraph',
    [A,B],
    [C]
)

#Define version (Opset import)
opset = onnx.OperatorSetIdProto()
opset.domain = ""
opset.version = 19

#Create the model
model = helper.make_model(
    graph,
    producer_name='onnx-mul-example',
    opset_imports=[opset]
)

#Ir version to supported value
model.ir_version = 9

#Save the model
onnx.save(model,'mul.onnx')