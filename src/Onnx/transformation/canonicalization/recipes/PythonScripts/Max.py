import onnx
from onnx import helper
from onnx import TensorProto

#Define the inputs and outputs
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2])
C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [2])

#Create the node 
min_node = helper.make_node(
    'Max',              #op_type
    inputs=['A', 'B'],  #inputs
    outputs=['C'],      #outputs
    name='MaxNode'      #node name
)

#Create the graph
graph = helper.make_graph(
    [min_node],     #nodes
    'MaxGraph',     #graph name
    [A, B],         #inputs
    [C]             #outputs
)

#Define the proper Opset Import
opset = onnx.OperatorSetIdProto()
opset.domain = ""
opset.version = 19

#Create the model
model = helper.make_model(
    graph, 
    producer_name='onnx-max-example',
    opset_imports=[opset]
)

#Set the IR version to a supported value
model.ir_version = 9

#Save the model
onnx.save(model, 'max.onnx')
print("Saved")