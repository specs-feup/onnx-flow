import onnx
from onnx import helper
from onnx import TensorProto

#Define the input and outputs
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2])
C = helper.make_tensor_value_info('C', TensorProto.BOOL, [2])

# Create the NotEqual node
not_equal_node = helper.make_node(
    'NotEqual',         #node
    inputs=['A', 'B'],  #inputs
    outputs=['C'],      #outputs
    name='NotEqualNode'
)

# Create the graph
graph = helper.make_graph(
    [not_equal_node],   #nodes
    'NotEqualGraph',    #graph name
    [A, B],             #inputs
    [C]                 #outputs
)

# Define the proper Opset Import
opset = onnx.OperatorSetIdProto()
opset.domain = ""
opset.version = 19

# Create the model
model = helper.make_model(
    graph,
    producer_name='onnx-notequal-example',
    opset_imports=[opset]
)

# Set the IR version to a supported value
model.ir_version = 9

# Save the onnx model
onnx.save(model, 'not_equal.onnx')
print("Save")