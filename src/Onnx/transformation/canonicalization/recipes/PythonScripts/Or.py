import onnx
from onnx import helper
from onnx import TensorProto

#Define the input and outputs
A = helper.make_tensor_value_info('A', TensorProto.BOOL, [2])
B = helper.make_tensor_value_info('B', TensorProto.BOOL, [2])
C = helper.make_tensor_value_info('C', TensorProto.BOOL, [2])

# Create the Or node
or_node = helper.make_node(
    'Or',               #node
    inputs=['A', 'B'],  #inputs
    outputs=['C'],      #outputs
    name='OrNode'       #name
)

# Create the graph
or_graph = helper.make_graph(
    [or_node],          #nodes
    'OrGraph',          #graph name
    [A, B],             #inputs
    [C]                 #outputs
)

# Define the proper Opset Import
opset = onnx.OperatorSetIdProto()
opset.domain = ""
opset.version = 19

# Create the model
model = helper.make_model(
    or_graph,
    producer_name='onnx-or-example',
    opset_imports=[opset]
)

# Set the IR version to a supported value
model.ir_version = 9

# Save the onnx model
onnx.save(model, '../examples/onnx/or.onnx')
print("Saved")