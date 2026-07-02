import onnx
from onnx import helper
from onnx import TensorProto

#Define the input and outputs
A = helper.make_tensor_value_info('A', TensorProto.BOOL, [2])
B = helper.make_tensor_value_info('B', TensorProto.BOOL, [2])

# Create the Not node
not_node = helper.make_node(
    'Not',              #node
    inputs=['A'],       #inputs
    outputs=['B'],      #outputs
    name='NotNode'      #name
)

# Create the graph
not_graph = helper.make_graph(
    [not_node],         #nodes
    'NotGraph',         #graph name
    [A],                #inputs
    [B]                 #outputs
)

# Define the proper Opset Import
opset = onnx.OperatorSetIdProto()
opset.domain = ""
opset.version = 19

# Create the model
model = helper.make_model(
    not_graph,
    producer_name='onnx-not-example',
    opset_imports=[opset]
)

# Set the IR version to a supported value
model.ir_version = 9

# Save the onnx model
onnx.save(model, '../examples/onnx/Not.onnx')
print("Saved")


