import onnx
from onnx import helper
from onnx import TensorProto

#Define the input and outputs
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2])
C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [2])

# Create the Mod node
mod_node = helper.make_node(
    'Mod',              #node
    inputs=['A', 'B'],  #inputs
    outputs=['C'],      #outputs
    name='ModNode',     #name
    fmod=1
)

# Create the graph
mod_graph = helper.make_graph(
    [mod_node],         #nodes
    'ModGraph',         #graph name
    [A, B],             #inputs
    [C]                 #outputs
)

# Define the proper Opset Import
opset = onnx.OperatorSetIdProto()
opset.domain = ""
opset.version = 19

# Create the model
model = helper.make_model(
    mod_graph,
    producer_name='onnx-mod-example',
    opset_imports=[opset]
)

# Set the IR version to a supported value
model.ir_version = 9

# Save the onnx model
onnx.save(model, '../examples/onnx/mod.onnx')
print("Saved")
