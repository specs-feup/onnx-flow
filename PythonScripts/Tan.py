import onnx
from onnx import helper
from onnx import TensorProto

#Define the input and outputs
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2])

#Create the node
node_def = helper.make_node(
    'Tan',              #node
    inputs=['A'],       #inputs
    outputs=['B'],      #outputs
    name='TanNode'      #name
)

# Create the graph
graph_def = helper.make_graph(
    [node_def],         #nodes
    'TanGraph',         #graph name
    [A],                #inputs
    [B]                 #outputs
)

# Define the proper Opset Import
opset = onnx.OperatorSetIdProto()
opset.domain = ""
opset.version = 19

# Create the model
model_def = helper.make_model(
    graph_def,
    producer_name='onnx-tan-example',
    opset_imports=[opset]
)

# Set the IR version to a supported value
model_def.ir_version = 9

# Save the onnx model
onnx.save(model_def, '../examples/onnx/tan.onnx')
print("Saved")