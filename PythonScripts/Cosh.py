import onnx
from onnx import helper
from onnx import TensorProto

#Define the input and outputs
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2])

#Create the node
node_def = helper.make_node(
    'Cosh',             #node
    inputs=['A'],       #inputs
    outputs=['B'],      #outputs
    name='CoshNode'     #name
)

# Create the graph
graph_def = helper.make_graph(
    [node_def],         #nodes
    'CoshGraph',        #graph name
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
    producer_name='onnx-cosh-example',
    opset_imports=[opset]
)

# Set the IR version to a supported value
model_def.ir_version = 9

# Save the onnx model
onnx.save(model_def, '../examples/onnx/cosh.onnx')
print("Saved")