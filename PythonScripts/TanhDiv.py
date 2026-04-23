import onnx
from onnx import helper
from onnx import TensorProto

#Define the input and outputs
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2])

#Create the node
tanh_node = helper.make_node(
    'Tanh',             #node
    inputs=['A'],       #inputs
    outputs=['B'],      #outputs
    name='TanhDivNode'  #name
)

# Create the graph
tanh_graph = helper.make_graph(
    [tanh_node],         #nodes
    'TanhDivGraph',     #graph name
    [A],                #inputs
    [B]                 #outputs
)

# Define the proper Opset Import
opset = onnx.OperatorSetIdProto()
opset.domain = ""
opset.version = 19

# Create the model
model = helper.make_model(
    tanh_graph,
    producer_name='onnx-tanh-div-example',
    opset_imports=[opset]
)

# Set the IR version to a supported value
model.ir_version = 9

# Save the onnx model
onnx.save(model, '../examples/onnx/tanh_div.onnx')
print("Saved")
