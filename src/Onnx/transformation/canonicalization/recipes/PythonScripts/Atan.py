import onnx
from onnx import helper
from onnx import TensorProto

#Create the input and output tensors
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2])

#Create the node
atan_node = helper.make_node(
    'Atan',                 #node
    inputs=['A'],           #inputs
    outputs=['B'],          #outputs
    name='AtanNode'         #name
)

# Create the graph
atan_graph = helper.make_graph(
    [atan_node],      #nodes
    'AtanGraph',            #graph name
    [A],                    #inputs
    [B]                     #outputs
)

# Define the proper Opset Import
opset = onnx.OperatorSetIdProto()
opset.domain = ""
opset.version = 19

# Create the model
model = helper.make_model(
    atan_graph,
    producer_name='onnx-atan-example',
    opset_imports=[opset]
)

# Set the IR version to a supported value
model.ir_version = 9

# Save the onnx model
onnx.save(model, '../examples/onnx/atan.onnx')
print("Saved")

