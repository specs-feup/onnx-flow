import onnx
from onnx import helper
from onnx import TensorProto

#Create the input and output tensors
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2])

#Create the node
sqrt_node = helper.make_node(
    'Sqrt',                 #node
    inputs=['A'],           #inputs
    outputs=['B'],          #outputs
    name='SqrtNode'         #name
)

# Create the graph
sqrt_graph = helper.make_graph(
    [sqrt_node],            #nodes
    'SqrtGraph',            #graph name
    [A],                    #inputs
    [B]                     #outputs
)

# Define the proper Opset Import
opset = onnx.OperatorSetIdProto()
opset.domain = ""
opset.version = 19

# Create the model
model = helper.make_model(
    sqrt_graph,
    producer_name='onnx-sqrt-example',
    opset_imports=[opset]
)

# Set the IR version to a supported value
model.ir_version = 9

# Save the onnx model
onnx.save(model, '../examples/onnx/sqrt.onnx')
print("Saved")
