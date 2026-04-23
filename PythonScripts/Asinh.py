import onnx 
from onnx import helper
from onnx import TensorProto

#Define the input and outputs
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2])

#Create the node
asinh_node = helper.make_node(
    'Asinh',            #node
    inputs=['A'],       #inputs
    outputs=['B'],      #outputs
    name='AsinhNode'    #name
)

# Create the graph
asinh_graph = helper.make_graph(
    [asinh_node],       #nodes
    'AsinhGraph',       #graph name
    [A],                #inputs
    [B]                 #outputs
)

# Define the proper Opset Import
opset = onnx.OperatorSetIdProto()
opset.domain = ""
opset.version = 19

# Create the model
model = helper.make_model(
    asinh_graph,
    producer_name='onnx-asinh-example',
    opset_imports=[opset]
)

# Set the IR version to a supported value
model.ir_version = 9

# Save the onnx model
onnx.save(model, '../examples/onnx/asinh.onnx')
print("Saved")