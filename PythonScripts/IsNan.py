import onnx 
from onnx import helper
from onnx import TensorProto

#Define the input and outputs
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2])
B = helper.make_tensor_value_info('B', TensorProto.BOOL, [2])

#Create the node
isnan_node = helper.make_node(
    'IsNaN',            #node
    inputs=['A'],       #inputs
    outputs=['B'],      #outputs
    name='IsNaNNode'    #name
)

# Create the graph
isnan_graph = helper.make_graph(
    [isnan_node],       #nodes
    'IsNaNGraph',       #graph name
    [A],                #inputs
    [B]                 #outputs
)

# Define the proper Opset Import
opset = onnx.OperatorSetIdProto()
opset.domain = ""
opset.version = 19

# Create the model
model = helper.make_model(
    isnan_graph,
    producer_name='onnx-isnan-example',
    opset_imports=[opset]
)

# Set the IR version to a supported value
model.ir_version = 9

# Save the onnx model
onnx.save(model, '../examples/onnx/isnan.onnx')
print("Saved")