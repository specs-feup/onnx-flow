import onnx 
from onnx import helper
from onnx import TensorProto

#Define the input and outputs
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2])
B = helper.make_tensor_value_info('B', TensorProto.BOOL, [2])

#Create the node
isinf_node = helper.make_node(
    'IsInf',            #node
    inputs=['A'],       #inputs
    outputs=['B'],      #outputs
    name='IsInfNode'    #name
)

# Create the graph
isinf_graph = helper.make_graph(
    [isinf_node],       #nodes
    'IsInfGraph',       #graph name
    [A],                #inputs
    [B]                 #outputs
)

# Define the proper Opset Import
opset = onnx.OperatorSetIdProto()
opset.domain = ""
opset.version = 19

# Create the model
model = helper.make_model(
    isinf_graph,
    producer_name='onnx-isinf-example',
    opset_imports=[opset]
)

# Set the IR version to a supported value
model.ir_version = 9

# Save the onnx model
onnx.save(model, '../examples/onnx/isinf.onnx')
print("Saved")