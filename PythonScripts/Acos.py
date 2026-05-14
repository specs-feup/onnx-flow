import onnx
from onnx import helper
from onnx import TensorProto

#Create the input and output tensors
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2])

#Create the node
acos_node = helper.make_node(
    'Acos',                 #node
    inputs=['A'],           #inputs
    outputs=['B'],          #outputs
    name='AcosNode'         #name
)

# Create the graph
acos_graph = helper.make_graph(
    [acos_node],      #nodes
    'AcosGraph',            #graph name
    [A],                    #inputs
    [B]                     #outputs
)

# Define the proper Opset Import
opset = onnx.OperatorSetIdProto()
opset.domain = ""
opset.version = 19

# Create the model
model = helper.make_model(
    acos_graph,
    producer_name='onnx-acos-example',
    opset_imports=[opset]
)

# Set the IR version to a supported value
model.ir_version = 9

# Save the onnx model
onnx.save(model, '../examples/onnx/acos.onnx')
print("Saved")

