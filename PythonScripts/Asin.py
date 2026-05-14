import onnx
from onnx import helper
from onnx import TensorProto

#Create the input and output tensors
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2])

#Create the node
asin_node = helper.make_node(
    'Asin',                 #node
    inputs=['A'],           #inputs
    outputs=['B'],          #outputs
    name='AsinNode'         #name
)

# Create the graph
asin_graph = helper.make_graph(
    [asin_node],      #nodes
    'AsinGraph',            #graph name
    [A],                    #inputs
    [B]                     #outputs
)

# Define the proper Opset Import
opset = onnx.OperatorSetIdProto()
opset.domain = ""
opset.version = 19

# Create the model
model = helper.make_model(
    asin_graph,
    producer_name='onnx-asin-example',
    opset_imports=[opset]
)

# Set the IR version to a supported value
model.ir_version = 9

# Save the onnx model
onnx.save(model, '../examples/onnx/asin.onnx')
print("Saved")

