import onnx
from onnx import helper
from onnx import TensorProto

#Define the input and outputs
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2])
C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [2])

# Create the Pow node
pow_node = helper.make_node(
    'Pow',              #node
    inputs=['A', 'B'],  #inputs
    outputs=['C'],      #outputs
    name='PowNode'      #name
)

# Create the graph
pow_graph = helper.make_graph(
    [pow_node],         #nodes
    'PowGraph',         #graph name
    [A, B],             #inputs
    [C]                 #outputs
)

# Define the proper Opset Import
opset = onnx.OperatorSetIdProto()
opset.domain = ""
opset.version = 19

# Create the model
model = helper.make_model(
    pow_graph,
    producer_name='onnx-pow-example',
    opset_imports=[opset]
)

# Set the IR version to a supported value
model.ir_version = 9

# Save the onnx model
onnx.save(model, '../examples/onnx/Pow.onnx')
print("Saved")