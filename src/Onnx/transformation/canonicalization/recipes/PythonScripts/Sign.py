import onnx 
from onnx import helper
from onnx import TensorProto

# Define the input and outputs
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2])

# Create the Sign node
sign_node = helper.make_node(
    'Sign',            # node
    inputs=['A'],      # inputs
    outputs=['B'],     # outputs
    name='SignNode'    # name
)

# Create the graph
graph = helper.make_graph(
    [sign_node],       # nodes
    'SignGraph',       # graph name
    [A],               # inputs
    [B]                # outputs
)

# Define the proper Opset Import
opset = onnx.OperatorSetIdProto()
opset.domain = ""
opset.version = 19

# Create the model
model = helper.make_model(
    graph,
    producer_name='onnx-sign-example',
    opset_imports=[opset]
)

# Set the IR version to a supported value
model.ir_version = 9

# Save the onnx model
onnx.save(model, 'sign.onnx')
print("Saved")