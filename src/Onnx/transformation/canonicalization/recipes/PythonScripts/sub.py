import onnx
from onnx import helper
from onnx import TensorProto

# 1. Define the inputs and outputs
# elem_type 1 in the JSON corresponds to TensorProto.FLOAT in ONNX
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2])
C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [2])

# 2. Create the Sub node
# We give it the exact same name and topology
sub_node = helper.make_node(
    'Sub',          # op_type
    ['A', 'B'],     # inputs
    ['C'],          # outputs
    name='SubNode'  # node name
)

# 3. Create the graph
graph = helper.make_graph(
    [sub_node],     # nodes
    'SubGraph',     # graph name
    [A, B],         # inputs
    [C]             # outputs
)

# 4. Define the proper Opset Import (fixes the opset 25 issue)
opset = onnx.OperatorSetIdProto()
opset.domain = ""
opset.version = 19

# 5. Create the model
model = helper.make_model(
    graph, 
    producer_name='onnx-sub-example',
    opset_imports=[opset]
)

# 6. Set the IR version to a supported value (fixes the IR 13 issue)
model.ir_version = 9

# Optional: Run the ONNX checker to verify the model is structurally sound
onnx.checker.check_model(model)
print("Model checked successfully!")

# 7. Save the model to a binary .onnx file
onnx.save(model, 'examples/onnx/sub.onnx')
print("Saved to 'examples/onnx/sub.onnx'")