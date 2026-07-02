import onnx 
from onnx import helper
from onnx import TensorProto

# 1. Define the inputs and outputs
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2])
C = helper.make_tensor_value_info('C', TensorProto.BOOL, [2])

# 2. Create the Less node
less_or_equal_node = helper.make_node(
    'LessOrEqual',          # op_type
    ['A', 'B'],             # inputs
    ['C'],                  # outputs
    name='LessOrEqualNode'  # node name
)

# 3. Create the graph
graph = helper.make_graph(
    [less_or_equal_node],  # nodes
    'LessOrEqualGraph',    # graph name
    [A, B],                # inputs
    [C]                    # outputs
)

# 4. Define the proper Opset Import
opset = onnx.OperatorSetIdProto()
opset.domain = ""
opset.version = 19

# 5. Create the model
model = helper.make_model(
    graph, 
    producer_name='onnx-less-or-equal-example',
    opset_imports=[opset]
)

# 6. Set the IR version to a supported value
model.ir_version = 9

# Save the model
onnx.save(model, 'less_or_equal.onnx')
print("Saved")