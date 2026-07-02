import onnx
from onnx import helper
from onnx import TensorProto

# Inputs
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4, 4, 4])
scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT, [4])
bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [4])

# Output
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4, 4, 4])

# Node
groupnorm_node = helper.make_node(
    'GroupNormalization',
    inputs=['X', 'scale', 'bias'],
    outputs=['Y'],
    name='GroupNormalizationNode',
    epsilon=1e-5,
    num_groups=2,
)

# Graph
graph = helper.make_graph(
    [groupnorm_node],
    'GroupNormalizationGraph',
    [X, scale, bias],
    [Y]
)

# Opset
opset = onnx.OperatorSetIdProto()
opset.domain = ""
opset.version = 19

# Model
model = helper.make_model(
    graph,
    producer_name='onnx-groupnorm-example',
    opset_imports=[opset]
)

model.ir_version = 9

onnx.save(model, '../examples/onnx/group_normalization.onnx')
print("Saved")