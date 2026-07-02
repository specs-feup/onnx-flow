import onnx
from onnx import helper
from onnx import TensorProto

# Inputs
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 3, 4])
Scale = helper.make_tensor_value_info('Scale', TensorProto.FLOAT, [4])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [4])

# Output
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 3, 4])

# Node
layernorm_node = helper.make_node(
    'LayerNormalization',
    inputs=['X', 'Scale', 'B'],
    outputs=['Y'],
    name='LayerNormalizationNode',
    axis=2,
    epsilon=1e-5,
)

# Graph
graph = helper.make_graph(
    [layernorm_node],
    'LayerNormalizationGraph',
    [X, Scale, B],
    [Y]
)

# Opset
opset = onnx.OperatorSetIdProto()
opset.domain = ""
opset.version = 19

# Model
model = helper.make_model(
    graph,
    producer_name='onnx-layernorm-example',
    opset_imports=[opset]
)

model.ir_version = 9

onnx.save(model, '../examples/onnx/layer_normalization.onnx')
print("Saved")