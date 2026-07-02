import onnx
from onnx import helper
from onnx import TensorProto

# Inputs
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 4, 4])
scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT, [3])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [3])
mean = helper.make_tensor_value_info('mean', TensorProto.FLOAT, [3])
var = helper.make_tensor_value_info('var', TensorProto.FLOAT, [3])

# Output
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3, 4, 4])

# BatchNormalization node
bn_node = helper.make_node(
    'BatchNormalization',
    inputs=['X', 'scale', 'B', 'mean', 'var'],
    outputs=['Y'],
    name='BatchNormalizationNode',
    epsilon=1e-5,
)

# Graph
graph = helper.make_graph(
    [bn_node],
    'BatchNormalizationGraph',
    [X, scale, B, mean, var],
    [Y]
)

# Opset
opset = onnx.OperatorSetIdProto()
opset.domain = ""
opset.version = 19

# Model
model = helper.make_model(
    graph,
    producer_name='onnx-batchnorm-example',
    opset_imports=[opset]
)

model.ir_version = 9

# Save
onnx.save(model, '../examples/onnx/batch_normalization.onnx')
print("Saved")