import onnx
from onnx import helper
from onnx import TensorProto

# Inputs
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 4, 4])
Scale = helper.make_tensor_value_info('Scale', TensorProto.FLOAT, [3])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [3])

# Output
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3, 4, 4])

# Node
instancenorm_node = helper.make_node(
    'InstanceNormalization',
    inputs=['X', 'Scale', 'B'],
    outputs=['Y'],
    name='InstanceNormalizationNode',
    epsilon=1e-5,
)

# Graph
graph = helper.make_graph(
    [instancenorm_node],
    'InstanceNormalizationGraph',
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
    producer_name='onnx-instancenorm-example',
    opset_imports=[opset]
)

model.ir_version = 9

onnx.save(model, '../examples/onnx/instance_normalization.onnx')
print("Saved")