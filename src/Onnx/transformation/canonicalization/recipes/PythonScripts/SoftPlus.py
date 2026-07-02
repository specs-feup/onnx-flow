import onnx
from onnx import helper
from onnx import TensorProto

#Define the inputs and outputs
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2])

#Create the Softplus node
softplus_node = helper.make_node(
    'Softplus',             # node
    inputs=['A'],           # inputs
    outputs=['B'],          # outputs
    name='SoftplusNode',    # name
)

#Create the graph
graph = helper.make_graph(
    [softplus_node],        # nodes
    'SoftplusGraph',        # graph name
    [A],                    # inputs
    [B]                     # outputs
)

#Define the proper Opset Import
opset = onnx.OperatorSetIdProto()
opset.domain = ""
opset.version = 19

#Create the model
model = helper.make_model(
    graph,
    producer_name='onnx-soft-plus-example',
    opset_imports=[opset]
)

#Set the IR version to a supported value
model.ir_version = 9

#Save the onnx model
onnx.save(model, 'onnx/softplus.onnx')
print("Saved")