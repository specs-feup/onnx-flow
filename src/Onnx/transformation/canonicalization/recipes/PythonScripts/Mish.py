import onnx 
from onnx import helper
from onnx import TensorProto

#Define the inputs and outputs
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2])

#Create the Mish node
mish_node = helper.make_node(
    'Mish',                 # node
    inputs=['A'],           # inputs
    outputs=['B'],          # outputs
    name='MishNode',        # name
)

#Create the graph
graph = helper.make_graph(
    [mish_node],            # nodes
    'MishGraph',            # graph name
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
    producer_name='onnx-mish-example',
    opset_imports=[opset]
)

#Set the IR version to a supported value
model.ir_version = 9

#Save the onnx model
onnx.save(model, 'onnx/mish.onnx')
print("Saved")