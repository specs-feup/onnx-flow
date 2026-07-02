import onnx
from onnx import helper
from onnx import TensorProto

#Define the input/output tensors
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2])
C = helper.make_tensor_value_info('C', TensorProto.BOOL, [2])

#Create the Less node
less_node = helper.make_node(
    'Less',               # op_type
    inputs=['A', 'B'],    # inputs
    outputs=['C'],        # outputs
    name='LessNode'       # node name
)

#Create the graph
graph=helper.make_graph(
    [less_node],    # nodes
    'LessGraph',    # graph name
    [A, B],         # inputs
    [C]             # outputs    
)

#Define the proper Opset Import
opset = onnx.OperatorSetIdProto()
opset.domain = ""
opset.version = 19

#Create the model
model = helper.make_model(
    graph, 
    producer_name='onnx-less-example',
    opset_imports=[opset]
)

#Set the IR version to a supported value
model.ir_version = 9

#Save the onnx model
onnx.save(model, 'less.onnx')
print("Save")
