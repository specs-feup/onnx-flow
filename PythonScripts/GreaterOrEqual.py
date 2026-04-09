import onnx
from onnx import helper
from onnx import TensorProto    

# 1. Define the inputs and outputs
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2])
C = helper.make_tensor_value_info('C', TensorProto.BOOL , [2])

# 2. Create the GreaterOrEqual node
great_or_equal_node = helper.make_node(
    'GreaterOrEqual',  # op_type
    ['A', 'B'],        # inputs
    ['C'],             # outputs
    name='GreaterOrEqualNode'  # node name
)

# 3. Create the graph
graph = helper.make_graph(
    [great_or_equal_node],  #nodes
    'GreaterOrEqualGraph',  # graph name
    [A, B],                 #inputs
    [C]                     #outputs  
)

# 4. Define the proper Opset Import 
opset = onnx.OperatorSetIdProto()
opset.domain = ""
opset.version = 19

# 5. Create the model
model = helper.make_model(
    graph,
    producer_name='onnx-greater-or-equal-example',
    opset_imports=[opset]
)

# 6. Set the IR version to a supported value
model.ir_version = 9

# 7. Save the model to a binary .onnx file
onnx.save(model, 'greater_or_equal.onnx')
print("Saved to 'greater_or_equal.onnx'")