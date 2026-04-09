from math import fmod

import onnx
from onnx import helper
from onnx import TensorProto

def create_mod_model(fmod_value):
    A = helper.make_tensor_value_info('A', TensorProto.INT64 if fmod_value == 0 else TensorProto.FLOAT, [6])
    B = helper.make_tensor_value_info('B', TensorProto.INT64 if fmod_value == 0 else TensorProto.FLOAT, [6])
    C = helper.make_tensor_value_info('C', TensorProto.INT64 if fmod_value == 0 else TensorProto.FLOAT, [6])

    mod_node = helper.make_node(
        'Mod',              
        inputs=['A', 'B'],  
        outputs=['C'],      
        name=f'ModNode_fmod{fmod_value}', 
        fmod=fmod_value     
    )

    mod_graph = helper.make_graph(
        [mod_node],         
        f'ModGraph_fmod{fmod_value}',
        [A, B],             
        [C]                 
    )

    opset = onnx.OperatorSetIdProto()
    opset.domain = ""
    opset.version = 19

    model = helper.make_model(
        mod_graph,
        producer_name=f'onnx-mod-fmod{fmod_value}',
        opset_imports=[opset]
    )
    
    model.ir_version = 9

    onnx.save(model, f'../examples/onnx/mod_fmod{fmod_value}.onnx')
    print("Saved")

if __name__ == "__main__":
    create_mod_model(0)
    create_mod_model(1)