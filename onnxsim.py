import onnx
from onnxsim import simplify
'''link https://github.com/daquexian/onnx-simplifier'''

# convert model
onnx_model = r"your onnx model path"
model = onnx.load(onnx_model)
model_simp, check = simplify(model)
onnx.save(model_simp,"lite_hrnet_onxxsim.onnx")
assert check, "Simplified ONNX model could not be validated"
