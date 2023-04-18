import onnx
import onnxruntime as ort
# Load target model; ONNX format
T = onnx.load("/home/akasaka/nas/models/arcface-resnet100_MS1MV3.onnx")
# Check that the model is well formed
onnx.checker.check_model(T)
# Print a human readable representation of the graph
print(onnx.helper.printable_graph(T.graph))
T_session = ort.InferenceSession("/home/akasaka/nas/models/arcface-resnet100_MS1MV3.onnx")
for input in T_session.get_inputs():
    print(input.name)
    print(input.shape)

for output in T_session.get_outputs():
    print(output.name)
    print(output.shape)
