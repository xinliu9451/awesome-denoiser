import onnx
import onnxruntime as ort
import numpy as np
import librosa
import torch

onnx_model_path = '../example_data/mossformer2_model.onnx'
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
ort_session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider'])

audio_input = "/path/to/your/audio.wav"
input_data, sr = librosa.load(audio_input, sr=None)
input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
input_name = ort_session.get_inputs()[0].name
outputs = ort_session.run(None, {input_name: input_data})
denoised = outputs[0][0, :, 0]



