import onnx
import onnxruntime as ort
import numpy as np
import librosa
import time
from utils import remove_silence_from_both


onnx_model_path = 'example_data/mossformer2_model.onnx'
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
ort_session = ort.InferenceSession(onnx_model_path)


def mossformer2_denoise(audio_input1, audio_input2, dB, min_silence_duration):
    start = time.time()
    input_data1, sr1 = librosa.load(audio_input1, sr=None)
    input_data1 = np.expand_dims(input_data1, axis=0).astype(np.float32)
    input_name = ort_session.get_inputs()[0].name
    outputs1 = ort_session.run(None, {input_name: input_data1})
    denoised1 = outputs1[0][0, :, 0]
    if audio_input2 is not None:
        input_data2, sr2 = librosa.load(audio_input2, sr=None)
        input_data2 = np.expand_dims(input_data2, axis=0).astype(np.float32)
        outputs2 = ort_session.run(None, {input_name: input_data2})
        denoised2 = outputs2[0][0, :, 0]
        denoised1 = remove_silence_from_both(denoised1, denoised2, sr1, threshold=dB, min_silence_duration=min_silence_duration)
        return (sr1, denoised1), f"Time-Consuming: {round(time.time() - start, 4)}s"
    else:
        return (sr1, denoised1), f"Time-Consuming: {round(time.time() - start, 4)}s"




