import onnxruntime
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm
from librosa import istft


file = '../example_data/gtcrn.onnx'
session = onnxruntime.InferenceSession(file, None, providers=['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider'])

def infer(audio_path):
    outputs = []
    mix_data, sr = sf.read(audio_path, dtype='float32')
    x = torch.from_numpy(mix_data)
    x = torch.stft(x, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)[None]
    inputs = x.numpy()
    conv_cache = np.zeros([2, 1, 16, 16, 33],  dtype="float32")
    tra_cache = np.zeros([2, 3, 1, 1, 16],  dtype="float32")
    inter_cache = np.zeros([2, 1, 33, 16],  dtype="float32")
    for i in tqdm(range(inputs.shape[-2])):
        out_i,  conv_cache, tra_cache, inter_cache \
                = session.run([], {'mix': inputs[..., i:i+1, :],
                    'conv_cache': conv_cache,
                    'tra_cache': tra_cache,
                    'inter_cache': inter_cache})

        outputs.append(out_i)

    outputs = np.concatenate(outputs, axis=2)
    enhanced = istft(outputs[...,0] + 1j * outputs[...,1], n_fft=512, hop_length=256, win_length=512, window=np.hanning(512)**0.5)
    enhanced =  enhanced.squeeze()
    
    return enhanced, sr
    
    
audio_input = "/path/to/your/audio.wav"
denoised, sr = infer(audio_input)
   


