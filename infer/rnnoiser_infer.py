import os
import numpy as np
from rnnoise_wrapper import RNNoise
import torchaudio
import soundfile as sf
import librosa


denoiser = RNNoise()

def read_mp3(file_path):
    # 加载 MP3 文件
    audio, sr = librosa.load(file_path, sr=None)
    wav_path = file_path.replace('mp3', 'wav')
    # 保存为临时 WAV 文件
    sf.write(wav_path, audio, sr)
    wav = denoiser.read_wav(wav_path)
    os.remove(wav_path)
    return wav, sr

def read_audio(file_path):
    if file_path.split('/')[-1].split('.')[-1] == 'wav':
        return denoiser.read_wav(file_path), torchaudio.info(file_path).sample_rate
    else:
        return read_mp3(file_path)

audio_input = "/path/to/your/audio.wav"
wav, sr = read_audio(audio_input)
denoised = denoiser.filter(wav)
denoised = np.array(denoised.get_array_of_samples())
