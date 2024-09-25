import os
import numpy as np
from rnnoise_wrapper import RNNoise
import torchaudio
import soundfile as sf
import librosa
from utils import remove_silence_from_both
import time

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

def rnnoiser_denoise(audio_input1, audio_input2, dB, min_silence_duration):
    start = time.time()
    wav1, sr1 = read_audio(audio_input1)
    denoised1 = denoiser.filter(wav1)
    denoised1 = np.array(denoised1.get_array_of_samples())

    if audio_input2 is not None:
        wav2, sr2 = read_audio(audio_input2)
        denoised2 = denoiser.filter(wav2)
        denoised2 = np.array(denoised2.get_array_of_samples())
        denoised1 = remove_silence_from_both(denoised1, denoised2, sr1, threshold=dB, min_silence_duration=min_silence_duration)
        return (sr1, denoised1), f"Time-Consuming: {round(time.time() - start, 4)}s"
    else:
        return (sr1, denoised1), f"Time-Consuming: {round(time.time() - start, 4)}s"