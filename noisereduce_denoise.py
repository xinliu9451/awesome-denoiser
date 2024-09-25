import librosa
import time
from utils import remove_silence_from_both
import noisereduce as nr


def noisereduce_denoise(audio_input1, audio_input2, dB, min_silence_duration):
    start = time.time()
    data1, sr1 = librosa.load(audio_input1, sr=None)
    denoised1 = nr.reduce_noise(y=data1, sr=sr1)
    if audio_input2 is not None:
        data2, sr2 = librosa.load(audio_input2, sr=None)
        denoised2 = nr.reduce_noise(y=data2, sr=sr2)
        denoised1 = remove_silence_from_both(denoised1, denoised2, sr1, threshold=dB, min_silence_duration=min_silence_duration)
        return (sr1, denoised1), f"Time-Consuming: {round(time.time() - start, 4)}s"
    else:
        return (sr1, denoised1), f"Time-Consuming: {round(time.time() - start, 4)}s"