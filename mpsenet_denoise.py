import librosa
import time
from MPSENet import MPSENet

from utils import remove_silence_from_both

model = 'JacobLinCool/MP-SENet-DNS'  # JacobLinCool/MP-SENet-DNS   JacobLinCool/MP-SENet-VB
device = "cuda"

model = MPSENet.from_pretrained(model).to(device)


def mpsenet_denoise(audio_input1, audio_input2, dB, min_silence_duration):
    start = time.time()
    wav1, sr1 = librosa.load(audio_input1, sr=model.sampling_rate)
    denoised1, sr1, notation1 = model(wav1)
    if audio_input2 is not None:
        wav2, sr2 = librosa.load(audio_input2, sr=model.sampling_rate)
        denoised2, sr2, notation2 = model(wav2)
        denoised1 = remove_silence_from_both(denoised1, denoised2, sr1, threshold=dB, min_silence_duration=min_silence_duration)
        return (sr2, denoised1), f"Time-Consuming: {round(time.time() - start, 4)}s"
    else:
        return (sr1, denoised1), f"Time-Consuming: {round(time.time() - start, 4)}s"



