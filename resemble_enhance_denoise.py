import torch
import time
import torchaudio

from resemble_enhance.enhancer.inference import denoise, enhance
from utils import remove_silence_from_both



def resemble_enhance_denoise(audio_input1, audio_input2, dB, min_silence_duration):
    start = time.time()
    data1, sr1 = torchaudio.load(audio_input1)
    data1 = data1.mean(0)
    denoised1, sr1 = denoise(dwav=data1, sr=sr1, device="cuda" if torch.cuda.is_available() else "cpu", run_dir=None)
    denoised1 = denoised1.cpu().numpy()
    if audio_input2 is not None:
        data2, sr2 = torchaudio.load(audio_input2)
        data2 = data2.mean(0)
        denoised2, sr2 = denoise(dwav=data2, sr=sr2, device="cuda" if torch.cuda.is_available() else "cpu", run_dir=None)
        denoised2 = denoised2.cpu().numpy()
        denoised1 = remove_silence_from_both(denoised1, denoised2, sr1, threshold=dB, min_silence_duration=min_silence_duration)
        return (sr1, denoised1), f"Time-Consuming: {round(time.time() - start, 4)}s"
    else:
        return (sr1, denoised1), f"Time-Consuming: {round(time.time() - start, 4)}s"



