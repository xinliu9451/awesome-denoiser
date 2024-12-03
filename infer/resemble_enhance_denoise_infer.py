import torch
import torchaudio
from resemble_enhance.enhancer.inference import denoise


audio_input = "/path/to/your/audio.wav"
data, sr = torchaudio.load(audio_input)
data = data.mean(0)
denoised, sr = denoise(dwav=data, sr=sr, device="cuda" if torch.cuda.is_available() else "cpu", run_dir=None)
denoised = denoised.cpu().numpy()

