import torch
import torchaudio
from utils import remove_silence_from_both
from denoiser import pretrained
from denoiser.dsp import convert_audio
import time


denoiser_dns64_model = pretrained.dns64().cuda()
denoiser_dns48_model = pretrained.dns48().cuda()


def denoiser_dns64(audio_input1, audio_input2, dB, min_silence_duration):
    start = time.time()
    wav1, sr1 = torchaudio.load(audio_input1)
    wav1 = wav1.to('cuda')
    wav1 = convert_audio(wav1, sr1, denoiser_dns64_model.sample_rate, denoiser_dns64_model.chin)
    with torch.no_grad():
        denoised1 = denoiser_dns64_model(wav1[None])[0]
    if audio_input2 is not None:
        wav2, sr2 = torchaudio.load(audio_input2)
        wav2 = wav2.to('cuda')
        wav2 = convert_audio(wav2, sr2, denoiser_dns64_model.sample_rate, denoiser_dns64_model.chin)
        with torch.no_grad():
            denoised2 = denoiser_dns64_model(wav2[None])[0]
        denoised1 = denoised1.squeeze(0).cpu().numpy()
        denoised2 = denoised2.squeeze(0).cpu().numpy()
        denoised1 = remove_silence_from_both(denoised1, denoised2, sr1, threshold=dB, min_silence_duration=min_silence_duration)
        return (denoiser_dns64_model.sample_rate, denoised1), f"Time-Consuming: {round(time.time() - start, 4)}s"
    else:
        denoised1 = denoised1.squeeze(0).cpu().numpy()
        return (denoiser_dns64_model.sample_rate, denoised1), f"Time-Consuming: {round(time.time() - start, 4)}s"

def denoiser_dns48(audio_input1, audio_input2, dB, min_silence_duration):
    start = time.time()
    wav1, sr1 = torchaudio.load(audio_input1)
    wav1 = wav1.to('cuda')
    wav1 = convert_audio(wav1, sr1, denoiser_dns48_model.sample_rate, denoiser_dns48_model.chin)
    with torch.no_grad():
        denoised1 = denoiser_dns48_model(wav1[None])[0]
    if audio_input2 is not None:
        wav2, sr2 = torchaudio.load(audio_input2)
        wav2 = wav2.to('cuda')
        wav2 = convert_audio(wav2, sr2, denoiser_dns48_model.sample_rate, denoiser_dns48_model.chin)
        with torch.no_grad():
            denoised2 = denoiser_dns48_model(wav2[None])[0]
        denoised1 = denoised1.squeeze(0).cpu().numpy()
        denoised2 = denoised2.squeeze(0).cpu().numpy()
        denoised1 = remove_silence_from_both(denoised1, denoised2, sr1, threshold=dB, min_silence_duration=min_silence_duration)
        return (denoiser_dns48_model.sample_rate, denoised1), f"Time-Consuming: {round(time.time() - start, 4)}s"
    else:
        denoised1 = denoised1.squeeze(0).cpu().numpy()
        return (denoiser_dns48_model.sample_rate, denoised1), f"Time-Consuming: {round(time.time() - start, 4)}s"