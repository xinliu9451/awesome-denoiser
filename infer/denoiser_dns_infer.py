import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio

device = "cuda" if torch.cuda.is_available() else "cpu"
denoiser_dns64_model = pretrained.dns64().to(device)
denoiser_dns48_model = pretrained.dns48().to(device)

audio_input = "/path/to/your/audio.wav"

# use denoiser_dns64_model
wav, sr = torchaudio.load(audio_input)
wav = wav.to(device)
wav = convert_audio(wav, sr, denoiser_dns64_model.sample_rate, denoiser_dns64_model.chin)
denoised = denoiser_dns64_model(wav[None])[0]
   
# use denoiser_dns48_model
wav, sr = torchaudio.load(audio_input)
wav = wav.to(device)
wav = convert_audio(wav, sr, denoiser_dns48_model.sample_rate, denoiser_dns48_model.chin)
denoised = denoiser_dns48_model(wav[None])[0]
    