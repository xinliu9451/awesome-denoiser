import librosa
from MPSENet import MPSENet

model = 'JacobLinCool/MP-SENet-DNS'  # JacobLinCool/MP-SENet-DNS   JacobLinCool/MP-SENet-VB
device = "cuda" if torch.cuda.is_available() else "cpu"

model = MPSENet.from_pretrained(model).to(device)

audio_input = "/path/to/your/audio.wav"
wav, sr = librosa.load(audio_input, sr=model.sampling_rate)
denoised, sr, notation = model(wav)


