import librosa
import noisereduce as nr


audio_input = "/path/to/your/audio.wav"
wav, sr = librosa.load(audio_input, sr=None)
denoised = nr.reduce_noise(y=wav, sr=sr)