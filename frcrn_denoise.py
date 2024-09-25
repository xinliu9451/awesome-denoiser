from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import librosa
import time
import os
from utils import remove_silence_from_both

pipeline = pipeline(
        Tasks.acoustic_noise_suppression,
        model='damo/speech_frcrn_ans_cirm_16k')

def frcrn_denoise(audio_input1, audio_input2, dB, min_silence_duration):
    start = time.time()
    output_path_1 = 'denoised1_output.wav'
    output_path_2 = 'denoised2_output.wav'
    pipeline(audio_input1, output_path=output_path_1)
    denoised1, sr1 = librosa.load(output_path_1, sr=None)
    os.remove(output_path_1)
    if audio_input2 is not None:
        pipeline(audio_input2, output_path=output_path_2)
        denoised2, sr2 = librosa.load(output_path_2, sr=None)
        denoised1 = remove_silence_from_both(denoised1, denoised2, sr1, threshold=dB, min_silence_duration=min_silence_duration)
        os.remove(output_path_2)
        return (sr1, denoised1), f"Time-Consuming: {round(time.time() - start, 4)}s"
    else:
        return (sr1, denoised1), f"Time-Consuming: {round(time.time() - start, 4)}s"
