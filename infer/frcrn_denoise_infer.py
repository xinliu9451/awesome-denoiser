from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# inference:https://www.modelscope.cn/models/iic/speech_frcrn_ans_cirm_16k

pipeline = pipeline(
        Tasks.acoustic_noise_suppression,
        model='damo/speech_frcrn_ans_cirm_16k')
        
audio_input = "/path/to/your/audio.wav"
pipeline(audio_input, output_path=output_path)

