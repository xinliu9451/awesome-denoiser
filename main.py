import gradio as gr

from denoiser_dns import denoiser_dns64, denoiser_dns48
from rnnoiser import rnnoiser_denoise
from noisereduce_denoise import noisereduce_denoise
from resemble_enhance_denoise import resemble_enhance_denoise
from frcrn_denoise import frcrn_denoise
from mossformer2_denoise import mossformer2_denoise
from deepfilternet_denoise import deepfilternet_denoise
from mdxnet_denoise import mdxnet_denoise
from mpsenet_denoise import mpsenet_denoise
from dtln_denoise import dtln_denoise



html_content = """
<div>
    <h2 style="font-size: 22px;margin-left: 0px;">Introduce</h2>
    <p style="font-size: 18px;margin-left: 20px;">这是一个音频降噪的Demo。</p>
    <h2 style="font-size: 22px;margin-left: 0px;">Configuration Introduction</h2>
    <p style="font-size: 18px;margin-left: 20px;">Capability：选择场景，这里可以选择语种识别降噪和语音翻译降噪，前者一个输入一个输出，后者两个输入一个输出。</p>
    <p style="font-size: 18px;margin-left: 20px;">Model：选择降噪模型，目前提供 11 种常用的降噪模型。</p>
    <p style="font-size: 18px;margin-left: 20px;">dB：检测静音的阈值，低于这个值会被认为是静音。一般dB值越大过滤掉的音频片段会越多，建议设置-40~-20之间。</p>
    <p style="font-size: 18px;margin-left: 20px;">Min_Silence_Duration：删除静音片段的最小间隔，单位是秒。建议设置为0.1，这样会删除连续静音超过0.1秒的音频片段。</p>
    <h2 style="font-size: 22px;margin-left: 0px;">Usage</h2>
    <p style="font-size: 18px;margin-left: 20px;">上传音频文件或通过麦克风输入，当选择"语音翻译降噪"，需要上传两个长度和采样率都一致的音频，否则只需要上传一个音频。</p>
</div>
"""
audio_examples = [
["example_data/前1m_3.wav"],
["example_data/前1m_4.wav"],
["example_data/TEST_MEETING_T0000000000.mp3"],
["example_data/TEST_MEETING_T0000000001.mp3"],
["example_data/TEST_MEETING_T0000000002.mp3"],
["example_data/TEST_MEETING_T0000000003.mp3"],
["example_data/TEST_MEETING_T0000000004.mp3"],
["example_data/TEST_MEETING_T0000000005.mp3"],
["example_data/TEST_MEETING_T0000000006.mp3"],
["example_data/TEST_MEETING_T0000000007.mp3"],
["example_data/TEST_MEETING_T0000000008.mp3"],
["example_data/TEST_MEETING_T0000000009.mp3"],
["example_data/TEST_MEETING_T0000000010.mp3"],
]

def model_inference(audio_input1, audio_input2, capability_inputs, model_inputs, dB, min_silence_duration):

    if model_inputs == "denoiser_dns64":
        return denoiser_dns64(audio_input1, audio_input2, dB, min_silence_duration)
    elif model_inputs == "denoiser_dns48":
        return denoiser_dns48(audio_input1, audio_input2, dB, min_silence_duration)
    elif model_inputs == "noisereduce":
        return noisereduce_denoise(audio_input1, audio_input2, dB, min_silence_duration)
    elif model_inputs == "resemble-enhance":
        return resemble_enhance_denoise(audio_input1, audio_input2, dB, min_silence_duration)
    elif model_inputs == "frcrn-ali":
        return frcrn_denoise(audio_input1, audio_input2, dB, min_silence_duration)
    elif model_inputs == "mossformer2-ali":
        return mossformer2_denoise(audio_input1, audio_input2, dB, min_silence_duration)
    elif model_inputs == "rnnoiser":
        return rnnoiser_denoise(audio_input1, audio_input2, dB, min_silence_duration)
    elif model_inputs == "deepfilternet":
        return deepfilternet_denoise(audio_input1, audio_input2, dB, min_silence_duration)
    elif model_inputs == "mdxnet":
        return mdxnet_denoise(audio_input1, audio_input2, dB, min_silence_duration)
    elif model_inputs == "mpsenet":
        return mpsenet_denoise(audio_input1, audio_input2, dB, min_silence_duration)
    elif model_inputs == "dtln":
        return dtln_denoise(audio_input1, audio_input2, dB, min_silence_duration)

def launch():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.HTML(html_content)
        with gr.Row():
            with gr.Column():
                with gr.Accordion("Configuration"):
                    capability_inputs = gr.Dropdown(choices=["language_recognition_denoiser", "speech_translation_denoiser"], value="language_recognition_denoiser", label="Capability")
                    model_inputs = gr.Dropdown(choices=["dtln", "mdxnet", "mpsenet", "denoiser_dns64", "denoiser_dns48", "noisereduce","resemble-enhance","frcrn-ali","mossformer2-ali","rnnoiser",'deepfilternet'], value="dtln", label="Model")
                    dB = gr.Slider(minimum=-80, maximum=0, step=1, label="dB", value=-20)
                    min_silence_duration = gr.Slider(minimum=0, maximum=1, step=0.01, label="Min_Silence_Duration", value=0.1)
                with gr.Row():
                    audio_input1 = gr.Audio(type="filepath", label="Primary Audio (Required)")
                    audio_input2 = gr.Audio(type="filepath", label="Secondary Audio (Optional)")

                fn_button = gr.Button("Start", variant="primary")
                audio_outputs = gr.Audio(label="Denoised Audio")
                text_outputs = gr.Textbox(label="Time-Consuming Inference")


            gr.Examples(examples=audio_examples, inputs=[audio_input1], examples_per_page=20)

        fn_button.click(model_inference, inputs=[audio_input1, audio_input2, capability_inputs, model_inputs, dB, min_silence_duration], outputs=[audio_outputs, text_outputs])

    demo.launch(share=True)


if __name__ == "__main__":
    launch()
