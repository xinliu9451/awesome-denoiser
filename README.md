# awesome-denoiser
This is a repository that collects common audio noise reduction models, using Gradio to demonstrate the use of each model, which is very friendly for beginners.

## Recommend
From my own experience with it, the mdxnet model works best, the dtln model balances effectiveness and efficiency, and the gtcrn works well. 

By the way, this repository will continue to update and add some new speech enhancement models, please give me a STAR if you find it helpful.

## Usage
This is a project for demonstrating the effect of multiple noise reduction models. If you just want to use a particular model for noise reduction, you can use the scripts in the infer directory.

You need to do is configure the environment and download the required model. You can use the following command to set up the environment and download the Mossformer2, DeepFilterNet2 model, then place the downloaded model in the example_data directory.
```
pip install -r requirements
wget https://www.modelscope.cn/models/dengcunqin/speech_mossformer2_noise_reduction_16k/resolve/master/simple_model.onnx
wget https://github.com/yuyun2000/SpeechDenoiser/blob/main/48k/denoiser_model.onnx
wget https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_1.onnx
wget https://github.com/breizhn/DTLN/raw/refs/heads/master/pretrained_model/model_1.tflite
wget https://github.com/breizhn/DTLN/raw/refs/heads/master/pretrained_model/model_2.tflite
wget https://github.com/Xiaobin-Rong/gtcrn/blob/main/stream/onnx_models/gtcrn.onnx
```

In my project, there are two modes of audio denoising. One mode takes two audio inputs and produces one audio output. The principle is to detect silent sections based on one audio and then remove the corresponding silent sections from the other audio. This is a specific application scenario, so you don't need to consider it. You just need to focus on the scenario where one audio input results in one audio output, as this is the more generalized application scenario.

## Gradio-generated interface display：
![image](https://github.com/xinliu9451/awesome-denoiser/blob/main/example_data/demo.png)

## Reference
1. https://github.com/timsainb/noisereduce
2. https://github.com/facebookresearch/denoiser
3. https://github.com/resemble-ai/resemble-enhance
4. https://github.com/dbklim/RNNoise_Wrapper
5. https://www.modelscope.cn/models/iic/speech_frcrn_ans_cirm_16k
6. https://www.modelscope.cn/models/dengcunqin/speech_mossformer2_noise_reduction_16k
7. https://github.com/yuyun2000/SpeechDenoiser
8. https://github.com/breizhn/DTLN
9. https://github.com/JacobLinCool/MPSENet
10. https://github.com/seanghay/uvr-mdx-infer
11. https://github.com/Xiaobin-Rong/gtcrn
