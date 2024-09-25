# awesome-denoiser
This is a repository that collects common audio noise reduction models, using Gradio to demonstrate the use of each model, which is very friendly for beginners.

All you need to do is configure the environment and download the required model. You can use the following command to set up the environment and download the Mossformer2 model, then place the downloaded model in the example_data directory.
```
pip install -r requirements
wget https://www.modelscope.cn/models/dengcunqin/speech_mossformer2_noise_reduction_16k/resolve/master/simple_model.onnx
```

Gradio-generated interface display：
![image](https://github.com/xinliu9451/awesome-denoiser/blob/main/example_data/demo.png)
