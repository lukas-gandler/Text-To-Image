# Text-To-Image
The goal of this project is to get a better understanding of how text-to-image generation such as in 
[DALL·E 2](https://openai.com/de-DE/index/dall-e-2/) or similar models work. This work was also inspired by the 
[Guest video of WelchLabsVideo on the channel of 3Blue1Brown](https://www.youtube.com/watch?v=iv-5mZ_9CPY). This project 
is working with the MNIST dataset in order to keep the dataset simple and not overcomplicate things. 

[demo animation](res/animation.gif)

## Project Structure
<pre>
.
├── dataloading/
│   ├── character_tokenizer.py    # Character-Level tokenization
│   └── mnist_dataset.py          # MNIST dataset
├── models/
│   ├── clip.py                   # CLIP model and loss function
│   ├── diffusion_model.py        # DDPM model and U-Net architecture
│   ├── image_encoder.py          # Simple CNN image encoder
│   └── text_encoder.py           # Character-Level LSTM text encoder
├── utils/
│   ├── plotting.py              
│   └── utils.py
├── clip_train.py                 # Training of CLIP model
├── unclip_train.py               # Training of DDPM and conditioning on learned CLIP text embeddings 
├── demo.ipynb                    # Demo notebook
└── requirements.txt              # Dependencies 


</pre>

## Models
For this re-implementation, the image encoder of CLIP is a simple CNN network while the TextEncoder is a stacked LSTM, 
working with individual characters as tokens. The Diffusion model uses a U-Net as a backbone which is conditioned on the 
time-step and the clip-embedding of the text during its training procedure.

## Demo
The accompanying note ```demo.ipynb``` demonstrates how to use this codebase and explores some insights from the CLIP 
training and the image generation process.

## Usage
To train the CLIP model, run the following command and set the ```save_path``` as your desired save path for the clip model.
```shell
python3 clip_train.py --save_path mnist_clip.pth
```

To train the text-to-image diffusion model using the clip learned clip embeddings run the following command and set ```clip_model_path``` to the path of the trained CLIP model and ```save_path``` as the desired save path for the diffusion model.
```shell
python3 unclip_train.py --clip_model_path mnist_clip.pth --save_path mnist_ddpm.pth
```

You can either train the models yourself (this should not take too long as the mnist dataset is rather small) or you can download the trained models in the release section.

Once the models are trained, they can be used to generate new digit images based on use input. To do so, run the following 
command and set ```prompt``` to the digit you want to generate and ```num_samples``` to the number of digits you want to create.
```shell
python3 generate_images.py --prompt six --num_samples 64
```

## References
[1] [Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021, July). Learning transferable visual models from natural language supervision. In International conference on machine learning (pp. 8748-8763). PmLR.](https://arxiv.org/pdf/2103.00020)

[2] [Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., & Chen, M. (2022). Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2), 3.](https://arxiv.org/abs/2204.06125)

[3] [Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. Advances in neural information processing systems, 33, 6840-6851.](https://arxiv.org/abs/2006.11239)
