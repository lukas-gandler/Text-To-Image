# Text-To-Image
The goal of this project is to get a better understanding of how text-to-image generation such as in [DALL·E 2](https://openai.com/de-DE/index/dall-e-2/) or similar models work. This work was also inspired by the [Guest video of WelchLabsVideo on the channel of 3Blue1Brown](https://www.youtube.com/watch?v=iv-5mZ_9CPY).

# Usage

MNIST settings:
CLIP: 
* description_encoder.target_sequence_length=10
* batch_size=128
* CNN.in_channels=1
* CNN.output_dim=64
* LSTMTextEncoder.embedding_dim?8
* LSTMTextEncoder.output_dim=32
* ClipModel.clip_embedding_dim=2
* epochs=5
* lr=3e-4

U-CLIP:
* description_encoder.target_sequence_length=10
* batch_size=128
* TIME_EMBEDDING_DIM = 128
* max_steps=1000
* min_beta=0.0001
* max_beta=0.02
* lr=1e-3
* epochs=10
* 


# References
[1] [Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021, July). Learning transferable visual models from natural language supervision. In International conference on machine learning (pp. 8748-8763). PmLR.](https://arxiv.org/pdf/2103.00020)

[2] [Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., & Chen, M. (2022). Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2), 3.](https://arxiv.org/abs/2204.06125)

[3] [Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. Advances in neural information processing systems, 33, 6840-6851.](https://arxiv.org/abs/2006.11239)
