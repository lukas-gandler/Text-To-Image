import math
import torch
import torch.nn as nn

from typing import Tuple, Optional

class DiffusionModel(nn.Module):
    def __init__(self, network: nn.Module, max_steps: int, min_beta: float, max_beta: float):
        super(DiffusionModel, self).__init__()

        self.network = network
        self.max_steps = max_steps

        betas = torch.linspace(min_beta, max_beta, max_steps)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, 0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)

    def forward_process(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes t-th image in the forward Markov process through closed form solution based on clean x0 and alphas.
        :param x0: Clean ground truth image
        :param t: Time-step
        :return: noisy image, added noise
        """

        batch_size = x0.shape[0]
        a_bar = self.alpha_bars[t].reshape(batch_size, 1, 1, 1)
        noise = torch.randn_like(x0).to(x0.device)

        xt = a_bar.sqrt() * x0 + (1 - a_bar).sqrt() * noise
        return xt, noise

    def backward_process(self, x: torch.Tensor, time: torch.Tensor, clip_embedding: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Pass image through the network to predict the added noise.
        :param x: Noisy image
        :param time: Time-step
        :param clip_embedding: CLIP text embedding
        :return: Noise estimation
        """
        return self.network(x, time, clip_embedding)

    @torch.no_grad()
    def sample_new_image(self, num_samples:int = 64, image_shape:Tuple[int, int, int] = (1, 32, 32), clip_embedding: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Sample new images according to the algorithm presented by Ho et.al. (2020).
        :param num_samples: The number of new images to generate in a batch
        :param image_shape: The shape of the images
        :param clip_embedding: CLIP text embedding
        :return: Generated batch of new images
        """

        device = self.betas.device

        channels, height, width = image_shape
        image = torch.randn(num_samples, channels, height, width).to(device)

        for t in reversed(range(self.max_steps)):
            # Sample z from standard normal distribution and scale my delta_t
            z = torch.randn(num_samples, channels, height, width).to(device) if t > 1 else torch.zeros(num_samples, channels, height, width).to(device)
            sigma = self.betas[t].sqrt()
            scaled_z = sigma * z

            # Compute the noise prediction and scale with alpha_t and alpha-bar_t
            step_tensor = torch.tensor([t] * num_samples).to(device)
            noise_prediction = self.backward_process(image, step_tensor, clip_embedding)
            scaled_noise_prediction = (1 - self.alphas[t]) / (1 - self.alpha_bars[t]).sqrt() * noise_prediction

            # Remove noise from image and continue iteration
            image = (1 / self.alphas[t].sqrt()) * (image - scaled_noise_prediction) + scaled_z

        return image

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels:int=1, time_embedding_dim:int=128, clip_embedding_dim:int=64):
        super(ConditionalUNet, self).__init__()

        self.time_embedding = TimeEmbedding(embedding_dim=time_embedding_dim)

        # Down-sampling
        self.conv1 = ConvolutionBlock(in_channels=in_channels, out_channels=64, time_emb_dim=time_embedding_dim, clip_dim=clip_embedding_dim)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = ConvolutionBlock(in_channels=64, out_channels=128, time_emb_dim=time_embedding_dim, clip_dim=clip_embedding_dim)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = ConvolutionBlock(in_channels=128, out_channels=256, time_emb_dim=time_embedding_dim, clip_dim=clip_embedding_dim)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = ConvolutionBlock(in_channels=256, out_channels=512, time_emb_dim=time_embedding_dim, clip_dim=clip_embedding_dim)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.bottleneck = ConvolutionBlock(in_channels=512, out_channels=1024, time_emb_dim=time_embedding_dim, clip_dim=clip_embedding_dim)

        # Up-sampling
        self.up1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.conv5 = ConvolutionBlock(in_channels=1024, out_channels=512, time_emb_dim=time_embedding_dim, clip_dim=clip_embedding_dim)

        self.up2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.conv6 = ConvolutionBlock(in_channels=512, out_channels=256, time_emb_dim=time_embedding_dim, clip_dim=clip_embedding_dim)

        self.up3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv7 = ConvolutionBlock(in_channels=256, out_channels=128, time_emb_dim=time_embedding_dim, clip_dim=clip_embedding_dim)

        self.up4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv8 = ConvolutionBlock(in_channels=128, out_channels=64, time_emb_dim=time_embedding_dim, clip_dim=clip_embedding_dim)

        self.final_layer = nn.Conv2d(in_channels=64, out_channels=in_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor, clip_embedding: Optional[torch.Tensor]=None) -> torch.Tensor:
        t_emb = self.time_embedding(time_embedding)

        # Down-sampling
        x1 = self.conv1(x, t_emb, clip_embedding)
        x2 = self.conv2(self.pool1(x1), t_emb, clip_embedding)
        x3 = self.conv3(self.pool2(x2), t_emb, clip_embedding)
        x4 = self.conv4(self.pool3(x3), t_emb, clip_embedding)

        # Bottleneck
        x5 = self.bottleneck(self.pool4(x4), t_emb, clip_embedding)

        # Up-sampling
        output = self.up1(x5)
        output = self.conv5(torch.cat([x4, output], dim=1), t_emb, clip_embedding)

        output = self.up2(output)
        output = self.conv6(torch.cat([x3, output], dim=1), t_emb, clip_embedding)

        output = self.up3(output)
        output = self.conv7(torch.cat([x2, output], dim=1), t_emb, clip_embedding)

        output = self.up4(output)
        output = self.conv8(torch.cat([x1, output], dim=1), t_emb, clip_embedding)

        output = self.final_layer(output)
        return output

class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim:int) -> None:
        super(TimeEmbedding, self).__init__()

        self.embedding_dim = embedding_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        time_embedding = self.get_sinusoidal_timestep_embedding(t, self.embedding_dim)
        return self.time_mlp(time_embedding)

    @staticmethod
    def get_sinusoidal_timestep_embedding(time_steps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=time_steps.device) * -emb)
        emb = time_steps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, time_emb_dim:int, clip_dim:int, kernel_size:int=3, stride:int=1, padding:int=1, num_groups:int=2):
        super(ConvolutionBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.norm = nn.GroupNorm(num_groups=num_groups if out_channels > 1 else 1, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.time_embedding_projection = nn.Linear(in_features=time_emb_dim, out_features=out_channels)
        self.clip_embedding_projection = nn.Linear(in_features=clip_dim, out_features=out_channels)

    def forward(self, x:torch.Tensor, time_embedding:torch.Tensor, clip_embedding:Optional[torch.Tensor]=None) -> torch.Tensor:
        # compute time projection
        time_embedding = self.time_embedding_projection(time_embedding)
        time_embedding = time_embedding[:, :, None, None]

        # compute clip projection
        if clip_embedding is not None:
            clip_embedding = self.clip_embedding_projection(clip_embedding)
            clip_embedding = clip_embedding[:, :, None, None]
        else:
            clip_embedding = 0

        # pass through block
        x = self.conv1(x)
        x = self.norm(x)
        x = x + time_embedding + clip_embedding  # condition on time and clip embedding
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm(x)
        x = self.relu(x)

        return x
