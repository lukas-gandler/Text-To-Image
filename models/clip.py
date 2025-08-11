from asyncio import sleep
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClipModel(nn.Module):
    def __init__(self, image_encoder: nn.Module, text_encoder: nn.Module, clip_embedding_dim=16):
        """
        Torch module for the clip model. Takes an image encoder and a text encoder produces embeddings of the specified embedding dimension.
        :param image_encoder: The image encoder.
        :param text_encoder: The text encoder.
        :param clip_embedding_dim: The embedding dimension of the clip model.
        """

        super(ClipModel, self).__init__()

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        self.clip_embedding_dim = clip_embedding_dim

        # learnable projection matrices for image and text embeddings
        self.image_projection = nn.Linear(in_features=image_encoder.output_dim, out_features=self.clip_embedding_dim)
        self.text_projection  = nn.Linear(in_features=text_encoder.output_dim,  out_features=self.clip_embedding_dim)

    def forward(self, image: torch.Tensor, text_encoding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passes the image through the image encoder and the tokenized text through the text encoder. Produces the shared embedding.
        :param image: The input image tensor.
        :param text_encoding: The tokenized text encoding.
        :return: Tuple of image embedding and text embedding
        """

        # extract feature representation of each modality
        image_embedding = self.image_encoder(image)
        text_embedding  = self.text_encoder(text_encoding)

        # joint multimodal embedding [BATCH, CLIP_EMBEDDING_DIM]
        image_embedding = F.normalize(self.image_projection(image_embedding), p=2, dim=1)
        text_embedding  = F.normalize(self.text_projection(text_embedding), p=2, dim=1)

        return image_embedding, text_embedding

    def get_text_embeddings(self, text: torch.Tensor) -> torch.Tensor:
        """
        Returns the embedding of the specified text.
        :param text: The input text.
        :return: Text embedding
        """

        text_embedding = self.text_encoder(text)
        text_embedding = F.normalize(self.text_projection(text_embedding), p=2, dim=1)
        return text_embedding

    def get_image_embeddings(self, image: torch.Tensor) -> torch.Tensor:
        """
        Returns the embedding of the specified image.
        :param image: The input image.
        :return: Image embedding
        """

        image_embedding = self.image_encoder(image)
        image_embedding = F.normalize(self.image_projection(image_embedding), p=2, dim=1)
        return image_embedding

class ClipLoss(nn.Module):
    def __init__(self) -> None:
        """
        Defines the loss function fo the clip model based on the paper 'Learning transferable visual models from natural language supervision.'
        """
        super(ClipLoss, self).__init__()

        self.t = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))  # Learnable log temperature

    def forward(self, image_embedding, text_embedding):
        batch_size = image_embedding.shape[0]
        device = image_embedding.device

        # we assume that the embeddings produced by the image and text encoders already have the same dimensions
        assert image_embedding.shape[0] == text_embedding.shape[0] and image_embedding.shape[1] == text_embedding.shape[1], \
            f'Image ({image_embedding.shape}) and text embedding ({text_embedding.shape}) must have same dimensions'

        # scaled pairwise cosine similarities [BATCH, BATCH]
        logits = (image_embedding @ text_embedding.t()) * torch.exp(self.t)

        # symmetric loss function
        labels = torch.arange(batch_size).to(device)
        loss_image = F.cross_entropy(logits, labels)
        loss_text  = F.cross_entropy(logits.t(), labels)

        loss = (loss_image + loss_text) / 2
        return loss


