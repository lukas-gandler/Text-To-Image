import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision.utils import make_grid

import matplotlib.pyplot as plt


def plot_image_batch(images: torch.Tensor, n_row:int=8, padding:int=2, normalize:bool=True, cmap:str='gray') -> None:
    """
    Plotting a batch of images.
    :param images: The batch of images.
    :param n_row: The number of rows for the grid.
    :param padding: The padding of the grid.
    :param normalize: Normalize the images if set to True.
    :param cmap: The colormap to use.
    :return:
    """

    grid = make_grid(images, nrow=n_row, padding=padding, normalize=normalize)
    np_images = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(n_row, n_row))

    if np_images.shape[2] == 1:
        plt.imshow(np_images[:,:,0], cmap=cmap)
    else:
        plt.imshow(np_images)

    plt.axis('off')
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def plot_class_embedding_directions(classes, clip_model, description_encoder, device) -> None:
    """
    Plots the learned embedding space of the clip model (only works with 2D embedding space).
    :param classes: List of classes.
    :param clip_model: The CLIP model to use.
    :param description_encoder: The description encoder to use.
    :param device: The device to use.
    :return:
    """

    # Tokenize and pad class descriptions, move to device
    classes_tokenized = torch.stack([description_encoder(c) for c in classes]).to(device)

    # Get class text embeddings [num_classes, 2]
    class_embeddings = clip_model.get_text_embeddings(classes_tokenized).cpu()

    # Normalize to unit vectors for direction visualization
    class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True)

    plt.figure(figsize=(8,8))
    origin = torch.zeros(2)

    for i, c in enumerate(classes):
        vec = class_embeddings[i]
        plt.arrow(origin[0], origin[1], vec[0], vec[1], head_width=0.05, head_length=0.1, length_includes_head=True, alpha=0.8)
        plt.text(vec[0]*1.1, vec[1]*1.1, c, fontsize=12)

    plt.xlim(-1.3, 1.3)
    plt.ylim(-1.3, 1.3)
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.title("Class Embedding Directions (Normalized)")
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

@torch.no_grad()
def show_clip_image_classification(images, classes, clip_model, description_encoder, device) -> None:
    """
    Plots the class prediction based on the clip embeddings.
    :param images: The batch of images.
    :param classes: List of classes.
    :param clip_model: The CLIP model to use.
    :param description_encoder: The description encoder to use.
    :param device: The device to use.
    :return:
    """

    # Tokenize and pad class descriptions, then move to device
    classes_tokenized = torch.stack([description_encoder(c) for c in classes]).to(device)

    # Get class text embeddings
    class_embeddings = clip_model.get_text_embeddings(classes_tokenized)  # [num_classes, embed_dim]

    # Get image embeddings
    image_embeddings = clip_model.get_image_embeddings(images.to(device))  # [batch_size, embed_dim]

    # Compute cosine similarity between each image embedding and each class embedding
    similarity = image_embeddings @ class_embeddings.T  # [batch_size, num_classes]

    # Normalize similarity for visualization
    similarity = similarity.detach().cpu()

    # Plot images with similarity bar charts below
    batch_size = images.shape[0]
    fig, axs = plt.subplots(batch_size, 2, figsize=(8, 4 * batch_size))

    for i in range(batch_size):
        # Plot image
        img = images[i].cpu().permute(1, 2, 0).squeeze(-1)  # Handle grayscale or RGB accordingly
        axs[i, 0].imshow(img, cmap='gray' if img.shape[-1] == 1 else None)
        axs[i, 0].axis('off')

        # Plot similarity bar chart
        axs[i, 1].bar(classes, similarity[i].numpy())
        axs[i, 1].set_ylim([0, 1])
        axs[i, 1].set_ylabel('Similarity')

        # Fix warning by setting ticks explicitly
        axs[i, 1].set_xticks(range(len(classes)))
        axs[i, 1].set_xticklabels(classes, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()