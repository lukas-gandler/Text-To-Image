import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt

from utils import set_seed, save_model
from dataloading import load_MNIST
from dataloading.CharacterEncoder import CharacterEncoder

from models.clip import ClipModel, ClipLoss
from models.image_encoders import SimpleImageEncoder
from models.text_encoder import LSTMTextEncoder

SEED = 42
SAVE_PATH = 'mnist_clip.pth'

def main() -> None:
    set_seed(SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    description_encoder = CharacterEncoder('abcdefghijklmnopqrstuvwxyz1234567890 .!?')
    train_loader, validation_loader = load_MNIST(encoder=description_encoder, batch_size=128)

    cnn = SimpleImageEncoder()
    lstm = LSTMTextEncoder(alphabet_size=description_encoder.alphabet_size)
    clip_model = ClipModel(image_encoder=cnn, text_encoder=lstm, clip_embedding_dim=2).to(device)

    epochs = 5
    clip_loss = ClipLoss().to(device)
    optimizer = torch.optim.AdamW(list(clip_model.parameters()) + list(clip_loss.parameters()), lr=3e-4)

    train(clip_model=clip_model, num_epochs=epochs, optimizer=optimizer, clip_loss=clip_loss, train_loader=train_loader, validation_loader=validation_loader, device=device)
    save_model(clip_model, path=SAVE_PATH)


def train(clip_model:nn.Module, num_epochs:int, optimizer:torch.optim.Optimizer, clip_loss: nn.Module,
          train_loader: DataLoader, validation_loader: DataLoader, device:torch.device) -> None:

    training_losses = []
    validation_losses = []

    print(f'Initial testing')
    validation_loss = validation_loop(clip_model, clip_loss, validation_loader, device)
    validation_losses.append(validation_loss)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        training_loss = training_loop(clip_model, optimizer, clip_loss, train_loader, device)
        training_losses.append(training_loss)

        validation_loss = validation_loop(clip_model, clip_loss, validation_loader, device)
        validation_losses.append(validation_loss)


def training_loop(clip_model:nn.Module, optimizer:torch.optim.Optimizer, clip_loss:nn.Module, train_loader:DataLoader, device:torch.device) -> float:
    clip_model.train()
    losses = []

    progress_bar = tqdm(train_loader)
    for image, description in progress_bar:
        image, description = image.to(device), description.to(device)

        image_embeddings, description_embeddings = clip_model(image, description)
        loss = clip_loss(image_embeddings, description_embeddings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Clamp temperature to reasonable range (e.g., log(1/100) to log(1/0.01))
        clip_loss.t.data.clamp_(min=torch.log(torch.tensor(1/100.0)).to(device), max=torch.log(torch.tensor(1/0.01)).to(device))

        losses.append(loss.item())
        progress_bar.set_description(f'  Training loss: {sum(losses) / len(losses):.3f}')

    return sum(losses) / len(losses)

@torch.no_grad()
def validation_loop(clip_model: nn.Module, clip_loss:nn.Module, validation_loader:DataLoader, device:torch.device) -> float:
    clip_model.eval()
    losses = []

    progress_bar = tqdm(validation_loader)
    for image, description in progress_bar:
        image, description = image.to(device), description.to(device)

        image_embeddings, description_embeddings = clip_model(image, description)
        loss = clip_loss(image_embeddings, description_embeddings)

        losses.append(loss.item())
        progress_bar.set_description(f'  Validation loss: {sum(losses) / len(losses):.3f}')

    return sum(losses) / len(losses)


if __name__ == '__main__':
    main()