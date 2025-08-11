import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import set_seed, save_model
from dataloading import load_mnist, CharacterTokenizer
from models import ClipModel, ClipLoss, SimpleImageEncoder, LSTMTextEncoder

SEED = 42

def main(args: argparse.Namespace) -> None:
    set_seed(SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    description_encoder = CharacterTokenizer('abcdefghijklmnopqrstuvwxyz1234567890 .!?', args.encoder_target_length)
    train_loader, validation_loader = load_mnist(description_encoder, args.batch_size, args.num_workers)
    cnn_in_channels = 1

    cnn = SimpleImageEncoder(cnn_in_channels, args.cnn_output_dim)
    lstm = LSTMTextEncoder(description_encoder.alphabet_size, args.lstm_embedding_dim, args.lstm_output_dim)
    clip_model = ClipModel(cnn, lstm, args.clip_embedding_dim).to(device)

    epochs = args.epochs
    clip_loss = ClipLoss().to(device)
    optimizer = torch.optim.AdamW(list(clip_model.parameters()) + list(clip_loss.parameters()), lr=args.lr)

    train(clip_model, epochs, optimizer, clip_loss, train_loader, validation_loader, device)
    save_model(clip_model, args.save_path)


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
    parser = argparse.ArgumentParser()

    # Training settings
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_workers', type=int, default=4)

    # Model settings
    parser.add_argument('--cnn_output_dim', type=int, default=64)
    parser.add_argument('--lstm_embedding_dim', type=int, default=8)
    parser.add_argument('--lstm_output_dim', type=int, default=32)
    parser.add_argument('--clip_embedding_dim', type=int, default=2)

    # General settings
    parser.add_argument('--encoder_target_length', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='mnist_clip.pth')

    command_line_args = parser.parse_args()
    main(command_line_args)