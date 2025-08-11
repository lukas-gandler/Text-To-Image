import argparse

from torch.utils.data import DataLoader
from tqdm import tqdm

from models import ClipModel, SimpleImageEncoder, LSTMTextEncoder
from utils import *
from dataloading import load_mnist, CharacterTokenizer

from models.diffusion_model import *

SEED = 42

def main(args: argparse.Namespace) -> None:
    set_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    description_encoder = CharacterTokenizer('abcdefghijklmnopqrstuvwxyz1234567890 .!?', args.encoder_target_length)
    train_loader, _ = load_mnist(description_encoder, args.batch_size, args.num_workers)
    cnn_in_channels = 1

    # Load clip model
    cnn = SimpleImageEncoder(cnn_in_channels, args.cnn_output_dim)
    lstm = LSTMTextEncoder(description_encoder.alphabet_size, args.lstm_embedding_dim, args.lstm_output_dim)
    clip_model = ClipModel(cnn, lstm, args.clip_embedding_dim)
    clip_model = load_model(clip_model, args.clip_model_path).to(device).eval()

    unet = ConditionalUNet(cnn_in_channels, args.time_embedding_dim, args.clip_embedding_dim).to(device)
    ddpm = DiffusionModel(unet, args.ddpm_max_steps, args.ddpm_min_beta, args.ddpm_max_beta).to(device)

    num_epochs = args.epochs
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    ddpm = train(ddpm, clip_model, num_epochs, optimizer, criterion, train_loader, device)
    save_model(ddpm, args.save_path)

def train(ddpm_model: nn.Module, clip_model: nn.Module, num_epochs:int, optimizer: torch.optim.Optimizer, criterion: nn.Module, train_loader: DataLoader, device: torch.device) -> nn.Module:
    ddpm_model.train()
    train_losses = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        losses = []
        progress_bar = tqdm(train_loader)
        for images, descriptions in progress_bar:
            # Prepare the images, descriptions and create random time-steps
            clean_images, descriptions = images.to(device), descriptions.to(device)
            step_tensor = torch.randint(0, ddpm_model.max_steps, size=(images.shape[0],)).to(device)

            # Get the description embeddings from clip
            description_embeddings = clip_model.get_text_embeddings(descriptions).detach()

            # Forward pass to generate noisy images and predict the added noise
            noisy_images, noise = ddpm_model.forward_process(clean_images, step_tensor)
            noise_prediction = ddpm_model.backward_process(noisy_images, step_tensor, description_embeddings)

            # Back-propagate and optimize
            loss = criterion(noise_prediction, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            progress_bar.set_description(f'  Training loss: {sum(losses) / len(losses):.3f}')

        train_losses.append(sum(losses) / len(losses))

    return ddpm_model

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
    parser.add_argument('--time_embedding_dim', type=int, default=128)
    parser.add_argument('--ddpm_max_steps', type=int, default=1000)
    parser.add_argument('--ddpm_min_beta', type=float, default=0.0001)
    parser.add_argument('--ddpm_max_beta', type=float, default=0.02)

    # General settings
    parser.add_argument('--encoder_target_length', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='mnist_ddpm.pth')
    parser.add_argument('--clip_model_path', type=str, default='mnist_clip.pth')

    command_line_args = parser.parse_args()
    main(command_line_args)