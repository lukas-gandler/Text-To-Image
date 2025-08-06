from torch.utils.data import DataLoader
from tqdm import tqdm

from models import ClipModel, SimpleImageEncoder, LSTMTextEncoder
from utils import *
from dataloading import load_MNIST, CharacterEncoder

from models.diffusion_models import *

SEED = 42

CLIP_MODEL_PATH = 'mnist_clip.pth'
DIFFUSION_MODEL_PATH = 'mnist_diff.pth'

TIME_EMBEDDING_DIM = 128
CLIP_EMBEDDING_DIM = 2

def main() -> None:
    set_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    description_encoder = CharacterEncoder('abcdefghijklmnopqrstuvwxyz1234567890 .!?')
    train_loader, _ = load_MNIST(encoder=description_encoder, batch_size=128, num_workers=4)

    # Load clip model
    clip_model = ClipModel(image_encoder=SimpleImageEncoder(), text_encoder=LSTMTextEncoder(description_encoder.alphabet_size), clip_embedding_dim=CLIP_EMBEDDING_DIM)
    clip_model = load_model(clip_model, CLIP_MODEL_PATH).to(device)
    clip_model.eval()

    unet = ConditionalUNet(in_channels=1, time_embedding_dim=TIME_EMBEDDING_DIM, clip_embedding_dim=CLIP_EMBEDDING_DIM).to(device)
    ddpm = DiffusionModel(network=unet, max_steps=1000, min_beta=0.0001, max_beta=0.02).to(device)

    num_epochs = 10
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    ddpm = train(ddpm_model=ddpm, clip_model=clip_model, num_epochs=num_epochs, optimizer=optimizer, criterion=criterion, guidance_scale=3.0, train_loader=train_loader, device=device)
    save_model(ddpm, DIFFUSION_MODEL_PATH)

    num_samples = 64
    test_prompt = torch.stack([description_encoder('six')]*num_samples).to(device)
    test_prompt_clip_embedding = clip_model.get_text_embeddings(test_prompt)

    generation = ddpm.sample_new_image(num_samples=num_samples, clip_embedding=test_prompt_clip_embedding)
    plot_image_batch(generation)

def train(ddpm_model: nn.Module, clip_model: nn.Module, num_epochs:int, optimizer: torch.optim.Optimizer, criterion: nn.Module, guidance_scale:float, train_loader: DataLoader, device: torch.device) -> nn.Module:
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
    main()