import argparse
import torch

from dataloading import CharacterTokenizer
from models import SimpleImageEncoder, LSTMTextEncoder, ClipModel, ConditionalUNet, DiffusionModel
from utils import load_model, plot_image_batch


def main(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    description_encoder = CharacterTokenizer('abcdefghijklmnopqrstuvwxyz1234567890 .!?', args.encoder_target_length)

    # Loading models
    cnn = SimpleImageEncoder(args.cnn_in_channels, args.cnn_out_dims)
    lstm = LSTMTextEncoder(description_encoder.alphabet_size, args.lstm_embedding_dim, args.lstm_output_dim)
    clip_model = ClipModel(cnn, lstm, args.clip_embedding_dim)
    clip_model = load_model(clip_model, args.clip_model_path).to(device).eval()

    unet = ConditionalUNet(args.cnn_in_channels, args.time_embedding_dim, args.clip_embedding_dim)
    ddpm = DiffusionModel(unet, args.ddpm_max_steps, args.ddpm_min_beta, args.ddpm_max_beta)
    ddpm = load_model(ddpm, args.ddpm_model_path).to(device).eval()

    # Image generation
    model_prompt = torch.stack([description_encoder(args.prompt)]*args.num_samples).to(device)
    prompt_clip_embedding = clip_model.get_text_embeddings(model_prompt)

    generated_images = ddpm.sample_new_images(num_samples=args.num_samples, image_shape=(args.cnn_in_channels, 32,32), clip_embedding=prompt_clip_embedding)
    plot_image_batch(generated_images)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Generation parameters
    parser.add_argument('--prompt', type=str, default='six')
    parser.add_argument('--num_samples', type=int, default=64)

    parser.add_argument('--encoder_target_length', type=int, default=10)
    parser.add_argument('--clip_model_path', type=str, default='mnist_clip.pth')
    parser.add_argument('--ddpm_model_path', type=str, default='mnist_ddpm.pth')

    # Model parameters
    parser.add_argument('--cnn_in_channels', type=int, default=1)
    parser.add_argument('--cnn_out_dims', type=int, default=64)

    parser.add_argument('--lstm_embedding_dim', type=int, default=8)
    parser.add_argument('--lstm_output_dim', type=int, default=32)

    parser.add_argument('--clip_embedding_dim', type=int, default=2)

    parser.add_argument('--time_embedding_dim', type=int, default=128)
    parser.add_argument('--ddpm_max_steps', type=int, default=1000)
    parser.add_argument('--ddpm_min_beta', type=float, default=0.0001)
    parser.add_argument('--ddpm_max_beta', type=float, default=0.02)

    command_line_arguments = parser.parse_args()
    main(command_line_arguments)