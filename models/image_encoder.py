import torch.nn as nn

class SimpleImageEncoder(nn.Module):
    def __init__(self, in_channels:int=1, output_dim=64):
        """
        A simple image encoder model.
        :param in_channels: The number of input channels.
        :param output_dim: The dimension of the encoding vector.
        """

        super(SimpleImageEncoder, self).__init__()

        self.output_dim = output_dim

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.LazyLinear(out_features=64),   # make lazy so we can use it for different datasets
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=output_dim),
        )


    def forward(self, x):
        x_conv = self.conv(x)
        clip_encoding = self.fc(x_conv.flatten(start_dim=1))
        return clip_encoding