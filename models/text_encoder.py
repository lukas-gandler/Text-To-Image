import re

import torch
import torch.nn as nn

class LSTMTextEncoder(nn.Module):
    def __init__(self, alphabet_size:int, embedding_dim:int=8, output_dim:int=32) -> None:
        super(LSTMTextEncoder, self).__init__()

        self.output_dim = output_dim

        self.embedding = nn.Embedding(num_embeddings=alphabet_size, embedding_dim=embedding_dim)
        self.lstm_layer = nn.LSTM(input_size=embedding_dim, hidden_size=256, num_layers=3, batch_first=True)
        self.linear = nn.Linear(in_features=256, out_features=output_dim)


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        embedded_input = self.embedding(x.int())

        output, (hidden, cells) = self.lstm_layer(embedded_input, None)
        last_hidden = hidden[-1]

        clip_embedding = self.linear(last_hidden)
        return clip_embedding
