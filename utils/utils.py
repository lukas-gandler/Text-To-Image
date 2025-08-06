import os

import torch
import torch.nn as nn
import random
import numpy as np

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_model(model: nn.Module, path: str | os.PathLike) -> None:
    print(f"Saving model to {path}")
    torch.save(model.state_dict(), path)

def load_model(model: nn.Module, path: str | os.PathLike) -> nn.Module:
    print(f"Loading model from {path}")
    model.load_state_dict(torch.load(path, weights_only=True, map_location=torch.device('cpu')))
    return model