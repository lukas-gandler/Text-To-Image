import torch

from typing import Tuple
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


def load_MNIST(encoder:callable, batch_size:int=64, num_workers:int=0, data_root:str='./data', download:bool=True) -> Tuple[DataLoader, DataLoader]:
    train_dataset = MNISTDataset(string_encoder=encoder, train_fold=True, data_dir=data_root, download=download)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    validation_dataset = MNISTDataset(string_encoder=encoder, train_fold=False, data_dir=data_root, download=download)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, validation_loader



class MNISTDataset(Dataset):
    def __init__(self, string_encoder:callable, train_fold:bool=True, data_dir:str='./data', download:bool=True) -> None:
        self.dataset = MNIST(root=data_dir, train=train_fold, download=download, transform=transforms.ToTensor())
        self.encoder = string_encoder

        # Here we can choose some captions for the images
        self.description_dictionary = {
            0: 'Handwritten image of the digit 0',
            1: 'Handwritten image of the digit 1',
            2: 'Handwritten image of the digit 2',
            3: 'Handwritten image of the digit 3',
            4: 'Handwritten image of the digit 4',
            5: 'Handwritten image of the digit 5',
            6: 'Handwritten image of the digit 6',
            7: 'Handwritten image of the digit 7',
            8: 'Handwritten image of the digit 8',
            9: 'Handwritten image of the digit 9',
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = self.dataset[index]

        # TODO: handle padding when descriptions are not the same length
        string_label = self.description_dictionary[label]
        string_encoding = self.encoder(string_label)

        return image, string_encoding
