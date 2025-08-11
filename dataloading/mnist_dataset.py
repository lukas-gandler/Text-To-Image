import torch
import torch.nn.functional as F

from typing import Tuple, Optional, Callable

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision.datasets import MNIST
from torchvision.transforms import transforms



def load_mnist(encoder:Optional[Callable]=None, batch_size:int=64, num_workers:int=0, data_root:str= './data', download:bool=True) -> Tuple[DataLoader, DataLoader]:
    train_dataset = MNISTDataset(string_encoder=encoder, train_fold=True, data_dir=data_root, download=download)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    validation_dataset = MNISTDataset(string_encoder=encoder, train_fold=False, data_dir=data_root, download=download)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, validation_loader


class MNISTDataset(Dataset):
    def __init__(self, string_encoder:Optional[Callable]=None, train_fold:bool=True, data_dir:str='./data', download:bool=True) -> None:
        """
        Custom dataset for MNIST dataset where the label can be converted into a tokenized description.
        :param string_encoder: Tokenizer for the description encoding.
        :param train_fold: Loads the training fold of set to True otherwise loads the test fold.
        :param data_dir: The directory to load the data from.
        :param download: Downloads the data from the internet if set to True.
        """

        transformation = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5],[0.5])])
        self.dataset = MNIST(root=data_dir, train=train_fold, download=download, transform=transformation)
        self.encoder = string_encoder

        # Here we can choose some captions for the images
        self.description_dictionary = {
            0: 'zero',
            1: 'one',
            2: 'two',
            3: 'three',
            4: 'four',
            5: 'five',
            6: 'six',
            7: 'seven',
            8: 'eight',
            9: 'nine',
        }


    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = self.dataset[index]

        if self.encoder is not None:
            string_label = self.description_dictionary[label]
            tokenized_string = self.encoder(string_label)
            return image, tokenized_string
        else:
            return image, label
