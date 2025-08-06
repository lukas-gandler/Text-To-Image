import torch

from typing import Tuple, Optional, Callable
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision.datasets import MNIST
from torchvision.transforms import transforms



def load_MNIST(encoder:Optional[Callable]=None, batch_size:int=64, num_workers:int=0, data_root:str='./data', download:bool=True) -> Tuple[DataLoader, DataLoader]:
    train_dataset = MNISTDataset(string_encoder=encoder, train_fold=True, data_dir=data_root, download=download)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=get_collate_fn(encoder))

    validation_dataset = MNISTDataset(string_encoder=encoder, train_fold=False, data_dir=data_root, download=download)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=get_collate_fn(encoder))

    return train_loader, validation_loader

def get_collate_fn(encoder:Optional[Callable]=None) -> Callable:
    def collate_fn(batch):
        if encoder is not None:
            images, tokens = zip(*batch)

            images = torch.stack(images, dim=0)
            padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=encoder.PAD_TOKEN)
            return images, padded_tokens
        else:
            return batch

    return collate_fn


class MNISTDataset(Dataset):
    def __init__(self, string_encoder:Optional[Callable]=None, train_fold:bool=True, data_dir:str='./data', download:bool=True) -> None:

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

        if self.encoder:
            string_label = self.description_dictionary[label]
            string_encoding = self.encoder(string_label)
            return image, string_encoding
        else:
            return image, label
