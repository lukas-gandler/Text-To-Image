import re
import torch

class CharacterEncoder:
    def __init__(self, alphabet:str) -> None:
        clean_alphabet = re.sub(r'[^a-z0-9 .!?]','', alphabet.lower())
        duplicate_free_alphabet = ''.join(sorted(set(clean_alphabet), key=clean_alphabet.index))

        # Reserve special tokens
        self.START_TOKEN = 0
        self.END_TOKEN = 1

        self.alphabet = duplicate_free_alphabet
        self.alphabet_size = len(duplicate_free_alphabet)

        self.text_to_id = {char: idx for idx, char in enumerate(self.alphabet, start=self.START_TOKEN + 1)}

    def __call__(self, text:str) -> torch.Tensor:
        clean_text = re.sub(r'[^a-z0-9 .!?]','', text.lower())
        encoding = [self.START_TOKEN] + [self.text_to_id[char] for char in clean_text] + [self.END_TOKEN]

        encoding = torch.Tensor(encoding)
        return encoding