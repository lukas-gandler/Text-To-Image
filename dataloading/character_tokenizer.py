import re
import torch

class CharacterTokenizer(object):
    def __init__(self, alphabet:str='abcdefghijklmnopqrstuvwxyz1234567890 .!?', target_sequence_length:int=10) -> None:
        """
        Tokenizes its given input string based on the given alphabet string and pads or truncates them to the target sequence length.
        :param alphabet: the string to tokenize the characters by.
        :param target_sequence_length: the length of the target sequence (excluding special START and END token).
        """

        clean_alphabet = re.sub(r'[^a-z0-9 .!?]','', alphabet.lower())
        duplicate_free_alphabet = ''.join(sorted(set(clean_alphabet), key=clean_alphabet.index))

        # Reserve special tokens
        self.PAD_TOKEN = 0
        self.START_TOKEN = 1
        self.END_TOKEN = 2

        self.alphabet = duplicate_free_alphabet
        self.alphabet_size = len(duplicate_free_alphabet)

        self.text_to_id = {char: idx for idx, char in enumerate(self.alphabet, start=3)}
        self.target_sequence_length = target_sequence_length + 2

    def __call__(self, text:str) -> torch.Tensor:
        clean_text = re.sub(r'[^a-z0-9 .!?]','', text.lower())
        encoding = [self.START_TOKEN] + [self.text_to_id[char] for char in clean_text] + [self.END_TOKEN]

        if len(encoding) > self.target_sequence_length:
            encoding = encoding[:self.target_sequence_length + 1] + [self.END_TOKEN]  # also consider START and END token
        else:
            pad_length = self.target_sequence_length - len(encoding)
            encoding = encoding + [self.PAD_TOKEN] * pad_length

        encoding = torch.Tensor(encoding)
        return encoding