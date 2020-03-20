import random
from typing import List

import pandas as pd
import torch
import torch.utils as utils

from nn_toolkit.vocab import Vocab

from src.tokenizer import ProjectTokenizer


class BaseMaskedDataset(utils.data.Dataset):
    """Base class for text data."""
    mask_rate = 0.15

    def __init__(self, df: pd.DataFrame, vocab: Vocab, tokenizer: ProjectTokenizer) -> None:
        self.df = df
        self.vocab = vocab
        self.tokenizer = tokenizer

    def get_and_set_tokens(self, sample: pd.Series) -> List:
        if 'tokens' in sample.index:
            return sample.tokens
        tokens = self.tokenizer.process_all([sample.text])[0]
        sample['tokens'] = tokens
        return tokens

    def transform(self, sample: pd.Series) -> dict:
        raise NotImplementedError()

    def __getitem__(self, idx: int) -> dict:
        sample = self.df.iloc[idx]
        return self.transform(sample)

    def get_token_mask(self, tokens: torch.LongTensor) -> torch.LongTensor:
        mask = torch.zeros_like(tokens).bernoulli_(self.mask_rate)
        mask[0] = 0
        mask[-1] = 0
        return mask

    def apply_mask(self, tokens: torch.LongTensor, mask: torch.LongTensor):
        masked_tokens = torch.zeros_like(tokens).copy_(tokens)
        for i, (token, mask_val) in enumerate(zip(tokens, mask)):
            if mask_val == 0:
                masked_tokens[i] = token
            else:
                coin = random.random()
                if coin < 0.8:  # replace with <mask>
                    masked_tokens[i] = self.vocab.mask_idx
                elif coin < 0.9:  # replace with random token
                    masked_tokens[i] = random.choice(range(self.vocab.size))
                else:  # do not repalce
                    masked_tokens[i] = token
        return masked_tokens

    def __len__(self) -> int:
        return self.df.shape[0]


class MLMDataset(BaseMaskedDataset):
    def transform(self, sample: pd.Series) -> dict:
        tokens = self.get_and_set_tokens(sample)
        tokens = self.vocab.map_to_ints(sample.tokens)
        tokens = torch.LongTensor(tokens)
        mask = self.get_token_mask(tokens)
        masked_tokens = self.apply_mask(tokens, mask)
        return {'masked_tokens': masked_tokens, 'tokens': tokens, 'mask': mask}


class ClasDataset(BaseMaskedDataset):
    mask_rate = 0.

    def transform(self, sample: pd.Series) -> dict:
        tokens = self.get_and_set_tokens(sample)
        tokens = self.vocab.map_to_ints(sample.tokens)
        tokens = torch.LongTensor(tokens)
        mask = self.get_token_mask(tokens)
        masked_tokens = self.apply_mask(tokens, mask)
        return {'masked_tokens': masked_tokens, 'tokens': tokens, 'mask': mask, 'label': sample.target}


class PredictionDataset(ClasDataset):

    def __init__(self, vocab: Vocab, tokenizer: ProjectTokenizer) -> None:
        self.vocab = vocab
        self.tokenizer = tokenizer
