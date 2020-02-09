from typing import Callable, List, Tuple

import pandas as pd
import torch
import torch.utils as utils


class TweetLMDataset(utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, transform_fn: Callable) -> None:
        self.df = df
        self.transform_fn = transform_fn

    def __getitem__(self, idx: int) -> torch.LongTensor:
        sample = self.df.iloc[idx]
        tokens = self.transform_fn(sample)
        return tokens

    def __len__(self) -> int:
        return self.df.shape[0]


class TweetClasDataset(utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, transform_fn: Callable) -> None:
        self.df = df
        self.transform_fn = transform_fn
        self.classes = [0, 1]
        self.c = len(self.classes)

    def __getitem__(self, idx: int) -> Tuple:
        sample = self.df.iloc[idx]
        tokens = self.transform_fn(sample)
        label = self.get_label(sample)
        return tokens, label

    def __len__(self) -> int:
        return self.df.shape[0]

    def get_label(self, sample: pd.Series) -> torch.Tensor:
        if 'target' not in sample:
            return torch.LongTensor([-1])
        return torch.LongTensor([sample.target])
