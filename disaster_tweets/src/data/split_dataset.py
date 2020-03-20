from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.processed_dataset import ProcessedDataset


class SplitDataset:
    def __init__(self, processed: ProcessedDataset, frac: float = 0.2) -> None:
        self.dfs = dict()
        self.text_col = processed.text_col
        self.target_col = processed.target_col
        self.tokenizer = processed.tokenizer
        if frac > 0:
            self.trainval_df, self.test_df = [
                df.copy() for df in
                train_test_split(processed.train_df, test_size=frac)
            ]
            self.train_df, self.val_df = [
                df.copy() for df in train_test_split(self.trainval_df, test_size=frac)
            ]
        else:
            self.train_df = processed.train_df
            self.val_df = pd.DataFrame(columns=self.train_df.columns)
        self.trainval_df = pd.concat([self.train_df, self.val_df], sort=False).copy()
        self.extra_df = processed.extra_df

    def __iter__(self) -> Tuple[str, pd.DataFrame]:
        for name in self.dfs:
            df = getattr(self, name)
            if df.shape[0] == 0:
                continue
            yield name, df

    def __setattr__(self, name, value) -> None:
        if name.endswith('_df'):
            self.dfs[name] = value
        super().__setattr__(name, value)

    def __repr__(self) -> str:
        return f'Train samples: {self.train_df.shape[0]}' \
            f'\nValid samples: {self.val_df.shape[0]}' \
            f'\nTest samples: {self.test_df.shape[0]}'
