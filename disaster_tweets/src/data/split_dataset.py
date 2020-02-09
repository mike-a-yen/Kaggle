from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.processed_dataset import ProcessedDataset


class SplitDataset:
    def __init__(self, processed: ProcessedDataset, frac: float = 0.2) -> None:
        self.train_df, self.val_df = [
            df.copy() for df in
            train_test_split(processed.train_df, test_size=frac)
        ]
        self.trainval_df = pd.concat([self.train_df, self.val_df], sort=False).copy()
        self.test_df = processed.test_df
        self.extra_df = processed.extra_df
    
    def __iter__(self) -> Tuple[str, pd.DataFrame]:
        df_names = ['train_df', 'val_df', 'test_df', 'extra_df']
        for name in df_names:
            df = getattr(self, name)
            if df.shape[0] == 0:
                continue
            yield name, df

    def __repr__(self) -> str:
        return f'Train samples: {self.train_df.shape[0]}' \
            f'\nValid samples: {self.val_df.shape[0]}' \
            f'\nTest samples: {self.test_df.shape[0]}'
