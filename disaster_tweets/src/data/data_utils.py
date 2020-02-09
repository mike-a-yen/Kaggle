from typing import List

import pandas as pd
from nn_toolkit.vocab import Vocab

from src.data.split_dataset import SplitDataset


def map_token_column_to_ints(split_ds: SplitDataset, token_col: str, vocab: Vocab, output_col:str) -> None:
    split_ds.train_df[output_col] = split_ds.train_df[token_col].apply(vocab.map_to_ints)
    split_ds.val_df[output_col] = split_ds.val_df[token_col].apply(vocab.map_to_ints)
    split_ds.trainval_df[output_col] = split_ds.trainval_df[token_col].apply(vocab.map_to_ints)
    split_ds.test_df[output_col] = split_ds.test_df[token_col].apply(vocab.map_to_ints)
    if split_ds.extra_df.shape[0] > 0:
        split_ds.extra_df[output_col] = split_ds.extra_df[token_col].apply(vocab.map_to_ints)
    return


def concat_cols(split_ds: SplitDataset, cols: List[str], name: str, sep: str = None) -> None:
    def concat(lsts: List[List]) -> List:
        new_lst = []
        for i, lst in enumerate(lsts):
            if sep is not None and i > 0:
                new_lst += [sep]
            new_lst += lst
        return new_lst

    for df in split_ds:
        df[name] = df[cols].apply(lambda row: concat(row), axis=1)
    return
