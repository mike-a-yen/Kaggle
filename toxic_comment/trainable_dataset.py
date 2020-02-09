import string

import numpy as np
import pandas as pd

from exports.vocab import Vocabulary, VocabEncoder
from exports.exp_00 import ProcessedDataset, split_df


class TrainableDataset:
    def __init__(self, processed_dataset: ProcessedDataset) -> None:
        self.toxicity_columns = processed_dataset.toxicity_subtypes
        self.input_column = 'comment_text'
        self.trainval_df = processed_dataset.train_df
        self.train_df, self.val_df = split_df(self.trainval_df, frac=0.1)
        self.test_df = processed_dataset.test_df
        
        self.n_train = self.train_df.shape[0]
        self.n_val = self.val_df.shape[0]
        self.n_test = self.test_df.shape[0]

        vocab = Vocabulary(list(string.printable))
        self.vocab_encoder = VocabEncoder(vocab)
        self.training = True
        self.testing = False

    def __len__(self):
        if self.training:
            return self.train_df.shape[0]
        elif self.testing:
            return self.test_df.shape[0]
        else:
            return self.val_df.shape[0]

    def get_row(self, idx):
        if self.training:
            return self.train_df.iloc[idx]
        elif self.testing:
            return self.test_df.iloc[idx]
        else:
            return self.val_df.iloc[idx]

    def __getitem__(self, idx: int):
        row = self.get_row(idx)
        X = row[self.input_column]
        #X = self.input_transform(X)
        if not self.testing:
            Y = row[self.toxicity_columns]
            Y = self.output_transform(Y)
            return X, Y
        return X
    
    def input_transform(self, X):
        return self.vocab_encoder.encode(X)

    def output_transform(self, Y):
        toxic_int = (Y >= 0.5).astype(int)
        return toxic_int
    
    @property
    def mean_target(self):
        return self.train_df[self.toxicity_columns].values.mean(axis=0)

    def __repr__(self) -> str:
        msg = f'Train samples: {self.n_train}, Val samples: {self.n_val}, Test samples: {self.n_test}'
        msg += f'\n' + ' '.join([f'{col}: {t:0.4f}' for t, col in zip(self.mean_target, self.toxicity_columns)])
        return msg
