import pandas as pd
from nn_toolkit.vocab import Vocab, VocabBuilder

from raw_dataset import RawDataset
from tokenizer import TweetTokenizer


class ProcessedDataset:

    train_df: pd.DataFrame
    test_df: pd.DataFrame

    def __init__(self, raw: RawDataset) -> None:
        self.__dict__.update(raw.__dict__)
        self.tokenizer = TweetTokenizer()

    def process(self) -> None:
        self.tokenize_df(self.train_df)
        self.tokenize_df(self.test_df)

    def tokenize_df(self, df: pd.DataFrame) -> None:
        df['tokens'] = df.text.apply(self.tokenizer)
        df['location_tokens'] = df.location.fillna('').apply(self.tokenizer)

    