import concurrent.futures
import multiprocessing
import re
from typing import Callable, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from nn_toolkit.vocab import Vocab, VocabBuilder
from src.data.augment_data import DataMatcher
from src.data.raw_dataset import RawDataset
from src.tokenizer import ProjectTokenizer


def extract_hashtags(tokens: List[str]) -> List[str]:
    hashtags = list(filter(lambda x: x.startswith('#'), tokens))
    return hashtags


def extract_mentions(tokens: List[str]) -> List[str]:
    mentions = list(filter(lambda x: x.startswith('@'), tokens))
    return mentions


def extract_emojis(tokens: str) -> List[str]:
    emojis = list(filter(lambda x: x == '<emoji>', tokens))
    return emojis


def _remove(tokens: List[str], condition: Callable) -> List[str]:
    return list(filter(lambda x: not condition(x), tokens))


def remove_hashtags(tokens: List[str]) -> List[str]:
    return _remove(tokens, lambda x: x.startswith('#'))


def remove_mentions(tokens: List[str]) -> List[str]:
    return _remove(tokens, lambda x: x.startswith('@'))


class ProcessedDataset:

    train_df: pd.DataFrame
    test_df: pd.DataFrame

    def __init__(self, raw: RawDataset, match_to_real: bool = False) -> None:
        self.__dict__.update(raw.__dict__)
        self.tokenizer = ProjectTokenizer()
        self.match_to_real = match_to_real

    def process(self) -> None:
        self.tokenize_df(self.train_df)
        self.tokenize_df(self.test_df)
        if self.extra_df.shape[0] > 0:
            self.tokenize_df(self.extra_df)
            self._match_extra_to_train()

    def clean(self, tokens: List[str]) -> List[str]:
        tokens = remove_hashtags(tokens)
        tokens = remove_mentions(tokens)
        return tokens

    def tokenize_df(self, df: pd.DataFrame) -> None:
            df['tokens'] = self.tokenizer.process_all(df.text)
            df['location_tokens'] = self.tokenizer.process_all(df.location.fillna(''))
            df['keyword_tokens'] = self.tokenizer.process_all(df.keyword.fillna('').str.replace('%20', ' '))
            with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
                df['hashtag_tokens'] = pool.map(extract_hashtags, df.tokens)
                df['mentions'] = pool.map(extract_mentions, df.tokens)
                df['emojis'] = pool.map(extract_emojis, df.text)

    def _match_extra_to_train(self, thresh: float = 0.1) -> None:
        """Force the extra data to 'look like' the train data.
        
        'look like' is best defined in `DataMatcher`
        """
        original_size = self.extra_df.shape[0]
        real_df = pd.concat([self.train_df, self.test_df], sort=False)
        self.matcher = DataMatcher(real_df, text_col='text', token_col='tokens')
        if self.match_to_real:
            self.extra_df = self.matcher.looks_real(self.extra_df, thresh=thresh)
        else:
            keep_prob = self.matcher._estimate_heuristics(self.extra_df)
            condition = keep_prob > thresh
            self.extra_df = self.extra_df[condition].copy()
        size = self.extra_df.shape[0]
        print(f'Extra data samples: {size}, removed {original_size - size}.')
