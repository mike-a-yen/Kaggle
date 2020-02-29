import concurrent.futures
import hashlib
import logging
import multiprocessing
import re
import string
from typing import Callable, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from nn_toolkit.vocab import Vocab, VocabBuilder
from src.data.augment_data import DataMatcher
from src.data.raw_dataset import RawDataset
from src.tokenizer import ProjectTokenizer

_WHITESPACE = set(string.whitespace)
_PUNCTUATION_WHITESPACE = set(string.punctuation + string.whitespace)
_PUNCTUATION_WHITESPACE_DIGITS = set(string.punctuation + string.whitespace + string.digits)


def extract(tokens: List[str], start_token: str, length: int = 1, skip_rules: Callable = None) -> List[str]:
    n = len(tokens)
    slow = 0
    extracted = []
    while slow < n:
        if tokens[slow].startswith(start_token) and (slow + length) < n:
            candidate = tokens[slow: slow + length]
            allow = skip_rules(candidate) if skip_rules is not None else True
            if allow:
                extracted.append(''.join(candidate))
                slow += length
            else:
                slow += 1
        else:
            slow += 1
    return extracted


def is_valid_twitter_token(tokens: List[str], start_token: str, first_char_of_second: str) -> bool:
    if len(tokens) <= 1:
        return False
    if not tokens[0] == start_token:
        return False
    if tokens[1][0] in first_char_of_second:
        return False
    return True


def is_valid_hashtag(tokens: List[str]) -> bool:
    n = len(tokens)
    if n <= 1:
        return False
    if not tokens[0] == '#':
        return False
    if tokens[1][0] in _PUNCTUATION_WHITESPACE:
        return False
    return True


def is_valid_mention(token: str) -> bool:
    n = len(token)
    if n <= 1:
        return False
    if token[0] != '@':
        return False
    if token[1] in _PUNCTUATION_WHITESPACE_DIGITS:
        return False
    return True


def find_next_whitespace(tokens: List[str], offset: int) -> int:
    for i, token in enumerate(tokens[offset:]):
        if token in _WHITESPACE:
            return i + offset
    return i


def extract_hashtags(tokens: List[str]) -> List[str]:
    n = len(tokens)
    slow = 0
    hashtags = []
    while slow < n:
        if tokens[slow] == '#':
            end = slow + 2
            candidate = tokens[slow: end]
            if is_valid_hashtag(candidate):
                hashtags.append(''.join(candidate))
                slow = end
            else:
                slow += 1
        else:
            slow += 1
    return hashtags


def extract_mentions(tokens: List[str]) -> List[str]:
    mentions = []
    for token in tokens:
        if is_valid_mention(token):
            mentions.append(token)
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


def hash_tokens(tokens: List[str]) -> str:
    """Compute the hash of a list of strings.
    Useful for detecting duplicates.
    """
    s = ''.join(tokens).encode()
    return hashlib.md5(s).hexdigest()


class ProcessedDataset:

    train_df: pd.DataFrame
    test_df: pd.DataFrame

    def __init__(self, raw: RawDataset, match_thresh: float = 0.) -> None:
        self.__dict__.update(raw.__dict__)
        self.tokenizer = ProjectTokenizer()
        self.match_thresh = match_thresh

    def process(self) -> None:
        self.tokenize_df(self.train_df)
        self.tokenize_df(self.test_df)
        if self.extra_df.shape[0] > 0:
            self.tokenize_df(self.extra_df)
            self._match_extra_to_train(self.match_thresh)
        self.drop_duplicates()

    def drop_duplicates(self) -> None:
        original_size = [self.train_df.shape[0], self.extra_df.shape[0]]
        self.train_df.drop_duplicates(subset=['token_hash'], inplace=True)
        self.extra_df.drop_duplicates(subset=['token_hash'], inplace=True)
        new_size = [self.train_df.shape[0], self.extra_df.shape[0]]
        diffs = [orig - new for orig, new in zip(original_size, new_size)]
        logging.info(f'Dropped {diffs[0]} duplicate samples from train_df.')
        logging.info(f'Dropped {diffs[1]} duplicate samples from extra_df.')

    def tokenize_df(self, df: pd.DataFrame) -> None:
            df['tokens'] = self.tokenizer.process_all(df.text.tolist())
            df['location_tokens'] = self.tokenizer.process_all(df.location.fillna('').tolist())
            df['keyword_tokens'] = self.tokenizer.process_all(df.keyword.fillna('').str.replace('%20', ' ').tolist())
            with multiprocessing.Pool(multiprocessing.cpu_count() // 2) as pool:
                df['hashtag_tokens'] = pool.map(extract_hashtags, df.tokens)
                df['mentions'] = pool.map(extract_mentions, df.tokens)
                df['emojis'] = pool.map(extract_emojis, df.text)
                df['token_hash'] = pool.map(hash_tokens, df.tokens)

    def _match_extra_to_train(self, thresh: float = 0.0) -> None:
        """Force the extra data to 'look like' the train data.
        
        'look like' is best defined in `DataMatcher`
        """
        original_size = self.extra_df.shape[0]
        real_df = pd.concat([self.train_df, self.test_df], sort=False)
        self.matcher = DataMatcher(real_df, text_col='text', token_col='tokens')
        if thresh > 0.:
            self.extra_df = self.matcher.looks_real(self.extra_df, thresh=thresh)
        else:
            keep_prob = self.matcher._estimate_heuristics(self.extra_df)
            self.extra_df = self.extra_df[keep_prob > 0.].copy()
        size = self.extra_df.shape[0]
        logging.info(f'Extra data samples: {size}, removed {original_size - size}.')
