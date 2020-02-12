import hashlib
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from nn_toolkit.vocab import Vocab


_PROJECT_DIR = Path(__file__).parents[1].resolve()
_DATA_DIR = _PROJECT_DIR / 'data'
_EXTRA_DATA_DIR = _DATA_DIR / 'raw' / 'tweets'


def get_extra_tweets(subsample: int) -> List[Path]:
    if subsample < 0:
        return []
    tweet_files = sorted(list(_EXTRA_DATA_DIR.rglob('*.csv.gz')))
    if subsample > 0:
        tweet_files = tweet_files[0: subsample]
    return tweet_files


def hash_df(df: pd.DataFrame) -> str:
    return hashlib.md5(df.values.tobytes()).hexdigest()


def compute_vocab_coverage(df: pd.DataFrame, token_col: str, vocab: Vocab) -> float:
    if df.shape[0] > 10000:
        df = df.sample(10000)
    elif df.shape[0] == 0:
        return float('nan')
    
    def in_vocab(tokens):
        return [t in vocab for t in tokens]

    tokens = df[token_col].apply(in_vocab)
    is_in_vocab = tokens.apply(sum)
    n_tokens = tokens.apply(len)
    return 100 * is_in_vocab.sum() / n_tokens.sum()