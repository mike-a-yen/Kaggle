from typing import Tuple

import pandas as pd

from nn_toolkit.vocab import Vocab


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