from pathlib import Path
from typing import List, Tuple

from fastai.basic_data import DataBunch
from fastai.text.data import TextLMDataBunch
import pandas as pd
import torch
import torch.utils as utils

from nn_toolkit.vocab import Vocab, VocabBuilder

from src.data.data_sampler import BucketedSampler
from src.data.split_dataset import SplitDataset
from src.data.offline_data import TweetLMDataset, TweetClasDataset
from src.tokenizer import ProjectTokenizer


_PROJECT_DIR = Path(__file__).parents[2].resolve()
_DATA_DIR = _PROJECT_DIR / 'data'
_MODEL_DIR = _DATA_DIR / 'models'


def compute_maxlen(batch: List):
    maxlen = 0
    for sample in batch:
        length = sample.size(0)
        maxlen = max(maxlen, length)
    return maxlen


def compute_vocab_coverage(df, token_col, vocab) -> float:
    if df.shape[0] > 10000:
        df = df.sample(10000)
    elif df.shape[0] == 0:
        return float('nan')
    tokens = df[token_col].apply(lambda x: [t in vocab for t in x])
    is_in_vocab = tokens.apply(sum)
    n_tokens = tokens.apply(len)
    return 100 * is_in_vocab.sum() / n_tokens.sum()


def pad_sequence(seq: torch.LongTensor, maxlen: int, pad_val: int) -> torch.LongTensor:
    tokens = torch.zeros(maxlen, dtype=torch.int64)
    seqlen = min(seq.size(0), maxlen)
    tokens[-seqlen:] = seq[0: seqlen]  # trim long tweets
    tokens[0: maxlen-seqlen] = pad_val
    return tokens


def build_vocab(split_ds: SplitDataset, token_col: str, **kwargs) -> Vocab:
    df = pd.concat([split_ds.trainval_df, split_ds.test_df, split_ds.extra_df], sort=False)
    vocab_builder = VocabBuilder(**kwargs)
    vocab = vocab_builder.from_df(df, token_col)
    return vocab


class DataPipeline:
    def __init__(self, split_ds: SplitDataset, token_col: str, max_vocab_size: int = 100000) -> None:
        self.split_ds = split_ds
        self.token_col = token_col
        self.max_vocab_size = max_vocab_size
        self.maxlen = max([df[self.token_col].apply(len).max() for _, df in self.split_ds]) + 2

    def _build_databunch(self, train_df: pd.DataFrame, valid_df: pd.DataFrame = None, test_df: pd.DataFrame = None, **kwargs) -> DataBunch:
        data = DataBunch.create(
            self._get_ds(train_df),
            valid_ds=self._get_ds(valid_df) if valid_df is not None else None,
            test_ds=self._get_ds(test_df) if test_df is not None else None,
            collate_fn=self.collate_batch,
            device=torch.device('cuda'),
            num_workers=4,
            **kwargs
        )
        return data

    def build_databunch(self, **kwargs) -> DataBunch:
        return self._build_databunch(
            self.split_ds.train_df, self.split_ds.val_df, self.split_ds.test_df, **kwargs
        )

    def build_extra_databunch(self, **kwargs) -> DataBunch:
        return self._build_databunch(self.split_ds.extra_df, self.split_ds.val_df, test_df=None, **kwargs)

    def _get_ds(self, df: pd.DataFrame) -> utils.data.Dataset:
        raise NotImplementedError()

    def _get_train_ds(self) -> utils.data.Dataset:
        return self._get_ds(self.split_ds.train_df)

    def _get_valid_ds(self) -> utils.data.Dataset:
        return self._get_ds(self.split_ds.val_df)

    def _get_test_ds(self) -> utils.data.Dataset:
        return self._get_ds(self.split_ds.test_df)

    def _get_extra_ds(self) -> utils.data.Dataset:
        return self._get_ds(self.split_ds.extra_df)

    def collate_batch(self, batch: List):
        raise NotImplementedError()

    def transform_fn(self, sample):
        raise NotImplementedError()

    def _set_vocab(self) -> None:
        self._vocab = build_vocab(
            self.split_ds,
            self.token_col,
            max_size=self.max_vocab_size,
            min_count=10
        )

    def load_vocab(self, file: Path) -> None:
        self._vocab = Vocab.from_file(file)

    @property
    def vocab(self):
        if getattr(self, '_vocab', None) is None:
            self._set_vocab()
        return self._vocab

    def display_vocab_coverage(self) -> None:
        train_coverage = compute_vocab_coverage(self.split_ds.train_df, self.token_col, self.vocab)
        valid_coverage = compute_vocab_coverage(self.split_ds.val_df, self.token_col, self.vocab)
        test_coverage = compute_vocab_coverage(self.split_ds.test_df, self.token_col, self.vocab)
        extra_coverage = compute_vocab_coverage(self.split_ds.extra_df, self.token_col, self.vocab)
        print(f'Vocab size: {self.vocab.size}')
        print('Coverage:' 
            f'\n\t train {train_coverage:0.2f}%'
            f'\n\t valid {valid_coverage:0.2f}%' 
            f'\n\t test {test_coverage:0.2f}%'
            f'\n\t extra {extra_coverage:0.2f}%'
        )

    @property
    def params(self) -> dict:
        payload = {
            'train_samples': self.split_ds.train_df.shape[0],
            'val_samples': self.split_ds.val_df.shape[0],
            'test_samples': self.split_ds.test_df.shape[0],
            'extra_samples': self.split_ds.extra_df.shape[0],
            'vocab_size': self.vocab.size,
            'maxlen': self.maxlen
        }
        dfs = ['train_df', 'val_df', 'test_df', 'extra_df']
        for name in dfs:
            df = getattr(self.split_ds, name)
            payload[f'{name}_tokens'] = df[self.token_col].apply(len).sum()
        return payload


class LanguageModelDataPipeline(DataPipeline):
    def _build_databunch(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None,
        **kwargs
    ) -> TextLMDataBunch:
        data = TextLMDataBunch.from_df(
                train_df=train_df,
                valid_df=valid_df,
                test_df=test_df,
                text_cols=self.split_ds.text_col,
                vocab=self.vocab.to_fastai(),
                tokenizer=self.split_ds.tokenizer,
                bptt=self.maxlen,
                include_bos=True,
                include_eos=True,
                **kwargs
        )
        return data

    def _get_ds(self, df: pd.DataFrame) -> utils.data.Dataset:
        return TweetLMDataset(df, self.transform_fn)

    def collate_batch(self, batch: List) -> Tuple:
        B = len(batch)
        # maxlen = compute_maxlen(batch)
        text = torch.stack([
            pad_sequence(sample, self.maxlen, self.vocab.pad_idx) for sample in batch
        ])
        xb = text[:, 0: -1]
        yb = text[:, 1:].contiguous()
        return xb, yb.view(-1)

    def transform_fn(self, sample) -> torch.LongTensor:
        tokens = sample[self.token_col]
        tokens = self.vocab.map_to_ints(tokens)
        return torch.LongTensor(tokens)


class ClasDataPipeline(DataPipeline):
    def _get_ds(self, df: pd.DataFrame) -> utils.data.Dataset:
        return TweetClasDataset(df, self.transform_fn)

    def collate_batch(self, batch: List) -> Tuple:
        maxlen = min(compute_maxlen([sample[0] for sample in batch]), self.maxlen)
        xb = torch.stack([
            pad_sequence(sample[0], maxlen, self.vocab.pad_idx) for sample in batch
        ])
        yb = torch.stack([sample[1] for sample in batch]).float()
        return xb, yb

    def transform_fn(self, sample: pd.Series) -> torch.LongTensor:
        tokens = sample[self.token_col]
        tokens = self.vocab.map_to_ints(tokens)
        return torch.LongTensor(tokens)
    
    def _set_vocab(self) -> None:
        vocab_file = Path(_MODEL_DIR / 'lm_token_store.pkl')
        self._vocab = Vocab.from_file(vocab_file)
