from functools import partial
import logging
from pathlib import Path
from typing import Iterable, Tuple

from fastprogress import progress_bar
from fastai.basic_data import DataBunch, DatasetType
from fastai.callbacks import EarlyStoppingCallback, SaveModelCallback
from fastai.metrics import accuracy
from fastai.text import language_model_learner, text_classifier_learner
from fastai.text.data import TextClasDataBunch
from fastai.text.models import AWD_LSTM
from fastai.train import Learner
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from nn_toolkit.logger import Logger
from nn_toolkit.text.model import LanguageModel

from src.data.data_analyzer import DataAnalyzer
from src.data.data_pipeline import LanguageModelDataPipeline
from src.data.raw_dataset import RawDataset
from src.data.processed_dataset import ProcessedDataset
from src.data.split_dataset import SplitDataset
from src.utils import get_extra_tweets


_PROJECT_DIR = Path(__file__).parents[1].resolve()
_DATA_DIR = _PROJECT_DIR / 'data'
_MODEL_DIR = _DATA_DIR / 'models'


def count_trainable_params(model: nn.Module):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


class Trainer:
    def __init__(self, model_params: dict, training_params: dict, id: str = None, sync: bool = False) -> None:
        self.logger = Logger('disaster-tweet-lm', id=id, sync=sync)
        self.model_params = model_params
        self.training_params = training_params
        self.run_dir = _MODEL_DIR / wandb.run.id
        self.data_dir = self.run_dir / 'data'
        self.model_dir = self.run_dir / 'model'

        if not self.run_dir.exists():
            self.run_dir.mkdir(parents=True, exist_ok=True)
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)

        self.vocab_file = 'token_store.pkl'
        self.encoder_name = 'ft_enc'

    def fit(self, **kwargs) -> None:
        self.training_params.update(kwargs)
        self.logger.log_config(self.training_params)
        self.logger.log_config({'trainable_parameters': count_trainable_params(self.learner.model)})
        self.learner.fit(wandb.run.config.epochs, lr=wandb.run.config.lr)
        self.learner.save_encoder(self.encoder_name)

    @staticmethod
    def load_data(subsample: int = 0, match_thresh: float = 0.0, frac: float = 0.1) -> SplitDataset:
        tweet_files = get_extra_tweets(subsample)
        raw = RawDataset(extra_data=tweet_files)
        processed = ProcessedDataset(raw, match_thresh)
        processed.process()
        split_ds = SplitDataset(processed, frac=frac)  # use almost all the data to train
        return split_ds

    def analyze(self, split_ds: SplitDataset) -> None:
        analyzer = DataAnalyzer()
        maxlen = analyzer.maxlen(split_ds, 'tokens')
        self.logger.log_config({'maxlen': maxlen})
        self.logger.log_plot(analyzer.plot_class_balance(split_ds), 'class_balance')
        self.logger.log_plot(analyzer.plot_number_of_tokens(split_ds), 'number_of_tokens')
        self.logger.log_plot(analyzer.length_plot(split_ds, 'tokens'), 'token_length')

    def set_datapipeline(self, split_ds: SplitDataset) -> None:
        self.data_pipeline = LanguageModelDataPipeline(
            split_ds, 'tokens', self.model_params['max_vocab_size']
        )
        self.save_vocab()

    def load_datapipeline(self, split_ds: SplitDataset) -> None:
        self.data_pipeline = LanguageModelDataPipeline(
            split_ds, 'tokens', self.model_params['max_vocab_size']
        )
        self.data_pipeline.load_vocab(self.run_dir / self.vocab_file)

    def save_vocab(self) -> None:
        for dir in [self.run_dir, self.logger.dirname]:
            self.data_pipeline.vocab.to_file(dir / self.vocab_file)

    def build_databunch(
            self,
            train_df: pd.DataFrame,
            valid_df: pd.DataFrame = None,
            test_df: pd.DataFrame = None,
            batch_size: int = None
    ) -> DataBunch:
        bs = batch_size if batch_size is not None else self.training_params['batch_size']
        self.logger.log_config({'batch_size': bs})
        return self.data_pipeline._build_databunch(train_df, valid_df, test_df, bs=bs, path=self.data_dir)

    def build_clas_databunch(
            self,
            train_df: pd.DataFrame,
            valid_df: pd.DataFrame = None,
            test_df: pd.DataFrame = None,
            batch_size: int = None
    ) -> DataBunch:
        bs = batch_size if batch_size is not None else self.training_params['batch_size']
        self.logger.log_config({'batch_size': bs})
        data = TextClasDataBunch.from_df(
            self.data_dir,
            train_df,
            valid_df,
            test_df,
            text_cols=self.data_pipeline.split_ds.text_col,
            label_cols=self.data_pipeline.split_ds.target_col,
            vocab=self.data_pipeline.vocab.to_fastai(),
            tokenizer=self.data_pipeline.split_ds.tokenizer,
            bs=bs,
            include_bos=True,
            include_eos=True,
        )
        return data

    def set_model(self, with_pos_embedding: bool = False, **kwargs) -> None:
        self.model_params.update(kwargs)
        self.model_params['maxlen'] = self.data_pipeline.maxlen if with_pos_embedding else None
        self.model_params['padding_idx'] = self.data_pipeline.vocab.pad_idx
        self.logger.log_config(self.model_params)
        self.model = LanguageModel(**self.model_params)

    def set_lm_learner(self, data: DataBunch) -> None:
        self.learner = language_model_learner(
            data,
            AWD_LSTM,
            drop_mult=self.model_params['dropout_rate'],
            pretrained=False,
            callback_fns=self.get_callbacks(),
            path=self.run_dir,
        )

    def set_clas_learner(self, data: DataBunch) -> None:
        self.learner = text_classifier_learner(
            data,
            AWD_LSTM,
            drop_mult=self.model_params['dropout_rate'],
            callback_fns=self.get_callbacks(),
            path=self.run_dir,
        )
        model_path = self.learner.path / self.learner.model_dir / f'{self.encoder_name}.pth'
        if model_path.exists():
            self.learner.load_encoder(self.encoder_name)
    
    def load_model(self, model_name: str) -> None:
        self.learner.load(model_name)

    def get_callbacks(self) -> None:
        callbacks = [
            partial(EarlyStoppingCallback, patience=self.training_params['early_stopping']),
            partial(SaveModelCallback, every='epoch'),
            self.logger.callback_fn()
        ]

    def suggest_lr(self) -> None:
        self.learner.lr_find()
        fig = self.learner.recorder.plot(suggestion=True, return_fig=True)
        self.logger.log_plot(fig, 'lr_finder')
        return self.learner.recorder.min_grad_lr

    def set_predictions(self, split_ds: SplitDataset, skip: Iterable[str] = []) -> None:
        """Set the model's prediction for each row in each dataframe."""
        for name, df in split_ds:
            if name not in skip:
                self._set_predictions_on_df(df)

    def _set_predictions_on_df(self, df: pd.DataFrame) -> None:
        probs, losses = self.get_sample_predictions(df)
        df['loss'] = losses
        df['prob_is_positive'] = probs[:, 1].numpy()
        df['prob_is_negative'] = probs[:, 0].numpy()
        df['prediction'] = probs.argmax(1).numpy()

    def get_sample_predictions(self, df: pd.DataFrame) -> Tuple:
        def binary_loss(targets, probs):
            return -1 * (targets * np.log(probs) + (1-targets) * np.log(1-probs))

        probs = self.make_predictions(df)
        losses = None
        if self.data_pipeline.split_ds.target_col in df.columns:
            targets = df[self.data_pipeline.split_ds.target_col]
            losses = binary_loss(targets, probs[:, 1].numpy())
        return probs, losses

    def make_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        old_data = self.learner.data
        df = df.copy()
        df[self.data_pipeline.split_ds.target_col] = 0
        data = self.build_clas_databunch(df, df, df)
        self.learner.data = data
        probs, *_ = self.learner.get_preds(DatasetType.Test)
        self.learner.data = old_data
        return probs

    def save_data(self, split_ds: SplitDataset) -> None:
        for name, df in split_ds:
            save_to = self.data_dir / f'{name}.json.gz'
            df.to_json(save_to)
            logging.info(f'Saved {name} to {save_to}')
