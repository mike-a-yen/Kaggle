from functools import partial
import logging
from pathlib import Path
from typing import Iterable, Tuple

from fastprogress import progress_bar
from fastai.basic_data import DataBunch, DatasetType
from fastai.callbacks import EarlyStoppingCallback, SaveModelCallback
from fastai.metrics import accuracy, auc_roc_score, fbeta
from fastai.train import Learner
import numpy as np
import pandas as pd
import pickle
import toml
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from nn_toolkit.logger import Logger
from nn_toolkit.vocab import Vocab
from nn_toolkit.utils import count_trainable_params

from src.data.data_analyzer import DataAnalyzer
from src.data.data_pipeline import MLMDataPipeline, ClasDataPipeline
from src.data.raw_dataset import RawDataset
from src.data.processed_dataset import ProcessedDataset
from src.data.split_dataset import SplitDataset
from src.model.language_model import MaskedLanguageModel
from src.model.clas_model import ClasModel
from src.model.masked_metrics import MaskedCrossEntropy, MaskedAccuracy


_PROJECT_DIR = Path(__file__).parents[1].resolve()
_DATA_DIR = _PROJECT_DIR / 'data'
_MODEL_DIR = _DATA_DIR / 'models'


def f1_score(y_pred, y_true):
    n = y_true.shape[0]
    proba = y_pred.softmax(dim=-1)[:, 1].view(n, 1)
    return fbeta(proba, y_true.view(n, 1), beta=1, thresh=0.5)


def auc_score(y_pred, y_true):
    n = y_true.shape[0]
    proba = y_pred.softmax(dim=-1)[:, 1]
    return auc_roc_score(proba, y_true)


def save_params(params: dict, path: Path) -> None:
    with open(path, 'w') as fo:
        toml.dump(params, fo)


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
        self.clas_name = 'clas_model.pth'
        self.lm_name = 'lm_model.pth'

    @staticmethod
    def load_data(limit: int = None, match_thresh: float = 0.0, frac: float = 0.1) -> SplitDataset:
        raw = RawDataset(limit=limit)
        processed = ProcessedDataset(raw, match_thresh)
        processed.process()
        split_ds = SplitDataset(processed, frac=frac)  # use almost all the data to train
        return split_ds
    
    def set_lm_datapipeline(self, split_ds: SplitDataset) -> None:
        self.lm_pipeline = MLMDataPipeline(split_ds, 'tokens', max_vocab_size=self.model_params['vocab_size'])

    def set_clas_datapipeline(self, split_ds: SplitDataset) -> None:
        self.clas_pipeline = ClasDataPipeline.from_pipeline(self.lm_pipeline)
        self.clas_pipeline.split_ds = split_ds

    def analyze(self, split_ds: SplitDataset) -> None:
        analyzer = DataAnalyzer()
        maxlen = analyzer.maxlen(split_ds, 'tokens')
        self.logger.log_config({'maxlen': maxlen})
        self.logger.log_plot(analyzer.plot_class_balance(split_ds), 'class_balance')
        self.logger.log_plot(analyzer.plot_number_of_tokens(split_ds), 'number_of_tokens')
        self.logger.log_plot(analyzer.length_plot(split_ds, 'tokens'), 'token_length')

    def save_vocab(self) -> None:
        for dir in [self.run_dir, self.logger.dirname]:
            self.lm_pipeline.vocab.to_file(dir / self.vocab_file)

    def build_lm_databunch(self, split_ds: SplitDataset, batch_size: int = None) -> DataBunch:
        bs = batch_size if batch_size is not None else self.training_params['batch_size']
        self.logger.log_config({'batch_size': bs})
        return self.lm_pipeline._build_databunch(
            split_ds.extra_df,
            split_ds.trainval_df,
            split_ds.test_df,
            batch_size=bs
        )

    def build_clas_databunch(self, split_ds: SplitDataset, batch_size: int = None) -> DataBunch:
        bs = batch_size if batch_size is not None else self.training_params['batch_size']
        self.logger.log_config({'batch_size': bs})
        return self.clas_pipeline._build_databunch(
            split_ds.train_df,
            split_ds.val_df,
            split_ds.test_df,
            batch_size=bs
        )

    def set_lm_model(self, **kwargs) -> None:
        self.model_params.update(kwargs)
        self.model_params['vocab_size'] = self.lm_pipeline.vocab.size
        self.model_params['padding_idx'] = self.lm_pipeline.vocab.pad_idx
        self.logger.log_config(self.model_params)
        self.lm_model = MaskedLanguageModel(**self.model_params)

    def set_clas_model(self, *kwargs) -> None:
        self.clas_model = ClasModel.from_language_model(self.lm_model)
        trainable_params = count_trainable_params(self.clas_model)

    def set_lm_learner(self, data: DataBunch) -> None:
        weight = self.lm_model.class_weight.to(data.device)
        loss_fn = MaskedCrossEntropy(weight=weight)
        metrics = [MaskedAccuracy()]
        self.learner = Learner(
            data,
            self.lm_model,
            loss_func=loss_fn,
            opt_func=optim.AdamW,
            metrics=metrics,
            callback_fns=self.get_callbacks(),
            path=self.run_dir,
        )

    def set_clas_learner(self, data: DataBunch) -> None:
        metrics = [accuracy, auc_score]
        self.learner = Learner(
            data,
            self.clas_model, 
            loss_func=nn.CrossEntropyLoss(),
            opt_func=optim.AdamW,
            metrics=metrics,
            callback_fns=self.get_callbacks() + [self.logger.callback_fn()],
            path=self.run_dir,
        )

    def get_callbacks(self) -> None:
        callbacks = [
            partial(EarlyStoppingCallback, patience=self.training_params['early_stopping']),
            partial(SaveModelCallback, every='epoch'),
        ]
        return callbacks

    def save_bits(self) -> None:
        save_params(self.model_params, self.run_dir / 'model_params.toml')
        save_params(self.training_params, self.run_dir / 'training_params.toml')
        self.lm_model.save(self.run_dir / self.lm_name)
        self.clas_model.save(self.run_dir / self.clas_name)
        self.save_vocab()

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
        if self.clas_pipeline.split_ds.target_col in df.columns:
            targets = df[self.clas_pipeline.split_ds.target_col]
            losses = binary_loss(targets, probs[:, 1].numpy())
        return probs, losses

    def make_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        old_data = self.learner.data
        df = df.copy()
        df[self.clas_pipeline.split_ds.target_col] = 0
        data = self.clas_pipeline._build_databunch(df, df, df)
        self.learner.data = data
        probs, *_ = self.learner.get_preds(DatasetType.Test)
        self.learner.data = old_data
        return probs

    def save_data(self, split_ds: SplitDataset) -> None:
        for name, df in split_ds:
            save_to = self.data_dir / f'{name}.json.gz'
            df.to_json(save_to)
            logging.info(f'Saved {name} to {save_to}')
