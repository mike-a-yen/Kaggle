from functools import partial
from pathlib import Path
from typing import List

from fastai.callbacks import EarlyStoppingCallback
from fastai.metrics import accuracy
from fastai.train import Learner
from nn_toolkit.text.model import LanguageModel
import pandas as pd
import toml
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from nn_toolkit.logger import Logger
from src.data.data_pipeline import LanguageModelDataPipeline
from src.data.raw_dataset import RawDataset
from src.data.processed_dataset import ProcessedDataset
from src.data.split_dataset import SplitDataset
from src.trainer import Trainer
from src.utils import get_extra_tweets


_PROJECT_DIR = Path(__file__).parents[1].resolve()
_DATA_DIR = _PROJECT_DIR / 'data'
_MODEL_DIR = _DATA_DIR / 'models'
_HYPERPARAMS_DIR = _PROJECT_DIR / 'src' /'hyperparams'


def load_hyperparams(config_file: str) -> dict:
    with open(_HYPERPARAMS_DIR / config_file) as fo:
        return toml.load(fo)


def find_best_lr(learner) -> float:
    learner.lr_find()
    learner.recorder.plot(suggestion=True)
    return learner.recorder.min_grad_lr


def main(
        hyperparams: str = 'lm_config.toml',
        subsample: int = 0,
        sync: bool = True,
        model_dir: str = str(_MODEL_DIR),
        lr: float = None,
        batch_size: int = 128,
        epochs: int = 50,
) -> None:
    hyperparams = load_hyperparams(hyperparams)
    config = locals()
    logger = Logger('disaster-tweet-lm', sync=sync)
    logger.log_config(config)
    run_dir = Path(model_dir) / wandb.run.id
    run_dir.mkdir(parents=True, exist_ok=True)

    split_ds = Trainer.load_data(subsample=subsample, match_to_real=True)
    print(split_ds)

    data_pipeline = LanguageModelDataPipeline(
        split_ds, 'tokens', hyperparams['max_vocab_size']
    )
    hyperparams['maxlen'] = data_pipeline.maxlen
    hyperparams['padding_idx'] = data_pipeline.vocab.pad_idx
    data_pipeline.display_vocab_coverage()
    extra_data = data_pipeline._build_databunch(
        split_ds.extra_df,
        split_ds.trainval_df,
        bs=batch_size
    )
    data_pipeline.vocab.to_file(run_dir / 'lm_token_store.pkl')
    data_pipeline.vocab.to_file(logger.dirname / 'lm_token_store.pkl')
    logger.log_config(data_pipeline.params)
    logger.log_config(hyperparams)
    print()
    print(hyperparams, end='\n\n')
    model = LanguageModel(**hyperparams)
    callbacks = [
        partial(EarlyStoppingCallback, patience=4),
        logger.callback_fn()
    ]
    weight = torch.ones(model.max_vocab_size).to(extra_data.device)
    weight[data_pipeline.vocab.pad_idx] = 1e-9
    learner = Learner(
        extra_data,
        model,
        loss_func=nn.NLLLoss(weight),
        opt_func=optim.AdamW,
        metrics=[accuracy],
        callback_fns=callbacks,
        path=logger.dirname,
        model_dir=logger.dirname / 'model'
    )
    if lr is None:
        lr = find_best_lr(learner)
    logger.log_config({'lr': lr})
    learner.fit(epochs, lr=lr)
    learner.model.embedding_layer.save(run_dir / 'fwd_embedding.pth')
    learner.model.encoder.save(run_dir / 'fwd_language_model.pth')


if __name__ == '__main__':
    import fire
    fire.Fire(main)
