import fire
from pathlib import Path
import string

import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import wandb

from exports.exp_00 import RawDataset, ProcessedDataset
from trainable_dataset import TrainableDataset
from data_generator import DataGenerator
from model import build_model, compile_model
from exports.vocab import Vocabulary, VocabEncoder

PROJECT_DIRNAME = Path('./').resolve()


def load_dataset(project_path: Path, subsample: int = 0):
    raw_dataset = RawDataset(project_path, subsample=subsample)
    processed_dataset = ProcessedDataset(raw_dataset)
    trainable_dataset = TrainableDataset(processed_dataset)
    return trainable_dataset


def main(subsample: int = 0) -> None:
    wandb.init(project='toxic_comment')
    trainable_dataset = load_dataset(PROJECT_DIRNAME, subsample)
    train_dg = DataGenerator(trainable_dataset, batch_size=32, mode='train')
    val_dg = DataGenerator(trainable_dataset, batch_size=32, mode='val')
    model_params = {
        'vocab_size': trainable_dataset.vocab_encoder.vocab.size,
        'embedding_size': 16,
        'output_size': 64
    }
    training_params = {
        "epochs": 150,
        "callbacks": [wandb.keras.WandbCallback()]
    }

    model = build_model(model_params, name='model')
    compile_model(model)
    history = model.fit_generator(
        train_dg,
        validation_data=val_dg,
        **training_params
        )
    
    return


if __name__ == '__main__':
    fire.Fire(main)
    print('Done.')
