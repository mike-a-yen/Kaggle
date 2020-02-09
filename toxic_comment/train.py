import fire
from pathlib import Path
import string

import numpy as np
import tensorflow as tf
import tensorflow.keras.callbacks as callbacks
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


def tf_hack():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    return sess

def model_checkpoint_dirname() -> Path:
    identifier = wandb.run.id
    dirname = PROJECT_DIRNAME / f'_models/{identifier}/checkpoints'
    dirname.mkdir(exist_ok=True, parents=True)
    return dirname

def main(subsample: int = 0) -> None:
    sess = tf_hack()
    wandb.init(project='toxic_comment')
    trainable_dataset = load_dataset(PROJECT_DIRNAME, subsample)
    print(trainable_dataset)

    model_params = {
        'vocab_size': trainable_dataset.vocab_encoder.vocab.size,
        'embedding_size': 16,
        'num_kernels': 256,
        'width': 4,
        'output_size': 128,
        'targets': len(trainable_dataset.toxicity_columns),
        'opt_params': {
            'lr': 0.001
        }
    }
    checkpoint_path = str(model_checkpoint_dirname() / 'epoch-{epoch:02d}-val_loss-{val_loss:.2f}.hdf5')
    training_params = {
        "epochs": 150,
        "batch_size": 64,
        "callbacks": [
            wandb.keras.WandbCallback(),
            callbacks.ModelCheckpoint(
                checkpoint_path
            )
        ]
    }

    batch_size = training_params.pop('batch_size')
    train_dg = DataGenerator(trainable_dataset, batch_size=batch_size, mode='train')
    val_dg = DataGenerator(trainable_dataset, batch_size=batch_size, mode='val')
    model = build_model(model_params, name='model')
    compile_model(model, model_params)
    history = model.fit_generator(
        train_dg,
        validation_data=val_dg,
        **training_params
        )
    
    return


if __name__ == '__main__':
    fire.Fire(main)
    print('Done.')
