import fire
from pathlib import Path
import string

import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

from exports.exp_00 import RawDataset, ProcessedDataset
from trainable_dataset import TrainableDataset
from exports.vocab import Vocabulary, VocabEncoder

PROJECT_DIRNAME = Path('./').resolve()


def load_dataset(project_path: Path, subsample: int = 0):
    raw_dataset = RawDataset(project_path, subsample=subsample)
    processed_dataset = ProcessedDataset(raw_dataset)
    trainable_dataset = TrainableDataset(processed_dataset)
    return trainable_dataset


class DataGenerator(Sequence):
    def __init__(self, trainable_dataset, batch_size: int = 32, mode='train'):
        self.dataset = trainable_dataset
        self.batch_size = batch_size
        self.training = mode == 'train'
        self.testing = mode == 'test'
        assert mode in('train', 'val', 'test')

    def __len__(self) -> int:
        n_samples = len(self.dataset)
        return int(np.ceil(n_samples/self.batch_size))

    def __getitem__(self, idx: int):
        batch = self._get_batch(idx)
        X = [item[0] for item in batch]
        maxlen = max(map(len, X))
        X = pad_sequences(X, maxlen)
        if not self.dataset.testing:
            Y = [np.expand_dims(item[1], 0) for item in batch]
            Y = np.concatenate(Y, 0)
            return X, Y
        return X

    def _get_batch(self, idx: int):
        offset = self.batch_size * idx
        self.dataset.training = self.training
        self.dataset.testing = self.testing
        end = min(len(self.dataset), offset+self.batch_size)
        items = [self.dataset[i] for i in range(offset, end)]
        return items


def main(subsample: int = 0) -> None:
    trainable_dataset = load_dataset(PROJECT_DIRNAME, subsample)
    train_dg = DataGenerator(trainable_dataset, batch_size=32, mode='train')
    val_dg = DataGenerator(trainable_dataset, batch_size=32, mode='val')
    X, Y = train_dg[0]
    print(X.shape)
    print(Y.shape)
    X, Y = val_dg[0]
    print(X.shape)
    print(Y.shape)
    return


if __name__ == '__main__':
    fire.Fire(main)
    print('Done.')
