import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences


class DataGenerator(Sequence):
    def __init__(self, trainable_dataset, batch_size: int = 32, mode='train'):
        self.dataset = trainable_dataset
        self.batch_size = batch_size
        self.training = mode == 'train'
        self.testing = mode == 'test'
        assert mode in('train', 'val', 'test')

    def __len__(self) -> int:
        self.dataset.training = self.training
        self.dataset.testing = self.testing
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

class TrainDataGenerator(Sequence):
    def __init__(self, df, batch_size: int = 32, target_columns: list = [], mode='train'):
        self.df = df
        self.batch_size = batch_size
        self.target_columns = target_columns


    def __len__(self) -> int:
        n_samples = len(self.df)
        return int(np.ceil(n_samples/self.batch_size))

    def __getitem__(self, idx: int):
        batch = self._get_batch(idx)
        X = batch.comment_text.str.lower().values
        Y = batch[self.target_columns].values
        return X, Y

    def _get_batch(self, idx: int):
        offset = self.batch_size * idx
        return self.df.iloc[offset: offset + self.batch_size]