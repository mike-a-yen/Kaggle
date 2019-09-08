import fire
from pathlib import Path
import string

from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

from exports.exp_00 import RawDataset, ProcessedDataset, TrainableDataset
from exports.vocab import Vocabulary, VocabEncoder

PROJECT_DIRNAME = Path('./').resolve()


def load_dataset(project_path: Path, subsample: int = 0):
    raw_dataset = RawDataset(project_path, subsample=subsample)
    processed_dataset = ProcessedDataset(raw_dataset)
    trainable_dataset = TrainableDataset(processed_dataset)
    return trainable_dataset

class DataGenerator(Sequence):
    def __init__(self, df,  transform=None, batch_size: int = 32):
        self.df = df
        self.transform = transform
        self.batch_size = batch_size
        self.target = 'toxic'

    def __len__(self) -> int:
        return int(np.ceil(self.df.shape[0]/self.batch_size))

    def __getitem__(self, idx: int):
        batch_df = self._get_batch(idx)
        X = batch_df.comment_words
        X = self.transform(X)
        Y = batch_df[self.target]
        return X, Y
def main(subsample: int = 0) -> None:
    trainable_dataset = load_dataset(PROJECT_DIRNAME, subsample)
    vocab = Vocabulary(list(string.printable))
    token_encoder = VocabEncoder(vocab)
    
    X_col = 'comment_words'
    X_train = trainable_dataset.train_df[X_col].apply(token_encoder.encode)
    print(X_train)
    return


if __name__ == '__main__':
    fire.Fire(main)
    print('Done.')
