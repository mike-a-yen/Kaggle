from pathlib import Path

from exports.exp_00 import RawDataset, ProcessedDataset, TrainableDataset

PROJECT_DIRNAME = Path('./').resolve()

def load_dataset(project_path: Path, subsample: int = 0):
    raw_dataset = RawDataset(project_path, subsample=subsample)
    processed_dataset = ProcessedDataset(raw_dataset)
    trainable_dataset = TrainableDataset(processed_dataset)
    return trainable_dataset

if __name__ == '__main__':
    print(PROJECT_DIRNAME)
    load_dataset(PROJECT_DIRNAME, 100)
    print('Done.')
