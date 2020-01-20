from sklearn.model_selection import train_test_split

from processed_dataset import ProcessedDataset


class SplitDataset:
    def __init__(self, processed: ProcessedDataset) -> None:
        self.train_df, self.val_df = [
            df.copy() for df in
            train_test_split(processed.train_df, test_size=0.2)
        ]
        self.test_df = processed.test_df