from fastai.basic_data import DatasetType
from fastai.basic_train import Learner
import pandas as pd

from src.data.evaluated_dataset import EvaluatedDataset


class Evaluator:
    def __init__(self, learner: Learner) -> None:
        self.learner = learner
        self.eval_ds: EvaluatedDataset
        self.roc_auc: float = None
        self.accuracy: float = None
        self.precision: float = None
        self.recall: float = None

    def evaluate(self, min_precision: float = 0.99):
        self.set_predictions(DatasetType.Fix)
        self.set_predictions(DatasetType.Valid)
        self.set_predictions(DatasetType.Test)
        self.eval_ds = EvaluatedDataset(*self.dfs)
        self.eval_ds.find_threshold(min_precision, set_threshold=True)
        self.roc_auc = self.eval_ds.roc_auc_score()['test_df']
        if not self.eval_ds.threshold is None:
            self.accuracy = self.eval_ds.accuracy_score()['test_df']
            self.precision = self.eval_ds.precision_score()['test_df']
            self.recall = self.eval_ds.recall_score()['test_df']

    def set_predictions(self, dstype: DatasetType):
        probas, labels, loss = self.learner.get_preds(dstype, with_loss=True)
        y_proba = probas[:, 1].cpu().numpy()
        y_true = labels.cpu().numpy()
        loss = loss.cpu().numpy()
        df = self.get_df(dstype)
        df['prob_is_positive'] = y_proba
        df['loss'] = loss

    def get_df(self, dstype: DatasetType) -> pd.DataFrame:
        if dstype == DatasetType.Fix:
            return self.learner.data.train_ds.df
        elif dstype == DatasetType.Valid:
            return self.learner.data.valid_ds.df
        elif dstype == DatasetType.Test:
            return self.learner.data.test_ds.df
        raise ValueError('Invalid dataset type')

    def get_results(self) -> dict:
        test_results = {
            'roc_auc': self.roc_auc,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'threshold': self.eval_ds.threshold
        }
        return test_results

    @property
    def dfs(self):
        dstypes = [DatasetType.Fix, DatasetType.Valid, DatasetType.Test]
        return [self.get_df(dstype) for dstype in dstypes]
