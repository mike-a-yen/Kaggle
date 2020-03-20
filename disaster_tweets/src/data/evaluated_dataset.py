from functools import partial
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score
)


class EvaluatedDataset:
    def __init__(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame = None,
        target_col: str = 'target',
        prediction_col: str = 'prob_is_positive'
    ) -> None:
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.target_col = target_col
        self.prediction_col = prediction_col

    def _compute_proba_metric(self, df: pd.DataFrame, metric_fn: Callable) -> float:
        y_true, y_proba = self.get_labels_and_predictions(df)
        return metric_fn(y_true, y_proba)

    def _compute_metric(self, df: pd.DataFrame, metric_fn: Callable, threshold: float = None) -> float:
        y_true, y_proba = self.get_labels_and_predictions(df)
        if threshold is None:
            threshold = self.threshold
        y_pred = (y_proba > threshold).astype(int)
        return metric_fn(y_true, y_pred)

    def roc_auc_score(self) -> dict:
        return {name: self._compute_proba_metric(df, roc_auc_score) for name, df in self}

    def accuracy_score(self) -> dict:
        return {name: self._compute_metric(df, accuracy_score) for name, df in self}

    def precision_score(self) -> dict:
        return {name: self._compute_metric(df, precision_score) for name, df in self}

    def recall_score(self) -> dict:
        return {name: self._compute_metric(df, recall_score) for name, df in self}

    def get_labels_and_predictions(self, df: pd.DataFrame) -> Tuple[np.ndarray]:
        return np.array(df[self.target_col].values), np.array(df[self.prediction_col].values)

    def get_thresholds(self, df: pd.DataFrame, decimals: int = 4) -> np.ndarray:
        thresh = df[self.prediction_col].apply(partial(np.round, decimals=decimals)).unique()
        thresh.sort()
        return thresh

    def find_threshold(self, min_precision: float, set_threshold: bool = True) -> float:
        thresholds = self.get_thresholds(self.valid_df)
        max_recall = 0
        best_thresh = 1.
        for thresh in thresholds:
            y_true, y_proba = self.get_labels_and_predictions(self.valid_df)
            y_pred = (y_proba > thresh).astype(int)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            if precision > min_precision:
                if recall > max_recall:
                    best_thresh = thresh
                    max_recall = recall
        if set_threshold:
            self.threshold = best_thresh
        return best_thresh

    def __iter__(self):
        for name in ['train_df', 'valid_df', 'test_df']:
            df = getattr(self, name, None)
            if df is None:
                continue
            yield name, df
