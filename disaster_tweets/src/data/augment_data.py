from typing import Tuple

import numpy as np
import pandas as pd
import sklearn.base as base
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.tokenizer import ProjectTokenizer


class DataMatcher:
    """Find samples from the augmented data that look like the real data.
    
    This specifically trains a simple tfidf classifier to separate augmented from real
    data. Predictions are then made on all of the augmented data and only the highest 
    probability samples are kept for training. The rest look too different from the real
    data to be helpful in training.
    """

    def __init__(self, real_df: pd.DataFrame) -> None:
        self.text_col = 'text'
        self.token_col = 'raw_text_tokens'
        self.real_df = real_df
        self.n_real = self.real_df.shape[0]
        self.pipeline : Pipeline
        self.clf : base.BaseEstimator
        self.column_transformer : ColumnTransformer
        self.vectorizer : TfidfVectorizer

    def looks_real(self, df: pd.DataFrame, thresh: float = 0.2) -> pd.DataFrame:
        """Select the samples from `df` that look like the real data."""
        keep_prob = self._estimate_is_real_prob(df)
        keep_prob *= self._estimate_heuristics(df)
        condition = keep_prob > thresh
        return df[condition].copy()

    def _estimate_heuristics(self, df: pd.DataFrame) -> np.ndarray:
        """Return binary heuristic based guesses."""
        long_chars = df[self.text_col].apply(len) > 160
        long_tokens = df[self.token_col].apply(len) > 64
        too_long = (long_chars | long_tokens)
        return (~too_long).values.astype(np.float32)

    def build_estimator(self) -> Pipeline:
        tokenizer = ProjectTokenizer()
        self.vectorizer = TfidfVectorizer(min_df=5, stop_words='english', tokenizer=tokenizer)
        self.column_transformer = ColumnTransformer(
            [
                ('tfidf', self.vectorizer, self.text_col),
                ('n_chars', LengthTransformer(scale=True), self.text_col),
                ('n_tokens', LengthTransformer(scale=True), self.token_col)
            ],
            n_jobs=-1
        )
        self.clf = SGDClassifier(loss='log', n_jobs=-1)  # logisitic regression
        pipeline = Pipeline([
            ('featurizer', self.column_transformer),
            ('clf', self.clf)
        ])
        return pipeline

    def _estimate_is_real_prob(self, aug_df: pd.DataFrame) -> np.ndarray:
        X_train, y_train = self._create_training_data(aug_df)
        self.pipeline = self.build_estimator()
        self.pipeline.fit(X_train, y_train)
        X_aug = self._featurize(aug_df)
        return self.pipeline.predict_proba(X_aug)[:, 1]

    def _create_training_data(self, aug_df: pd.DataFrame) -> Tuple[np.ndarray]:
        heuristic_match = (self._estimate_heuristics(aug_df) == 1.)
        fake_df = aug_df[heuristic_match].sample(self.n_real, replace=True)
        X_real = self._featurize(self.real_df)
        X_fake = self._featurize(fake_df)
        y_real = np.ones(X_real.shape[0])
        y_fake = np.zeros(X_fake.shape[0])
        X_train = pd.concat([X_real, X_fake], sort=False)
        y_train = np.hstack([y_real, y_fake])
        return X_train, y_train

    def _featurize(self, df: pd.DataFrame) -> np.ndarray:
        return df[[self.text_col, self.token_col]]


class LengthTransformer(base.BaseEstimator):
    """Get the length of each element."""
    def __init__(self, scale: bool = True) -> None:
        self.scale = scale
        self.scaler = StandardScaler() if self.scale else None

    def transform(self, X, y=None) -> np.ndarray:
        X = X.apply(len).values[:, np.newaxis]
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return X

    def fit(self, X, y=None):
        X = X.apply(len).values[:, np.newaxis]
        if self.scaler is not None:
            self.scaler.fit(X)
        return self
