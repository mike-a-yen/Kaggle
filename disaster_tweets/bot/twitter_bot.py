from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from fastai.core import defaults
from fastai.train import Learner, load_learner
from fastai.text.data import TextClasDataBunch
import pandas as pd
import torch
import tweepy

from bot.model_listener import StreamListener
from bot.db.utils import fetch_tweets_without_prediction
from src.tokenizer import ProjectTokenizer


defaults.device = torch.device('cpu')
_DATA_DIR = Path(__file__).parents[1].resolve() / 'data'
_MODEL_DIR = _DATA_DIR / 'models'


def get_learner_from_run_id(run_id: str) -> Learner:
    run_dir = _MODEL_DIR / run_id
    return load_learner(run_dir, 'clas_learner.pkl', device=torch.device('cpu'))


class TwitterBot:
    def __init__(self, api: tweepy.API, run_id: str, threshold: float = 0.75, silence: bool = True) -> None:
        self.api = api
        self.run_id = run_id
        self.learner = get_learner_from_run_id(self.run_id)
        self.listener = StreamListener(self.api)
        self.stream = tweepy.Stream(self.api.auth, self.listener)
        self.tokenizer = ProjectTokenizer()

    def begin_listening(self, terms: List[str]) -> None:
        self.stream.filter(track=terms, languages=['en'], is_async=True)

    def disconnect(self) -> None:
        self.stream.disconnect()

    def say_hello(self) -> None:
        if self.silence:
            return
        display_time = datetime.now().strftime("%a %B %H:%M")
        message = f"Hello, my watch is starting now at {display_time}"
        self.api.update_status(message)

    def say_goodbye(self) -> None:
        if self.silence:
            return
        display_time = datetime.now().strftime("%a %B %H:%M")
        message = f"Goodbye, my watch has ended at {display_time}"
        self.api.update_status(message)
    
    def get_latest_tweets(self) -> pd.DataFrame:
        since_time = self.listener.start_time - timedelta(hours=8)
        since = since_time.strftime('%Y-%m-%d')
        return fetch_tweets_without_prediction(self.run_id, since, limit=1000)

    def make_databunch(self, df: pd.DataFrame) -> TextClasDataBunch:
        data = TextClasDataBunch.from_df(
            './',
            train_df=df,
            valid_df=df,
            text_cols='text',
            tokenizer=self.tokenizer,
            vocab=self.learner.data.vocab,
            device=self.learner.data.device
        )
        return data

    def make_predictions(self, df: pd.DataFrame) -> None:
        data = self.make_databunch(df)
        with torch.no_grad():
            probas = []
            for xb, _ in data.valid_dl:
                y, *_ = self.learner.model(xb)
                proba = y.softmax(-1)
                probas.append(proba)
            probas = torch.cat(probas, dim=0)
        return probas
