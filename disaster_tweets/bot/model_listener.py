from datetime import datetime
import logging
from pathlib import Path
from typing import Collection

from fastai.core import defaults
from fastai.train import load_learner
from fastai.train import Learner
import torch
import tweepy

from bot.db.utils import log_tweet_to_db, log_prediction_to_db, fetch_model_id


defaults.device = torch.device('cpu')
_DATA_DIR = Path(__file__).parents[1].resolve() / 'data'
_MODEL_DIR = _DATA_DIR / 'models'


def load_model_assets(run_id: str):
    if run_id is None:
        return
    run_dir = _MODEL_DIR / run_id
    learner = load_learner(run_dir, 'clas_learner.pkl', device=torch.device('cpu'))
    return learner


class StreamListener(tweepy.StreamListener):
    def __init__(self, api: tweepy.API, time_limit: int = 300) -> None:
        self.api = api
        self.me = self.api.me()
        self.tweets_seen = 0
        self.ids = []
        self.time_limit = time_limit  # how long to listen for in seconds
        self.start_time = datetime.utcnow()

    def is_valid_tweet(self, tweet) -> bool:
        if tweet.user.id == self.me.id or tweet.in_reply_to_status_id is not None:
            return False
        if tweet.retweeted or tweet.text.startswith('RT @'):
            return False
        return True

    def on_status(self, tweet) -> bool:
        if self.is_valid_tweet(tweet):
            self.tweets_seen += 1
            # tid = log_tweet_to_db(tweet)
            tid = tweet.id
            self.ids.append(tid)
        if self.stop_condition():
            logging.info('Stopping listener.')
            return False
        return True

    def stop_condition(self) -> bool:
        time_streaming = (datetime.utcnow() - self.start_time).total_seconds()
        over_time = time_streaming > self.time_limit
        if over_time:
            self.start_time = datetime.utcnow()
            return True
        return False

    def dump(self) -> Collection[int]:
        return self.ids

    def reset(self) -> None:
        self.start_time = datetime.utcnow()
        self.tweets_seen = 0
        self.ids = []


class DisasterBotListener(tweepy.StreamListener):
    def __init__(self, api: tweepy.API, run_id: str, threshold: float = 0.9, silence: bool = False) -> None:
        self.api = api
        self.me = api.me()
        self.learner = load_model_assets(run_id)
        self.model_id = fetch_model_id(run_id) if run_id is not None else None
        self.threshold = threshold
        self.silence = silence
        self.time_between_response = 180  # how long to wait between responses in seconds
        self.time_limit = 300  # how long to listen for in seconds
        self.tweets_seen = 0
        self.responded = 0
        self.start_time = datetime.utcnow()
        self.last_response_time = datetime(1900, 1, 1)

        self.say_hello()

    def say_hello(self) -> None:
        if self.silence:
            return
        display_time = self.start_time.strftime("%a %B %H:%M")
        message = f"Hello, my watch is starting now at {display_time}"
        self.api.update_status(
            message
        )

    def say_goodbye(self) -> None:
        if self.silence:
            return
        display_time = datetime.utcnow().strftime("%a %B %H:%M")
        message = f"Goodbye, my watch has ended at {display_time}"
        self.api.update_status(
            message
        )

    def check_time_limit(self) -> bool:
        if self.time_between_response is None:
            return True
        now = datetime.utcnow()
        diff = now - self.last_response_time
        since_last = diff.total_seconds()
        return since_last > self.time_between_response

    def respond(self, tweet):
        if self.silence:
            return
        try:
            self.favorite(tweet)
            # self.reply(tweet)
            # self.retweet(tweet)
            self.responded += 1
        except:
            pass

    def favorite(self, tweet) -> None:
        self.api.create_favorite(tweet.id)

    def reply(self, tweet) -> None:
        tweet_id = tweet.id
        self.api.update_status(
            status = 'I hope it is all ok over there.',
            in_reply_to_status_id = tweet_id ,
            auto_populate_reply_metadata=True
        )

    def retweet(self, tweet) -> None:
        tweet.retweet()
    
    def get_prediction(self, tweet):
        text = tweet._json['text']
        _, _, probs = self.learner.predict(text)
        prob_is_positive = float(probs.cpu().numpy()[1])
        is_positive = prob_is_positive > self.threshold
        return prob_is_positive, is_positive

    def display_tweet(self, tweet) -> None:
        logging.info(tweet.text)

    def on_status(self, tweet) -> bool:
        if self.is_valid_tweet(tweet):
            self.tweets_seen += 1
            tid = log_tweet_to_db(tweet)
            if self.learner is not None:
                prob, is_positive = self.get_prediction(tweet)
                pid = log_prediction_to_db(tid, self.model_id, int(is_positive), prob)
                if is_positive and self.check_time_limit():
                    self.display_tweet(tweet)
                    self.respond(tweet)
                    self.last_response_time = datetime.utcnow()
        if self.stop_condition():
            logging.info('Stopping.')
            return False
        return True

    def is_valid_tweet(self, tweet) -> bool:
        if tweet.user.id == self.me.id or tweet.in_reply_to_status_id is not None:
            return False
        if tweet.retweeted or tweet.text.startswith('RT @'):
            return False
        return True
    
    def stop_condition(self) -> bool:
        time_streaming = (datetime.utcnow() - self.start_time).total_seconds()
        over_time = time_streaming > self.time_limit
        if over_time:
            self.start_time = datetime.utcnow()
            return True
        return False
        