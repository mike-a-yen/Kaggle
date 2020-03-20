from datetime import datetime
from nn_toolkit.twitter_tools import Tweet
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String

from bot.db.base import Base


class Tweets(Base):
    __tablename__ = 'tweets'

    id = Column(Integer, primary_key=True)
    tweet_id = Column(Float)
    text = Column(String)
    user_id = Column(String)
    user_name = Column(String)
    screen_name = Column(String)
    retweeted = Column(Boolean)
    lang = Column(String, default=None)
    lat = Column(Float, default=None)
    long = Column(Float, default=None)

    created_at = Column(DateTime, default=datetime.utcnow())
    updated_at = Column(DateTime, default=datetime.utcnow())

    def __init__(self, tweet: Tweet) -> None:
        self.tweet_id = tweet.id
        self.text = tweet.text
        self.user_id = tweet.user_id
        self.user_name = tweet.user_name
        self.screen_name = tweet.user_screen_name
        self.retweeted = tweet.retweeted
        self.lang = tweet.lang
        self.lat = tweet.lat
        self.long = tweet.long
        self.created_at = tweet.created_at

    def __repr__(self):
        return f'<id {self.id}: {self.text[0:32]}>'


class TweetPredictions(Base):
    __tablename__ = 'tweet_predictions'

    id = Column(Integer, primary_key=True)
    tweet_id = Column(ForeignKey('tweets.id'))
    model_id = Column(ForeignKey('models.id'))
    pred_label = Column(Integer)
    proba = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow())
    updated_at = Column(DateTime, default=datetime.utcnow())

    def __init__(self, tweet_id: int, model_id: int, pred_label: int, proba: float) -> None:
        self.tweet_id = tweet_id
        self.model_id = model_id
        self.pred_label = pred_label
        self.proba = proba


class Annotations(Base):
    __tablename__ = 'annotations'

    id = Column(Integer, primary_key=True)
    tweet_id = Column(ForeignKey('tweets.id'))
    label = Column(Integer)

    created_at = Column(DateTime, default=datetime.utcnow())
    updated_at = Column(DateTime, default=datetime.utcnow())

    def __init__(self, tweet_id: int, label: int) -> None:
        self.tweet_id = tweet_id
        self.label = label


class Models(Base):
    __tablename__ = 'models'

    id = Column(Integer, primary_key=True)
    run_id = Column(String)
    run_dir = Column(String)

    created_at = Column(DateTime, default=datetime.utcnow())
    updated_at = Column(DateTime, default=datetime.utcnow())

    def __init__(self, run_id: str, run_dir: str) -> None:
        self.run_id = run_id
        self.run_dir = run_dir


class ModelEvaluations(Base):
    __tablename__ = 'model_evaluations'

    id = Column(Integer, primary_key=True)
    model_id = Column(ForeignKey('models.id'))
    threshold = Column(Float)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    roc_auc = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow())
    updated_at = Column(DateTime, default=datetime.utcnow())

    def __init__(self, model_id: int, threshold: float, accuracy: float, precision: float, recall: float, roc_auc: float) -> None:
        self.model_id = model_id
        self.threshold = threshold
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.roc_auc = roc_auc
