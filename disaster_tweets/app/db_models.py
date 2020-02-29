from datetime import datetime
from sqlalchemy.dialects.postgresql import JSON
from nn_toolkit.twitter_tools import Tweet

from app import db


class Tweets(db.Model):
    __tablename__ = 'tweets'

    id = db.Column(db.Integer, primary_key=True)
    tweet_id = db.Column(db.Float)
    text = db.Column(db.String())
    user_id = db.Column(db.String())
    user_name = db.Column(db.String())
    screen_name = db.Column(db.String())
    retweeted = db.Column(db.Boolean())
    lang = db.Column(db.String(8), default=None)
    lat = db.Column(db.Float, default=None)
    long = db.Column(db.Float, default=None)
    prediction = db.Column(db.Float, default=None)
    label = db.Column(db.Integer, default=None)

    created_at = db.Column(db.DateTime, default=datetime.utcnow())
    updated_at = db.Column(db.DateTime, default=datetime.utcnow())

    def __init__(self, tweet:Tweet, prediction: float = None, label: int = None) -> None:
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
        self.prediction = prediction
        self.label = label

    def __repr__(self):
        return f'<id {self.id}: {self.text[0:32]}>'
