import logging

import flask

from app import app, db, twitter_data_stream
from app.db_models import Tweets


def log_tweet_to_db(tweet) -> None:
    try:
        db_tweet = Tweets(tweet.text, None)
        db.session.add(db_tweet)
        db.session.commit()
    except Exception as e:
        logging.info('Failed to log tweet.')
        return False
    return True


@app.route('/')
@app.route('/index')
def index():
    return "Hello"

@app.route('/<name>')
def hello_name(name):
    return f"Hello {name}!"


@app.route('/track/<term>')
def track(term):
    twitter_data_stream.start_tracking([term])
    tweet = twitter_data_stream.get_one()
    code = log_tweet_to_db(tweet)
    logging.info(f'Saved tweet.')
    return tweet.text
