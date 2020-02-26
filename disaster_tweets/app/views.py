import logging
import time

import flask
import pandas as pd
from rq.job import Job


from app import app, conn, db, twitter_data_stream, redis_queue
from app.db_models import Tweets


def log_tweet_to_db(tweet):
    try:
        db_tweet = Tweets(tweet, None)
        db.session.add(db_tweet)
        db.session.commit()
    except Exception as e:
        return None, False
    return db_tweet.id, True


def get_one_tweet(term: str):
    twitter_data_stream.start_tracking([term])
    tweet = twitter_data_stream.get_one()
    return log_tweet_to_db(tweet)


@app.route('/')
@app.route('/index')
def index():
    return "Hello"

@app.route('/<name>')
def hello_name(name):
    return f"Hello {name}!"

@app.route('/db')
def db_summary():
    df = pd.read_sql(
        """
        SELECT * from tweets;
        """,
        con=db.session.connection()
    )
    return df.describe().to_html()

@app.route('/track/<term>', methods=['GET', 'POST'])
def track(term):
    job = redis_queue.enqueue_call(
            func=get_one_tweet, args=(term,), result_ttl=60
        )
    return flask.redirect(flask.url_for('results', jobkey=job.get_id()))


@app.route('/results/<jobkey>', methods=['GET'])
def results(jobkey):
    job = Job.fetch(jobkey, connection=conn)
    if job.is_finished:
        id, success = job.result
        if not success:
            return flask.redirect(flask.url_for('track', term=job.args[0]))
        query = Tweets.query.filter_by(id=id).first()
        return str(query.text), 200
    else:
        time.sleep(0.5)
        return flask.redirect(flask.url_for('results', jobkey=job.get_id()))
