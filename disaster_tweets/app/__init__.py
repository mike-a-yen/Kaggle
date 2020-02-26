import os

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import redis
from rq import Queue


from nn_toolkit.twitter_tools import load_credentials, init_tweepy_api
from nn_toolkit.twitter_tools.stream import init_tweepy_stream

from src.data.online_data import TwitterDataStream


app = Flask(__name__)
app.config.from_object(os.environ['APP_SETTINGS'])
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
redis_url = os.environ.get('REDISTOGO_URL', 'redis://localhost:6379')
conn = redis.from_url(redis_url)

twitter_api = init_tweepy_api(load_credentials())
twitter_stream = init_tweepy_stream(twitter_api)
twitter_data_stream = TwitterDataStream(twitter_stream, filter_retweets=True)

redis_queue = Queue(connection=conn)

from app import views
from app.db_models import Tweets
