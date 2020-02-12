import os

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from nn_toolkit.twitter_tools import load_credentials, init_tweepy_api
from nn_toolkit.twitter_tools.stream import init_tweepy_stream

from src.data.online_data import TwitterDataStream


app = Flask(__name__)
app.config.from_object(os.environ['APP_SETTINGS'])
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

twitter_api = init_tweepy_api(load_credentials())
twitter_stream = init_tweepy_stream(twitter_api)
twitter_data_stream = TwitterDataStream(twitter_stream, filter_retweets=True)


from app import views
from app.db_models import Tweets
