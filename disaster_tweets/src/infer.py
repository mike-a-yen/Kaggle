from pathlib import Path

import tweepy
from nn_toolkit.twitter_tools import TweetCollector, load_credentials, init_tweepy_api, Tweet
from nn_toolkit.twitter_tools.stream import init_tweepy_stream
from nn_toolkit.vocab import Vocab

from tokenizer import ProjectTokenizer
from torch_data import TwitterStreamDataset, batchify_tweet


class Inferer:
    def __init__(self, api: tweepy.API, model: nn.Module, tokenizer, vocab) -> None:
        self.model = model
        self.api = api
        self.stream = init_tweepy_stream(self.api)
        self.stream_ds = TwitterStreamDataset(self.stream, tokenizer, vocab)

    def get_one_tweet(self, track: str) -> Tweet:
        self.stream.start_tracking()
