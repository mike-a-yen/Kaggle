import time
from typing import List

import torch
import torch.utils as utils
import tweepy

from src.tokenizer import ProjectTokenizer
from nn_toolkit.twitter_tools import Tweet
from nn_toolkit.vocab import Vocab


def batchify_tweet(tweet: Tweet) -> torch.LongTensor:
    """Convert tweet to a model digestable input."""
    token_ints = tweet.token_ints
    token_ints = [0] + token_ints + [0]
    return torch.LongTensor([token_ints])


class TwitterDataStream:
    def __init__(self, stream, filter_retweets: bool = True) -> None:
        self.stream = stream
        self.tracking = False
        self.filter_retweets = filter_retweets
        self.last_id = None
    
    def get(self, num: int) -> List[Tweet]:
        count = 0
        for tweet in next(self):
            yield tweet
            count += 1
            if count == num:
                break
        self.stop_tracking()

    def get_one(self) -> Tweet:
        tweet = next(self)
        self.stop_tracking()
        return tweet

    def start_tracking(self, track: List[str]) -> None:
        try:
            self.stream.filter(track=track, languages=['en'], is_async=True)
            self.tracking = True
        except tweepy.TweepError:
            self.stream.disconnect()
            return self.start_tracking(track)
    
    def stop_tracking(self) -> None:
        self.stream.disconnect()
        self.tracking = False
    
    def __next__(self) -> Tweet:
        assert self.tracking
        while True:
            latest_tweet = self.stream.listener.latest_tweet
            if latest_tweet is not None and self.last_id != latest_tweet.id:
                self.last_id = latest_tweet.id
                if self.filter_retweets and latest_tweet.retweeted:
                    continue
                return latest_tweet
            else:
                time.sleep(0.1)


class TwitterStreamDataset(utils.data.IterableDataset):
    """Listen to a twitter stream and yield each tweet as a sample.
    
    This dataset yields a `Tweet` object which has been given `tokens` and `token_ints`
    as attributes.
    """

    def __init__(self, stream, tokenizer:ProjectTokenizer, vocab: Vocab, filter_retweets: bool = True) -> None:
        self.stream = stream
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.filter_retweets = filter_retweets
        self.tracking = False
        self.length = 10

    def start_tracking(self, track: List[str]) -> None:
        try:
            self.stream.filter(track=track, languages=['en'], is_async=True)
            self.tracking = True
        except tweepy.TweepError:
            self.stream.disconnect()
            return self.start_tracking(track)

    def __iter__(self):
        assert self.tracking
        self.last_id = None
        for _ in range(self.length):
            yield next(self)
        self.stop_tracking()

    def __next__(self):
        assert self.tracking
        while True:
            latest_tweet = self.stream.listener.latest_tweet
            if latest_tweet is not None and self.last_id != latest_tweet.id:
                self.last_id = latest_tweet.id
                if self.filter_retweets and latest_tweet.retweeted:
                    continue
                latest_tweet.tokens = self.tokenizer(latest_tweet.text)
                latest_tweet.token_ints = self.vocab.map_to_ints(latest_tweet.tokens)
                return latest_tweet
            else:
                time.sleep(0.1)
    
    def stop_tracking(self) -> None:
        self.stream.disconnect()
        self.tracking = False
