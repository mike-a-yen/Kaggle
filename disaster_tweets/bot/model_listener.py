from fastai.train import Learner
import tweepy


class DisasterBotListener(tweepy.StreamListener):
    def __init__(self, api: tweepy.API, learner: Learner, threshold: float = 0.9) -> None:
        self.api = api
        self.me = api.me()
        self.learner = learner
        self.threshold = threshold
        self.rate = 0.01
        self.tweets_seen = 0
        self.retweeted = 0
        self.delta = 1e-2
    
    def retweet(self, tweet) -> None:
        tweet_id = tweet.id
        self.api.update_status(
            status = 'Hope it is all ok over there.',
            in_reply_to_status_id = tweet_id ,
            auto_populate_reply_metadata=True
        )

    def on_status(self, tweet) -> bool:
        if tweet.user.id == self.me.id or tweet.in_reply_to_status_id is not None:
            return True
        if not tweet.retweeted and not tweet.text.startswith('RT @'):
            self.tweets_seen += 1
            text = tweet._json['text']
            _, _, probs = self.learner.predict(text)
            is_positive = probs.cpu().numpy()[1]
            if is_positive > self.threshold:
                self.retweeted += 1
                cur_rate = self.retweeted / self.tweets_seen
                print(is_positive, '\n', tweet.text)
                self.retweet(tweet)
            self.set_new_rate()
        return True

    def set_new_rate(self) -> None:
        cur_rate = self.retweeted / self.tweets_seen
        if cur_rate > self.rate:
            self.threshold += self.delta
        else:
            self.threshold -= self.delta
        self.threshold = max(0.5, self.threshold)
