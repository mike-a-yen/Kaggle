import logging
import os
import random
import threading
from typing import Collection

import pandas as pd
import tweepy

from bot.model_listener import StreamListener


class StreamManager:
    def __init__(
        self, api: tweepy.API,
        keywords: Collection[str],
        tracking_size: int = 32,
        time_limit: int = 300
    ) -> None:
        self.api = api
        self.keywords = keywords
        self.tracking_size = tracking_size
        self.time_limit = time_limit
        self.listener = StreamListener(self.api, time_limit=self.time_limit)
        self.stream = tweepy.Stream(self.api.auth, self.listener)

        self.thread: threading.Thread
        self.reset()

    def start(self) -> None:
        self._running = True
        logging.info('Starting to stream.')
        self.thread.start()

    def stop(self) -> None:
        self._running = False
        logging.info('Terminating stream.')
        self.thread.join()
        logging.info('Thread terminated.')
    
    def dump(self) -> Collection[int]:
        return self.listener.dump()

    def reset(self) -> None:
        self.thread = threading.Thread(target=self._run, args=())
        self.thread.daemon = True
        self.listener.reset()

    def _run(self) -> None:
        try:
            while self._running:
                track_terms = random.sample(self.keywords, self.tracking_size)
                logging.info(f'Tracking : {track_terms}')
                try:
                    self.stream.filter(track=track_terms, languages=["en"], is_async=False)
                except:
                    continue
                logging.info(f'End tracking: Seen {self.listener.tweets_seen} tweets.')
        except:
            return

    @property
    def tweets_seen(self) -> int:
        return self.listener.tweets_seen


class DataManager:
    def __init__(self):
        self.con = os.environ['DATABASE_URL']

        self.query = """
        SELECT tweets.id, tweets.text, tweets.created_at
        FROM tweets
        WHERE tweets.id = ANY(%(ids)s);
        """

    def get_data(self, ids: Collection[int]) -> pd.DataFrame:
        return pd.read_sql_query(self.query, self.con, params={'ids': ids})


class ModelManager:
    def __init__(self, model) -> None:
        self.model = model
        self.data_manager = DataManager()

    def predict_on_ids(self, ids: Collection[int]) -> Collection[float]:
        df = self.data_manager.get_data(ids)
        # TODO: model predict


class RunManager:
    def __init__(self, stream_manager: StreamManager, model_manager: ModelManager) -> None:
        self.stream_manager = stream_manager
        self.model_manager = model_manager
    
    def start(self) -> None:
        try:
            while True:
                self.stream_manager.start()
                while self.stream_manager.tweets_seen < 10:
                    time.sleep(0.001)
                self.stream_manager.stop()
                ids = self.stream_manager.dump()
                logging.info(f'Seen {len(ids)}')
                self.stream_manager.reset()
        except:
            logging.info(f'Stopping...')
            self.stream_manager.stop()
            ids = self.stream_manager.dump()
            num_seen += len(ids)
