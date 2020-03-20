import logging
import os
from pathlib import Path
import time

from nn_toolkit.twitter_tools import load_credentials, init_tweepy_api

from bot.manager import StreamManager, RunManager, DataManager


CREDENTIALS_PATH = Path(os.environ['TWITTER_API_CREDS'])

def main():
    creds = load_credentials(CREDENTIALS_PATH)
    api = init_tweepy_api(creds)

    stream_manager = StreamManager(api, ['a', 'the', 'it'], 3, 10)
    data_manager = DataManager()
    df = data_manager.get_data([1, 2, 3])
    print(df)

if __name__ == "__main__":
    import fire
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S'
    )
    fire.Fire(main)
