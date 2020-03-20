from datetime import datetime, timedelta
import json
import logging
import os
import random
from pathlib import Path

from nn_toolkit.twitter_tools import load_credentials, init_tweepy_api
import plotly
import plotly.io as pio
import torch
import tweepy

from bot.model_listener import DisasterBotListener
from bot.db.utils import fetch_model_id, make_session
from bot.db.models import Models
from bot.plot import get_recent_predictions, plot_disasters


plotly.io.orca.config.executable = str(Path.home() / 'miniconda3/bin/orca')


DATA_DIR = Path(__file__).parents[1] / 'data'
IMAGES_DIR = DATA_DIR / 'images' / 'plots'
TERMS_DIR = DATA_DIR / 'search_terms'
CREDENTIALS_PATH = Path(os.environ['TWITTER_API_CREDS'])


def load_keywords(filename: str) -> dict:
    with open(TERMS_DIR / filename) as fo:
        return json.load(fo)


def get_latest_model():
    session = make_session()
    model = session.query(Models).order_by(Models.created_at.desc()).first()
    session.close()
    return model


def plot_latest_disasters(num_days: int = 1):
    df = get_recent_predictions(num_days)
    filename = None
    if (df.prediction == 'disaster').sum() > 1:
        fig = plot_disasters(df)
        now = datetime.now()
        since = now - timedelta(num_days)
        since_str = since.strftime('%A %B %d, %Y')
        now_str = now.strftime('%A %B %d, %Y')
        fig.update_layout(title=f'From {since_str} til {now_str}.')
        img_name = now.strftime('%Y-%m-%d-%H:%M:%S')
        filename = IMAGES_DIR / f'{img_name}.png'
        pio.write_image(fig, str(filename))
    return filename


def tweet_latest_disasters(api: tweepy.API, num_days: int = 3):
    filename = plot_latest_disasters(num_days)
    if filename is not None:
        media_response = api.media_upload(str(filename))
        media_id = media_response.media_id
        api.update_status(status=f'A summary of the last {num_days} days.', media_ids=[media_id])


def main(run_id: str = None, threshold: float = 0.75, terms_file: str = 'key_terms.json', silence: bool = False):
    creds = load_credentials(CREDENTIALS_PATH)
    api = init_tweepy_api(creds)

    if run_id == 'latest':
        model = get_latest_model()
        run_id = model.run_id
        logging.info(f'Loading model from {run_id} created at {model.created_at}')

    listener = DisasterBotListener(api, run_id, threshold=threshold, silence=silence)
    stream = tweepy.Stream(api.auth, listener)

    keywords = load_keywords(terms_file)
    num_terms = min(32, len(keywords))
    if not silence:
        tweet_latest_disasters(api, 3)
    try:
        while True:
            track_terms = random.sample([key for key in keywords], num_terms)
            logging.info(f'Tracking : {track_terms}')
            try:
                stream.filter(track=track_terms, languages=["en"])
            except:
                continue
            logging.info(f'End tracking: responded to {stream.listener.responded} from {stream.listener.tweets_seen} seen')
    except:
        stream.listener.say_goodbye()


if __name__ == '__main__':
    import fire
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S'
    )
    fire.Fire(main)
