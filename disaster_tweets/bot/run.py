import os
from pathlib import Path

from fastai.core import defaults
from fastai.train import load_learner
from nn_toolkit.twitter_tools import load_credentials, init_tweepy_api
import torch
import tweepy

from bot.model_listener import DisasterBotListener


defaults.device = torch.device('cpu')
_DATA_DIR = Path(__file__).parents[1].resolve() / 'data'
_MODEL_DIR = _DATA_DIR / 'models'
CREDENTIALS_PATH = Path(os.environ['TWITTER_API_CREDS'])


def main(run_id: str):
    run_dir = _MODEL_DIR / run_id
    learner = load_learner(run_dir, 'clas_learner', device=torch.device('cpu'))
    creds = load_credentials(CREDENTIALS_PATH)
    api = init_tweepy_api(creds)
    listener = DisasterBotListener(api, learner, threshold=0.5)
    stream = tweepy.Stream(api.auth, listener)
    stream.filter(
        track=[
            'earthquake',
            'tremor',
            'fire',
            'wildfire',
            'thunder',
            'lightning',
            'flood',
            'crash',
            'collision',
            'hail',
            'snow',
            'storm',
            'weather',
        ],
        languages=["en"]
    )


if __name__ == '__main__':
    import fire
    fire.Fire(main)
