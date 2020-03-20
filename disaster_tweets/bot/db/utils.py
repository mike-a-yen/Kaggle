import os

from nn_toolkit.twitter_tools import Tweet
import pandas as pd

from bot.db.base import make_session
from bot.db.models import Annotations, Tweets, TweetPredictions, Models, ModelEvaluations


def log_tweet_to_db(tweet) -> int:
    tweet = Tweet(tweet._json)
    session = make_session()
    in_db = session.query(Tweets).filter(Tweets.tweet_id==tweet.id).count()
    if in_db:
        session.close()
        return
    db_tweet = Tweets(tweet)
    session.add(db_tweet)
    session.commit()
    tweet_id = db_tweet.id
    session.close()
    return tweet_id


def log_prediction_to_db(tweet_id: int, model_id: int, pred_label: int, proba: float) -> int:
    session = make_session()
    prediction = TweetPredictions(tweet_id, model_id, pred_label, proba)
    prediction_id = prediction.id
    session.add(prediction)
    session.commit()
    session.close()
    return prediction_id


def log_annotation(tweet_id: int, annotation: int) -> int:
    session = make_session()
    in_db = session.query(Annotations).filter(Annotations.tweet_id==tweet_id).count()
    if in_db:
        session.close()
        return
    anno = Annotations(tweet_id, annotation)
    session.add(anno)
    anno_id = anno.id
    session.commit()
    session.close()
    return anno_id


def log_model(run_id: str, run_dir: str) -> int:
    session = make_session()
    model = Models(run_id, run_dir)
    session.add(model)
    session.commit()
    model_id = model.id
    session.close()
    return model_id


def log_evaluation(model_id: int, **kwargs) -> int:
    session = make_session()
    model_eval = ModelEvaluations(model_id, **kwargs)
    model_eval_id = model_eval.id
    session.add(model_eval)
    session.commit()
    session.close()
    return model_eval_id


def fetch_model_id(run_id: str) -> int:
    session = make_session()
    query = session.query(Models).filter(Models.run_id == run_id)
    model_id = query.first().id
    session.close()
    return model_id


def fetch_tweets_without_prediction(run_id: str, since: str, limit: int = 100) -> pd.DataFrame:
    con = os.environ['DATABASE_URL']
    query = f"""
    WITH tweet_predictions AS (
        SELECT
            tweet_predictions.tweet_id,
            tweet_predictions.proba
        FROM tweet_predictions
        INNER JOIN models ON models.id = tweet_predictions.model_id
        WHERE models.run_id = '{run_id}'
        )
    SELECT 
        tweets.id,
        tweets.tweet_id,
        tweets.text,
        tweets.created_at
    FROM tweets
    LEFT JOIN tweet_predictions ON tweet_predictions.tweet_id = tweets.id
    WHERE tweet_predictions.proba is null
    AND tweets.created_at >= '{since}'
    ORDER BY tweets.created_at DESC
    LIMIT {limit}
    ;
    """
    return pd.read_sql_query(query, con)
