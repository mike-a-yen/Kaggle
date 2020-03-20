import os

import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go


def get_recent_predictions(num_days: int = 1) -> pd.DataFrame:
    con = os.environ['DATABASE_URL']
    query = f"""
        SELECT
            tweets.id,
            tweets.text,
            tweets.lat,
            tweets.long,
            tweets.created_at,
            tweet_predictions.model_id,
            tweet_predictions.proba,
            CASE
                WHEN tweet_predictions.pred_label = 1 THEN 'disaster'
                ELSE 'not disaster'
            END as prediction
        FROM tweets
        INNER JOIN tweet_predictions ON tweets.id = tweet_predictions.tweet_id
        WHERE tweets.lat is not null and tweets.long is not null
        AND tweets.created_at >= NOW() - INTERVAL '{num_days} days'
        ;
    """
    return pd.read_sql_query(query, con)


def plot_predictions(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter_geo(
        df,
        lat=df['lat'],
        lon=df['long'],
        text=df['text'],
        color='prediction',
        projection='natural earth'
    )
    return fig


def plot_disasters(df: pd.DataFrame) -> go.Figure:
    is_disaster = df.prediction == 'disaster'
    return plot_predictions(df[is_disaster])
