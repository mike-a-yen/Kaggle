import logging
import os
from pathlib import Path
from typing import List
import zipfile

import pandas as pd

from src.utils import hash_df


_PROJECT_DIRNAME = Path(__file__).parents[2]
_DATA_DIRNAME = _PROJECT_DIRNAME / 'data'
_RAW_DIRNAME = _DATA_DIRNAME / 'raw'
_ANNOTATION_DIRNAME = _DATA_DIRNAME / 'annotations'


def _load_file_as_df(filepath, remove_failed: bool = False) -> pd.DataFrame:
    try:
        data = pd.read_csv(filepath, engine='c')
    except pd.errors.ParserError:
        try:
            data = pd.read_csv(filepath, engine='python')
            if data.shape[0] < 10000:
                os.remove(filepath)
                data = None
        except:
            data = None
    except:
        print(f'Failed on {filepath}.')
        if remove_failed:
            os.remove(str(filepath))
        return
    return data


def _load_unannotated_from_db(num_samples: int = None) -> pd.DataFrame:
    con = os.environ.get('DATABASE_URL')
    if con is None:
        return
    if num_samples == 0:
        return
    query = """
    SELECT
        tweets.tweet_id as id,
        tweets.text as text,
        tweets.created_at as created_at,
        annotations.label as annotation
    FROM tweets
    LEFT JOIN annotations ON tweets.id = annotations.tweet_id
    WHERE annotations.label IS NULL
    """
    if not num_samples is None:
        limit = f'LIMIT {num_samples}'
        query += limit
    query += ';'
    return pd.read_sql_query(query, con)


def _load_annotated_from_db(num_samples: int = None) -> pd.DataFrame:
    con = os.environ.get('DATABASE_URL')
    if con is None:
        return
    query = """
    SELECT
        tweets.tweet_id as id,
        tweets.text,
        tweets.created_at,
        annotations.label as target
    FROM tweets
    INNER JOIN annotations ON tweets.id = annotations.tweet_id
    """
    if not num_samples is None:
        limit = f'{num_samples}'
        query += limit
    query += ';'
    return pd.read_sql_query(query, con)


class RawDataset:
    def __init__(self, filename: str = "nlp-getting-started.zip", limit: int = None) -> None:
        self.text_col = 'text'
        self.target_col = 'target'

        self.train_df = _load_annotated_from_db()
        self.extra_df = self._load_extra_data(limit)
        self._set_annotations()
        self.hash = hash_df(self.extra_df)
        logging.info(f'Loaded {self.train_df.shape[0]} labeled data.')
        logging.info(f'Loaded {self.extra_df.shape[0]} extra data.')

    def _set_annotations(self) -> None:
        annotation_filename = _ANNOTATION_DIRNAME / 'annotations.json.gz'
        if annotation_filename.exists():
            annotation_df = pd.read_json(annotation_filename)
            logging.info(f'Loaded {annotation_df.shape[0]} annotations.')
            anno_map = dict(zip(annotation_df.id, annotation_df.annotation))
            self.train_df['annotation'] = self.train_df.id.map(anno_map)
            # self.test_df['annotation'] = self.test_df.id.map(anno_map)
            self.extra_df.annotation.fillna(self.extra_df.id.map(anno_map), inplace=True)
            training_annotations = (~self.train_df.annotation.isnull()).sum()
            # test_annotations = (~self.test_df.annotation.isnull()).sum()
            extra_annotations = (~self.extra_df.annotation.isnull()).sum()
            logging.info(f'Set {training_annotations} training annotations.')
            # logging.info(f'Set {test_annotations} test annotations.')
            logging.info(f'Set {extra_annotations} extra annotations.')
            self.train_df.annotation.fillna(self.train_df.target, inplace=True)
            self.train_df['annotation'] = self.train_df.annotation.astype(pd.Int64Dtype())
            self.extra_df['annotation'] = self.extra_df.annotation.astype(pd.Int64Dtype())
            self.train_df['target'] = self.train_df.annotation.astype(int)
            self.extra_df['target'] = self.extra_df.annotation
        else:
            self.train_df['annotation'] = None
            self.extra_df['annotation'] = None
        self.target_col = 'target'

    def _load_extra_data(self, limit: int = None) -> pd.DataFrame:
        extra_df = pd.DataFrame(columns=self.train_df.columns)
        more_data = _load_unannotated_from_db(num_samples=limit)
        if more_data is not None:
            extra_df = pd.concat([extra_df, more_data], sort=False, ignore_index=True)
        extra_df = extra_df.drop_duplicates(subset=['text'])
        extra_df['created_at'] = pd.to_datetime(extra_df.created_at)
        return extra_df
    