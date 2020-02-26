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


class RawDataset:
    def __init__(self, filename: str = "nlp-getting-started.zip", extra_data: List[Path] = []) -> None:
        self.text_col = 'text'
        self.target_col = 'target'
        archive_names = ['train', 'test']
        with zipfile.ZipFile(_RAW_DIRNAME / filename) as zo:
            for name in archive_names:
                df = pd.read_csv(zo.open(f'{name}.csv'))
                if name != 'test':
                    df = df.drop_duplicates(subset=['text'])
                setattr(self, f'{name}_df', df)

        self._set_annotations()
        self.extra_df = self._load_extra_data(extra_data)
        self.hash = hash_df(self.extra_df)
        logging.info(f'Loaded {self.train_df.shape[0]} training data.')
        logging.info(f'Loaded {self.test_df.shape[0]} test data.')
        logging.info(f'Loaded {self.extra_df.shape[0]} extra data.')

    def _set_annotations(self) -> None:
        annotation_filename = _ANNOTATION_DIRNAME / 'annotations.json.gz'
        if annotation_filename.exists():
            annotation_df = pd.read_json(annotation_filename)
            logging.info(f'Loaded {annotation_df.shape[0]} annotations.')
            self.train_df = self.train_df.merge(
                annotation_df[['id', 'annotation']],
                left_on='id',
                right_on='id',
                how='left'
            )
            self.train_df.annotation.fillna(self.train_df.target, inplace=True)
            self.train_df['annotation'] = self.train_df.annotation.astype(self.train_df.target.dtype)
            self.target_col = 'annotation'
        else:
            self.train_df['annotation'] = None
            self.target_col = 'target'

    def _load_extra_data(self, extra_data: List[Path]) -> pd.DataFrame:
        extra_df = pd.DataFrame(columns=self.train_df.columns)
        for filepath in extra_data:
            data = _load_file_as_df(filepath, remove_failed=False)
            if data is None:
                continue
            data = data[~data.text.str.startswith('RT @')]
            data = data[~(data.text == '')]
            extra_df = pd.concat([extra_df, data], sort=False)
        extra_df = extra_df.drop_duplicates(subset=['text'])
        extra_df['created_at'] = pd.to_datetime(extra_df.created_at)
        return extra_df
    