import os
from pathlib import Path
from typing import List
import zipfile

import pandas as pd


_PROJECT_DIRNAME = Path(__file__).parents[2]
_DATA_DIRNAME = _PROJECT_DIRNAME / 'data'
_RAW_DIRNAME = _DATA_DIRNAME / 'raw'


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
        archive_names = ['train', 'test']
        with zipfile.ZipFile(_RAW_DIRNAME / filename) as zo:
            for name in archive_names:
                df = pd.read_csv(zo.open(f'{name}.csv'))
                if name != 'test':
                    df = df.drop_duplicates(subset=['text', 'keyword'])
                setattr(self, f'{name}_df', df)
        self.extra_df = self._load_extra_data(extra_data)
        print(f'Loaded {self.train_df.shape[0]} training data.')
        print(f'Loaded {self.test_df.shape[0]} test data.')
        print(f'Loaded {self.extra_df.shape[0]} extra data.')

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
        return extra_df
    