from pathlib import Path
import zipfile

import pandas as pd


_PROJECT_DIRNAME = Path(__file__).parents[1]
_DATA_DIRNAME = _PROJECT_DIRNAME / 'data'
_RAW_DIRNAME = _DATA_DIRNAME / 'raw'


class RawDataset:
    def __init__(self, filename: str = "nlp-getting-started.zip") -> None:
        archive_names = ['train', 'test']
        with zipfile.ZipFile(_RAW_DIRNAME / filename) as zo:
            for name in archive_names:
                df = pd.read_csv(zo.open(f'{name}.csv'))
                setattr(self, f'{name}_df', df)
