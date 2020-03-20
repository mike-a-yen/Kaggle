from pathlib import Path
from typing import List, Tuple

from fastai.basic_train import Learner, load_learner
import pandas as pd
import pickle
import toml
import torch
import torch.nn as nn

from nn_toolkit.vocab import Vocab
from src.data.torch_data import PredictionDataset
from src.data.data_pipeline import MLMDataPipeline
from src.model.clas_model import ClasModel
from src.tokenizer import BertTokenizer


_PROJECT_DIR = Path(__file__).parents[1].resolve()
_DATA_DIR = _PROJECT_DIR / 'data'
_MODEL_DIR = _DATA_DIR / 'models'


def load_params(path):
    with open(path) as fo:
        return toml.load(fo)


def get_device(module: nn.Module) -> torch.device:
    return list(module.parameters())[0].device


class Predictor:
    def __init__(self, model: nn.Module, ds: PredictionDataset):
        self.model = model
        self.model.eval()
        self.ds = ds

    @classmethod
    def from_run_dir(cls, dir: Path):
        model_params = load_params(dir / 'model_params.toml')
        model = ClasModel(**model_params)
        model.load(dir / 'clas_model.pth')
        vocab = Vocab.from_file(dir / 'token_store.pkl')
        #with open(dir / 'tokenizer.pkl', 'rb') as fo:
        #    tokenizer = pickle.load(fo)
        tokenizer = BertTokenizer()
        ds = PredictionDataset(vocab, tokenizer)
        return cls(model, ds)

    def predict(self, texts: List[str]) -> torch.FloatTensor:
        batch_size = 32
        i = 0
        probas = []
        while i < len(texts):
            batch = texts[i: i+batch_size]
            probas.append(self.predict_batch(batch))
            i += batch_size
        return torch.cat(probas, dim=0)

    def predict_batch(self, texts: List[str]) -> torch.FloatTensor:
        samples = [self.text_to_sample(text) for text in texts]
        batch = [self.prepare_sample(sample) for sample in samples]
        xb = self.prepare_batch_input(batch)
        with torch.no_grad():
            logits = self.model(xb)
            probas = logits.softmax(-1)
        return probas

    def predict_one(self, text: str) -> torch.FloatTensor:
        sample = self.text_to_sample(text)
        sample = self.prepare_sample(sample)
        xb = self.prepare_batch_input([sample])
        with torch.no_grad():
            logits = self.model(xb)
            probas = logits.softmax(-1)
        return probas

    def prepare_batch_input(self, batch: List[dict]) -> Tuple[dict]:
        xb, yb = MLMDataPipeline.collate_batch(batch)
        device = get_device(self.model)
        xb = {key: val.to(device) for key, val in xb.items()}
        return xb

    def text_to_sample(self, text: str) -> pd.Series:
        return pd.Series({'text': text, 'target': 0})

    def prepare_sample(self, sample: pd.Series) -> dict:
        return self.ds.transform(sample)

    @classmethod
    def from_run_id(cls, run_id: str):
        path = _MODEL_DIR / f'{run_id}'
        learner = load_learner(path, 'clas_learner.pkl')
        return cls(learner)
