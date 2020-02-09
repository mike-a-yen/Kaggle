import json
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.dense import HighwayBlock, DropConnect
from src.model.language_model import ResTextEncoder


def attention(query, key, value, mask = None):
    Tq = query.size(1)
    Tv = key.size(1)
    mask = torch.ones(1, Tq, Tv).triu(diagonal=1).to(query.device)
    attn_score = torch.bmm(query, torch.transpose(key, 1, 2))  # (B, T, T)
    if mask is not None:
        attn_score -= mask * 1e9
    attn_weights = torch.softmax(attn_score, dim=-1)
    output = torch.bmm(attn_weights, value)  # (B, T, e)
    return output, attn_weights


class SimpleClassifier(nn.Module):
    def __init__(self, model_dir: Path) -> None:
        super().__init__()
        self.config = {'model_dir': str(model_dir)}
        self.encoder = ResTextEncoder.from_file(model_dir / 'bi_language_model.pth')
        hidden_size = self.encoder.hidden_size
        dropout_rate = 0.5
        self.densor = nn.Sequential(
            DropConnect(nn.Linear(2 * hidden_size, hidden_size), dropout_rate),
            nn.Tanh(),
            HighwayBlock(hidden_size, dropout_rate),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, X: torch.LongTensor) -> torch.FloatTensor:
        femb, bemb = self.encoder(X)
        femb, bemb = femb[:, :-2], bemb[:, 2:]
        emb = torch.cat([femb, bemb], dim=2)
        emb = emb[:, -1]
        logit = self.densor(emb)
        return logit

    def save(self, path: Path) -> None:
        with open(path.with_suffix('.config'), 'w') as fo:
            json.dump(self.config, fo)
        torch.save(self.state_dict(), path)
    
    @classmethod
    def from_file(cls, path: Path) -> None:
        config_file = path.with_suffix('.config')
        with open(config_file) as fo:
            config = json.load(fo)
        model = cls(Path(config['model_dir']))
        model.load_state_dict(
            torch.load(path)
        )
        return model

    def split_layers(self) -> List:
        groups = [
            self.encoder.embedding_layer,
            [
                self.encoder.fwd_rnns,
                self.encoder.bwd_rnns,
                self.encoder.fwd_residual,
                self.encoder.bwd_residual
            ],
            self.densor
        ]
        return groups
