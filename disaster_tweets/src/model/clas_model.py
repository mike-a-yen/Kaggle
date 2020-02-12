import json
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn_toolkit.text.model.dense import HighwayBlock, DropConnect
from nn_toolkit.text.model.embedding import Embedding
from nn_toolkit.text.model.sequence import TextSequenceEncoder


def attention(query, key, value, mask = None):
    Tq = query.size(1)
    Tv = key.size(1)
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
        self.embedding_layer = Embedding.from_file(model_dir / 'fwd_embedding.pth')
        self.encoder = TextSequenceEncoder.from_file(model_dir / 'fwd_language_model.pth')
        hidden_size = self.encoder.hidden_size
        dropout_rate = 0.5
        self.q_proj, self.k_proj = nn.Linear(hidden_size, hidden_size), nn.Linear(hidden_size, hidden_size)
        self.densor = nn.Sequential(
            DropConnect(nn.Linear(hidden_size, hidden_size), dropout_rate),
            nn.Tanh(),
            HighwayBlock(hidden_size, dropout_rate),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, X: torch.LongTensor) -> torch.FloatTensor:
        emb = self.embedding_layer(X)
        mask = self.embedding_layer.get_mask(X)
        emb = self.encoder(emb, mask)
        q, k = self.q_proj(emb), self.k_proj(emb)
        emb, attn = attention(q, k, k, mask.float().unsqueeze(-1))
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
            [self.embedding_layer],
            [self.encoder],
            [
                self.rnn,
                self.densor
            ],
        ]
        return groups
