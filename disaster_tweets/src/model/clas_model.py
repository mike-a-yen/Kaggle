import json
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn_toolkit.text.model import Embedding
from nn_toolkit.text.model.dense import DropConnect, HighwayBlock
from nn_toolkit.utils import freeze_parameters


def attention(query, key, value, mask = None):
    Tq = query.size(1)
    Tv = key.size(1)
    attn_score = torch.bmm(query, torch.transpose(key, 1, 2))  # (B, T, T)
    if mask is not None:
        attn_score -= mask * 1e9
    attn_weights = torch.softmax(attn_score, dim=-1)
    output = torch.bmm(attn_weights, value)  # (B, T, e)
    return output, attn_weights


class ClasModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int = 2, dropout_rate: float = 0.0, padding_idx: int = 1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.embedding_layer = Embedding(vocab_size, hidden_size, maxlen=None, padding_idx=padding_idx)
        self.encoder = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=True,
        )
        self.decoder = self.build_decoder()

    def forward(self, X: dict) -> torch.FloatTensor:
        tokens = X['tokens']
        pad_mask = self.embedding_layer.get_mask(tokens)
        emb = self.embedding_layer(tokens)
        emb, *_ = self.encoder(emb)
        emb = self.apply_pad_mask(emb, pad_mask)
        emb = emb.mean(1)
        logit = self.decoder(emb)
        return logit

    def apply_pad_mask(self, emb: torch.FloatTensor, mask: torch.ByteTensor) -> torch.FloatTensor:
        mask = mask.unsqueeze(-1)
        emb = emb * (1 - mask.float())
        return emb

    @classmethod
    def from_language_model(cls, language_model: nn.Module) -> nn.Module:
        vocab_size = language_model.embedding_layer.vocab_size
        hidden_size = language_model.embedding_layer.hidden_size
        num_layers = language_model.encoder.num_layers
        dropout_rate = language_model.encoder.dropout
        padding_idx = language_model.embedding_layer.padding_idx
        model = cls(vocab_size, hidden_size, num_layers, dropout_rate, padding_idx)
        model.embedding_layer = language_model.embedding_layer
        model.encoder = language_model.encoder
        model.decoder = model.build_decoder()
        freeze_parameters(model.embedding_layer)
        freeze_parameters(model.encoder)
        return model

    def build_decoder(self) -> nn.Sequential:
        hidden_size = self.embedding_layer.hidden_size
        encoder_size = (1 + self.encoder.bidirectional) * hidden_size
        decoder = nn.Sequential(
            DropConnect(
                nn.Linear(encoder_size, hidden_size),
                self.dropout_rate
            ),
            nn.Tanh(),
            DropConnect(
                nn.Linear(hidden_size, hidden_size),
                self.dropout_rate
            ),
            nn.Tanh(),
            nn.Linear(hidden_size, 2)
        )
        return decoder

    def save(self, path) -> None:
        state = self.state_dict()
        torch.save(state, path)

    def load(self, path) -> None:
        self.load_state_dict(torch.load(path))
