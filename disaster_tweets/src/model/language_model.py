import json
import math
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dense import ResidualBlock, DropConnect


def reverse(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Reverse the dimension of a tensor."""
    idx = [i for i in range(x.size(dim) - 1, -1, -1)]
    idx = torch.LongTensor(idx).to(x.device)
    return x.index_select(dim, idx)


class LanguageModel(nn.Module):
    def __init__(
            self, 
            max_vocab_size: int, 
            hidden_size: int = 256, 
            num_layers: int = 3, 
            dropout_rate: float = 0.2, 
            maxlen: int = 64,
            padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.max_vocab_size = max_vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.maxlen = maxlen
        self.padding_idx = padding_idx
        self.encoder = ResTextEncoder(
            self.max_vocab_size,
            self.hidden_size,
            self.num_layers,
            self.dropout_rate,
            self.maxlen,
            self.padding_idx
        )
        self.decoder = nn.Sequential(
            DropConnect(nn.Linear(2 * self.hidden_size, self.hidden_size), self.dropout_rate),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.max_vocab_size)
        )
        self.decoder[-1].weight = self.encoder.embedding_layer.weight

    def forward(self, X: torch.LongTensor) -> torch.FloatTensor:
        B = X.size(0)
        femb, bemb = self.encoder(X)
        emb = torch.cat([femb, bemb], dim=2)
        logit = self.decoder(emb)
        log_prob = F.log_softmax(logit, dim=-1)
        return log_prob.view(-1, self.max_vocab_size)

    @property
    def device(self):
        return self.encoder.device


class TextEncoder(nn.Module):
    def __init__(
            self,
            max_vocab_size: int, 
            hidden_size: int = 256, 
            num_layers: int = 3, 
            dropout_rate: float = 0.2, 
            maxlen: int = 64,
            padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.max_vocab_size = max_vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.maxlen = maxlen
        self.padding_idx = padding_idx
        self.embedding_layer = Embedding(
            self.max_vocab_size,
            hidden_size,
            padding_idx=self.padding_idx,
            maxlen=self.maxlen
        )
        self.rnn = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_rate,
            batch_first=True
        )

    def forward(self, X: torch.LongTensor, state: Tuple[torch.FloatTensor] = None) -> torch.FloatTensor:
        mask = self.get_mask(X).float()  # (B, T)
        if state is None:
            state = self._init_state(X.size(0))
        emb = self.embedding_layer(X)  # (B, T, e)
        emb, state = self.rnn(emb, state)  # (B, T, e)
        emb = emb * (1 - mask.unsqueeze(-1))
        return emb, state
    
    def get_mask(self, X: torch.LongTensor) -> torch.ByteTensor:
        pad_val = self.embedding_layer.padding_idx
        mask = (X == pad_val)
        return mask

    def _init_state(self, batch_size: int) -> Tuple[torch.FloatTensor]:
        directions = self.rnn.bidirectional + 1
        shape = (self.rnn.num_layers * directions, batch_size, self.rnn.hidden_size)
        h = torch.zeros(*shape).to(self.device)
        c = torch.zeros(*shape).to(self.device)
        return h, c
    
    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def get_config(self) -> dict:
        return {
            'max_vocab_size': self.max_vocab_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'maxlen': self.maxlen,
            'padding_idx': self.padding_idx
        }
    
    def save(self, path: Path) -> None:
        config = self.get_config()
        with open(path.with_suffix('.config'), 'w') as fo:
            json.dump(config, fo)
        torch.save(self.state_dict(), path)

    @classmethod
    def from_file(cls, path: Path) -> None:
        config_file = path.with_suffix('.config')
        with open(config_file) as fo:
            config = json.load(fo)
        model = cls(**config)
        model.load_state_dict(
            torch.load(path)
        )
        return model

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True


class ResTextEncoder(TextEncoder):
    def __init__(
            self,
            max_vocab_size: int, 
            hidden_size: int = 256, 
            num_layers: int = 3, 
            dropout_rate: float = 0.2, 
            maxlen: int = 64,
            padding_idx: int = 0,
    ) -> None:
        super().__init__(max_vocab_size, hidden_size, num_layers, dropout_rate, maxlen, padding_idx)
        del self.rnn  # remove the rnn from TextEncoder
        self.fwd_residual = nn.ModuleList([
            ResidualBlock(self.hidden_size, self.dropout_rate)
            for _ in range(self.num_layers)
        ])
        self.bwd_residual = nn.ModuleList([
            ResidualBlock(self.hidden_size, self.dropout_rate)
            for _ in range(self.num_layers)
        ])
        self.fwd_rnns = self._init_rnn(self.hidden_size, self.num_layers)
        self.bwd_rnns = self._init_rnn(self.hidden_size, self.num_layers)

    def forward(self, X: torch.LongTensor) -> torch.FloatTensor:
        mask = self.get_mask(X).float().unsqueeze(-1)  # (B, T, 1)
        emb = self.embedding_layer(X)  # (B, T, e)
        # forward
        femb = emb[:, :-2]
        for i, (res, rnn) in enumerate(zip(self.fwd_residual, self.fwd_rnns)):
            femb = femb * (1 - mask[:, :-2])
            new_emb, state = rnn(femb)
            femb = res(femb, new_emb)
            femb = F.dropout(femb, self.dropout_rate, self.training, False)
        # backward
        bemb = reverse(emb, 1)[:, :-2]
        rmask = reverse(mask, 1)
        for i, (res, rnn) in enumerate(zip(self.bwd_residual, self.bwd_rnns)):
            bemb = bemb * (1 - rmask[:, :-2])
            new_emb, state = rnn(bemb)
            bemb = res(bemb, new_emb)
            bemb = F.dropout(bemb, self.dropout_rate, self.training, False)
        bemb = reverse(bemb, 1)
        return femb, bemb

    def _init_rnn(self, hidden_size: int, num_layers: int) -> nn.ModuleList:
        return nn.ModuleList([
            nn.LSTM(
                hidden_size,
                hidden_size,
                num_layers=1,
                dropout=0.0,
                batch_first=True
            ) for _ in range(num_layers)
        ])

    def _init_state(self, batch_size: int) -> List[Tuple[torch.FloatTensor]]:
        directions = self.rnns[0].bidirectional + 1
        shape = (directions, batch_size, self.rnn.hidden_size)
        state = []
        for _ in range(len(self.rnns)):
            h = torch.zeros(*shape).to(self.device)
            c = torch.zeros(*shape).to(self.device)
            state.append((h, c))
        return state
