import torch
import torch.nn as nn

from nn_toolkit.text.model import Embedding
from nn_toolkit.text.model.language_model import SimpleDecoder


class MaskedLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int = 2, dropout_rate: float = 0.0, padding_idx: int = 1) -> None:
        super().__init__()
        self.embedding_layer = Embedding(vocab_size, hidden_size, maxlen=None, padding_idx=padding_idx)
        self.encoder = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=True,
        )
        self.decoder = SimpleDecoder(
            vocab_size,
            (1 + self.encoder.bidirectional) * hidden_size,
            dropout_rate
        )

    def forward(self, X: dict) -> torch.FloatTensor:
        tokens = X['masked_tokens']
        pad_mask = self.embedding_layer.get_mask(tokens)
        emb = self.embedding_layer(tokens)
        emb, *_ = self.encoder(emb)
        logit = self.decoder(emb)
        logit = self.apply_pad_mask(logit, pad_mask)
        return logit

    def apply_pad_mask(self, emb: torch.FloatTensor, mask: torch.ByteTensor) -> torch.FloatTensor:
        pad_val = self.embedding_layer.padding_idx
        ext_mask = mask.unsqueeze(-1)
        emb = emb.masked_fill(ext_mask, -1e-9)
        emb = emb + (-1e9 * ext_mask)
        return emb
    
    def reset(self) -> None:
        """Required by fastai RNNTrainer"""
        return

    @property
    def class_weight(self) -> torch.FloatTensor:
        weight = torch.ones(self.embedding_layer.vocab_size)
        pad_val = self.embedding_layer.padding_idx
        weight[pad_val] = 0
        return weight.to(self.device)

    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def save(self, path) -> None:
        state = self.state_dict()
        torch.save(state, path)
    
    def load(self, path) -> None:
        self.load_state_dict(torch.load(path))
