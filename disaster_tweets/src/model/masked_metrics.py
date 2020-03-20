from fastai.callback import Callback
from fastai.torch_core import add_metrics
import torch
import torch.nn as nn


class MaskedCrossEntropy(nn.CrossEntropyLoss):
    def __init__(self, weight: torch.FloatTensor = None, padding_idx: int = 1) -> None:
        super().__init__(weight)
        self.padding_idx = padding_idx

    def forward(self, input: dict, target: dict):
        mask = target['mask']
        masked_input = input[mask==1]
        masked_target = target['tokens'][mask==1]
        return super().forward(masked_input, masked_target)

    def get_pad_mask(self, tokens: torch.LongTensor):
        pad_mask = tokens == self.padding_idx
        return pad_mask


class MaskedAccuracy(Callback):
    def on_epoch_begin(self, **kwargs):
        "Set the inner value to 0."
        self.val, self.count = 0., 0

    def on_batch_end(self, last_output: torch.FloatTensor, last_target: dict, **kwargs):
        "Update metric computation with `last_output` and `last_target`."
        pred_labels = last_output.argmax(dim=-1)  # (B, T)
        mask = last_target['mask']  # (B, T)
        masked_labels = pred_labels[mask==1]
        targets = last_target['tokens']
        masked_targets = targets[mask==1]

        self.count += masked_targets.size(0)
        val = (masked_labels == masked_targets).float().sum()
        self.val += val.detach().cpu()

    def on_epoch_end(self, last_metrics, **kwargs):
        "Set the final result in `last_metrics`."
        return add_metrics(last_metrics, self.val / self.count)
