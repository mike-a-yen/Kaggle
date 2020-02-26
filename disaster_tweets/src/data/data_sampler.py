import numpy as np
import torch
import torch.utils as utils


class BucketedSampler(utils.data.Sampler):
    def __init__(self, data_source: utils.data.Dataset) -> None:
        self.data_source = data_source

    def get_one_length(self, sample: torch.Tensor) -> int:
        return sample.size(0)

    def get_lengths(self) -> np.ndarray:
        samples = (self.data_source[i] for i in range(len(self.data_source)))
        lengths = [self.get_one_length(sample) for sample in samples]
        return np.array(lengths)

    @property
    def sample_lengths(self) -> np.ndarray:
        if getattr(self, '_lengths', None) is None:
            self._lengths = self.get_lengths()
        return self._lengths

    def __iter__(self):
        # add a random num between (0, 1) to pseudo-shuffle
        noisy_lengths = self.sample_lengths + np.random.rand(len(self))
        order = noisy_lengths.argsort()
        return iter(order)

    def __len__(self) -> int:
        return len(self.data_source)