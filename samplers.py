import random

import numpy as np
from torch.utils.data import Sampler


class ProportionalTwoClassesBatchSampler(Sampler):
    """
    dataset: DataSet class that returns torch tensors
    batch_size: Size of mini-batches
    minority_size_in_batch: Number of minority class samples in each mini-batch
    majority_priority: If it is True, iterations will include all majority
    samples in the data. Otherwise, it will be completed after all minority samples are used.
    """

    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int,
        minority_size_in_batch: int,
        majority_priority=True,
    ):
        super().__init__(labels)
        self.labels = labels
        self.minority_size_in_batch = minority_size_in_batch
        self.batch_size = batch_size
        self.priority = majority_priority
        self._num_batches = (labels == 0).sum() // (batch_size - minority_size_in_batch)
        self._num_samples = (len(self.labels) // self.batch_size + 1) * self.batch_size

    def __len__(self):
        return self._num_samples

    def __iter__(self):
        if self.minority_size_in_batch > self.batch_size:
            raise ValueError(
                "Number of minority samples in a batch must be lower than batch size!"
            )
        y_indices = [np.where(self.labels == label)[0] for label in np.unique(self.labels)]
        y_indices = sorted(y_indices, key=lambda x: x.shape)

        minority_copy = y_indices[0].copy()

        indices = []
        for _ in range(self._num_batches):
            if len(y_indices[0]) < self.minority_size_in_batch:
                if self.priority:
                    # reloading minority samples
                    y_indices[0] = minority_copy.copy()
            minority = np.random.choice(
                y_indices[0], size=self.minority_size_in_batch, replace=False
            )
            majority = np.random.choice(
                y_indices[1],
                size=(self.batch_size - self.minority_size_in_batch),
                replace=False,
            )
            batch_inds = np.concatenate((minority, majority), axis=0)
            batch_inds = np.random.permutation(batch_inds)
            y_indices[0] = np.setdiff1d(y_indices[0], minority)
            y_indices[1] = np.setdiff1d(y_indices[1], majority)
            indices.extend(batch_inds.tolist())
        return iter(indices[: self._num_samples])
