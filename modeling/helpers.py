from typing import List

import torch


def pad_tensor(t: torch.Tensor, length: List[int]):
    batch_size = len(t)
    dim = t[0].shape[-1]
    max_L = max(length)

    pad_t = torch.ones((batch_size, max_L, dim)).to(t[0].device)
    pad_mask = torch.zeros((batch_size, max_L)).to(t[0].device)
    for b in range(batch_size):
        pad_t[b, : length[b]] = t[b]
        pad_mask[b, : length[b]] = 0
    pad_mask = pad_mask > 0.5
    return pad_t, pad_mask


def unpad_tensor(pad_t: torch.Tensor, length: List[int]):
    batch_size = len(pad_t)
    t = []
    for b in range(batch_size):
        t.append(pad_t[b, : length[b]])
    return t
