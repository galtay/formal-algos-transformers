"""

"""
import math
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn


class ContentEmbeddings(nn.Module):

    """Learned token content embeddings

    Args:
        n_v (int): size of vocabulary
        d_e (int): size of each token embedding
        padding_idx (int): entries at padding_idx do not contribute to the gradient

    Input:
        input_ids (tensor) [batch_size, seq_len]: vocabulary input ids

    Output:
        embeddings (tensor) [batch_size, seq_len, d_e]: token embeddings
    """

    def __init__(self, n_v: int, d_e: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.n_v = n_v
        self.d_e = d_e
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(n_v, d_e, padding_idx = padding_idx)

    def forward(self, input_ids: Tensor):
        assert input_ids.dim() == 2
        return self.embedding(input_ids)

    def extra_repr(self):
        return "n_v={}, d_e={}, padding_idx={}".format(
            self.n_v, self.d_e, self.padding_idx)


class PositionEmbeddings(nn.Module):

    """Learned position embeddings

    Args:
        l_max (int): max sequence length
        d_e (int): size of each token embedding

    Input:
        input_ids (tensor) [batch_size, seq_len]: vocabulary input ids

    Output:
        embeddings (tensor) [1, seq_len, d_e]: token embeddings

    """

    def __init__(self, l_max: int, d_e: int):
        super().__init__()
        self.l_max = l_max
        self.d_e = d_e
        self.embedding = torch.nn.Embedding(l_max, d_e)

    def forward(self, input_ids: Tensor):
        _, ll = input_ids.shape
        return self.embedding(torch.arange(ll)[None,:])

    def extra_repr(self):
        return "l_max={}, d_e={}".format(self.l_max, self.d_e)


class PositionEncodings(nn.Module):

    """Fixed position encodings

    Args:
        l_max (int): max sequence length
        d_e (int): size of each token embedding

    Input:
        input_ids (tensor) [batch_size, seq_len]: vocabulary input ids

    Output:
        embeddings (tensor) [1, seq_len, d_e]: token embeddings

    """

    def __init__(self, l_max: int, d_e: int):
        super().__init__()
        self.l_max = l_max
        self.d_e = d_e

        encodings = torch.zeros(l_max, d_e)                          # [l_max, de]
        position = torch.arange(l_max, dtype=torch.float32)[:, None] # [l_max, 1]
        two_i = torch.arange(0, d_e, 2, dtype=torch.float32)         # [l_max//2]
        div_term = torch.exp(two_i * -(math.log(10000.0) / d_e))
        encodings[:, 0::2] = torch.sin(position * div_term)
        encodings[:, 1::2] = torch.cos(position * div_term)
        encodings = encodings[None, :, :]  # [1, l_max, d_e]
        self.register_buffer("pos_enc", encodings)

    def forward(self, input_ids: Tensor):
        _, ll = input_ids.shape
        return self.pos_enc[:, :ll, :]

    def extra_repr(self):
        return "l_max={}, d_e={}".format(self.l_max, self.d_e)


if __name__ == "__main__":

    n_v = 10
    d_e = 4
    l_max = 3

    input_ids = torch.tensor([
        [8,1,3],
        [9,4,2],
    ])

    ce = ContentEmbeddings(n_v, d_e)
    x_ce = ce(input_ids)

    pemb = PositionEmbeddings(l_max, d_e)
    x_pemb = pemb(input_ids)

    penc = PositionEncodings(l_max, d_e)
    x_penc = penc(input_ids)
