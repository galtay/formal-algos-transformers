"""

"""
import math
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn

from formal_algos_transformers.fat_multi_head_attention import MultiHeadAttention


class ContentEmbeddings(nn.Module):

    def __init__(self, n_v: int, d_e: int, padding_idx: Optional[int] = None):
        """Learned token content embeddings

        Args:
            n_v (int): size of vocabulary
            d_e (int): size of each token embedding
            padding_idx (int): entries at padding_idx do not contribute to the gradient
        """
        super().__init__()
        self.n_v = n_v
        self.d_e = d_e
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(n_v, d_e, padding_idx = padding_idx)

    def forward(self, input_ids: Tensor):
        """
        Args:
            input_ids (tensor): [b, s] vocabulary input ids

        Output:
            embeddings (tensor): [b, s, d_e] token embeddings
        """
        return self.embedding(input_ids)


class PositionEmbeddings(nn.Module):

    def __init__(self, l_max: int, d_e: int):
        """Learned token position embeddings

        Args:
            l_max (int): max sequence length
            d_e (int): size of each token embedding
        """
        super().__init__()
        self.l_max = l_max
        self.d_e = d_e
        self.embedding = torch.nn.Embedding(l_max, d_e)

    def forward(self, input_ids):
        """
        Args:
            input_ids (tensor): [b, s] vocabulary input ids

        Output:
            embeddings (tensor): [1, s, d_e] token embeddings
        """
        _, ll = input_ids.shape
        return self.embedding(torch.arange(ll)[None,:])


class PositionEncodings(nn.Module):

    def __init__(self, l_max: int, d_e: int):
        """Fixed position encodings

        Args:
            l_max (int): max sequence length
            d_e (int): size of each token embedding
        """
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

    def forward(self, input_ids):
        """
        Args:
            input_ids (tensor): [b, s] vocabulary input ids

        Output:
            embeddings (tensor): [1, s, d_e] token embeddings
        """
        _, ll = input_ids.shape
        return self.pos_enc[:, :ll]
