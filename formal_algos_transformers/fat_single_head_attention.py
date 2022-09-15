"""
Implements Algorithm 4: attention.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class SingleHeadAttention(nn.Module):

    """Applies masked self or cross single-head attention.

    Args:
        d_x (int): size of primary token embedding
        d_z (int): size of each token emebdding in context sequence
        d_attn (int): query-key projection space has size d_attn
        d_out (int): size of contextualized token embedding
        bias (bool): if true, use bias terms for q,k,v
        do_init (bool): can set to False to leave tensors unintialized

    Input:
        x (tensor) [b, l_x, d_x]: token embeddings of primary sequence
        z (tensor) [b, l_z, d_z]: token embeddings of context sequence
        x_mask (tensor) [b, l_x]: primary sequence attention mask (1=attend, 0=dont)
        z_mask (tensor) [b, l_z]: context sequence attention mask (1=attend, 0=dont)

    Output:
        q (tensor) [b, l_x, d_attn]: query vectors for x
        k (tensor) [b, l_z, d_attn]: key vectors for z
        v (tensor) [b, l_z, d_out]: value vectors for z
        score (tensor) [b, l_x, l_z]: (q @ k^T) / sqrt(d_attn) for each batch
            where mask = 1 else minimum value for score tensor dtype
        mask (tensor) [b, l_x, l_z]: mask[i,x,z] = 0 if x_mask[i,x] = 0 or
            z_mask[i,z] = 0 else 1
        attention (tensor) [b, l_x, l_z]: attention weights
            explicitly set to 0 where mask = 0
        vtilde (tensor) [b, l_x, d_out]: contextualized representation of x

    Attributes:
        w_q (tensor): [d_x, d_attn] query weight tensor
        w_k (tensor): [d_z, d_attn] key weight tensor
        w_v (tensor): [d_z, d_out] value weight tensor

        b_q (tensor): [d_attn] query bias tensor
        b_k (tensor): [d_attn] key projection tensor
        b_v (tensor): [d_out] value projection tensor

    """

    def __init__(
        self,
        d_x: int,
        d_z: int,
        d_attn: int,
        d_out: int,
        bias: bool = True,
        do_init: bool = True,
    ):

        super().__init__()
        self.d_x = d_x
        self.d_z = d_z
        self.d_out = d_out
        self.d_attn = d_attn
        self.bias = bias
        self.do_init = do_init
        self.scale = 1 / math.sqrt(d_attn)

        self.w_q = nn.Parameter(torch.empty(d_x, d_attn))
        self.w_k = nn.Parameter(torch.empty(d_z, d_attn))
        self.w_v = nn.Parameter(torch.empty(d_z, d_out))

        if bias:
            self.b_q = nn.Parameter(torch.empty(d_attn))
            self.b_k = nn.Parameter(torch.empty(d_attn))
            self.b_v = nn.Parameter(torch.empty(d_out))

        if self.do_init:
            self.init_weights()


    def init_weights(self):
        nn.init.xavier_normal_(self.w_q)
        nn.init.xavier_normal_(self.w_k)
        nn.init.xavier_normal_(self.w_v)
        if self.bias:
            nn.init.zeros_(self.b_q)
            nn.init.zeros_(self.b_k)
            nn.init.zeros_(self.b_v)


    def forward(
        self,
        x: Tensor,
        z: Tensor,
        x_mask: Tensor,
        z_mask: Tensor,
    ):

        b_x, l_x, d_x = x.shape
        b_z, l_z, d_z = z.shape

        assert b_x == b_z
        b = b_x

        assert d_x == self.d_x
        assert d_z == self.d_z
        assert x_mask.shape == (b, l_x)
        assert z_mask.shape == (b, l_z)

        # this is batch matrix multiplication
        einsum_str = "b i k, k j -> b i j"
        if self.bias:
            q = torch.einsum(einsum_str, x, self.w_q) + self.b_q
            k = torch.einsum(einsum_str, z, self.w_k) + self.b_k
            v = torch.einsum(einsum_str, z, self.w_v) + self.b_v
        else:
            q = torch.einsum(einsum_str, x, self.w_q)
            k = torch.einsum(einsum_str, z, self.w_k)
            v = torch.einsum(einsum_str, z, self.w_v)

        assert q.shape == (b, l_x, self.d_attn)
        assert k.shape == (b, l_z, self.d_attn)
        assert v.shape == (b, l_z, self.d_out)

        # this is batch matrix multiplication with k transposed
        score = torch.einsum("b i k, b j k -> b i j", q, k) * self.scale
        # combine and expand x_mask [b, l_x] and z_mask [b, l_z]
        # [b, l_x, 1] @ [b, 1, l_z] = [b, l_x, l_z]
        mask = x_mask[:, :, None] @ z_mask[:, None, :]
        score = score.masked_fill(~mask.to(torch.bool), torch.finfo(score.dtype).min)
        # multiplying by mask below is not required but ensures
        # attention is 0 where mask is 0
        attention = torch.softmax(score, dim=-1) * mask
        assert mask.shape == score.shape == attention.shape == (b, l_x, l_z)

        vtilde = attention @ v
        assert vtilde.shape == (b, l_x, self.d_out)

        return {
            "q": q,
            "k": k,
            "v": v,
            "score": score,
            "mask": mask,
            "attention": attention,
            "vtilde": vtilde,
        }

    def extra_repr(self):
        return "d_x={}, d_z={}, d_attn={}, d_out={}, bias={}".format(
            self.d_x, self.d_z, self.d_attn, self.d_out, self.bias)


if __name__ == "__main__":

    b = 2
    l_x = 3
    l_z = 4
    d_x = 5
    d_z = 6
    d_out = 7
    d_attn = 8

    x = torch.rand(b, l_x, d_x)
    z = torch.rand(b, l_z, d_z)
    x_mask = torch.tensor([
        [1, 1, 1],
        [1, 1, 1],
    ])
    z_mask = torch.tensor([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ])

    sha = SingleHeadAttention(d_x, d_z, d_attn, d_out)
    out = sha(x, z, x_mask, z_mask)
