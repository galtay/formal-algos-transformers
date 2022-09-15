"""
Implements Algorithm 5: multi head attention.
"""

import math

import einops
import torch
import torch.nn as nn
from torch import Tensor


class MultiHeadAttention(nn.Module):

    """Applies masked self or cross multi-head attention.

    Args:
        d_x (int): size of each token embedding in primary sequence
        d_z (int): size of each token emebdding in context sequence
        d_attn (int): query-key projection space has size d_attn
        d_mid (int): value projection space has size d_mid
        n_h (int): number of attention heads
        d_out (int): size of each embedding in output sequence
        bias (bool): if true, use bias terms in q,k,v,o transforms
        do_init (bool): can set to False to leave tensors unintialized

    Input:
        x (tensor) [b, l_x, d_x]: token embeddings of primary sequence
        z (tensor) [b, l_z, d_z]: token embeddings of context sequence
        x_mask (tensor) [b, l_x]: primary sequence attention mask (1=attend, 0=dont)
        z_mask (tensor) [b, l_z]: context sequence attention mask (1=attend, 0=dont)

    Output:
        q (tensor) [b, n_h, l_x, d_attn]: query vectors for x
        k (tensor) [b, n_h, l_z, d_attn]: key vectors for z
        v (tensor) [b, n_h, l_z, d_mid]: value vectors for z
        score (tensor) [b, n_h, l_x, l_z]: (q @ k^T) / sqrt(d_attn) for each batch and head
            where mask = 1 else minimum value for score tensor dtype
        mask (tensor) [b, l_x, l_z]: mask[i,x,z] = 0 if x_mask[i,x] = 0 or
            z_mask[i,z] = 0 else 1
        attention (tensor) [b, n_h, l_x, l_z]: attention weights
            explicitly set to 0 where mask = 0
        yh (tensor) [b, n_h, l_x, d_mid]: contextualized x from each head
        y (tensor): [b, l_x, n_h * d_mid] rearrangement of yh
        vtilde (tensor): [b, l_x, d_out] contextualized representation of x

    Attributes:
        w_q (tensor) [n_h, d_x, d_attn] query weight tensor
        w_k (tensor) [n_h, d_z, d_attn] key weight tensor
        w_v (tensor) [n_h, d_z, d_mid] value weight tensor
        w_o (tensor) [n_h * d_mid, d_out] output weight tensor

        b_q (tensor) [n_h, d_attn]: query bias tensor
        b_k (tensor) [n_h, d_attn]: key bias tensor
        b_v (tensor) [n_h, d_mid]: value bias tensor
        b_o (tensor) [d_out]: output bias tensor

    """

    def __init__(
        self,
        d_x: int,
        d_z: int,
        d_attn: int,
        d_mid: int,
        n_h: int,
        d_out: int,
        bias: bool = True,
        do_init: bool = True,
    ):

        super().__init__()
        self.d_x = d_x
        self.d_z = d_z
        self.d_attn = d_attn
        self.d_mid = d_mid
        self.n_h = n_h
        self.d_out = d_out
        self.bias = bias
        self.do_init = do_init
        self.scale = 1 / math.sqrt(d_attn)

        self.w_q = nn.Parameter(torch.empty(n_h, d_x, d_attn))
        self.w_k = nn.Parameter(torch.empty(n_h, d_z, d_attn))
        self.w_v = nn.Parameter(torch.empty(n_h, d_z, d_mid))
        self.w_o = nn.Parameter(torch.empty(n_h * d_mid, d_out))

        if bias:
            self.b_q = nn.Parameter(torch.empty(n_h, d_attn))
            self.b_k = nn.Parameter(torch.empty(n_h, d_attn))
            self.b_v = nn.Parameter(torch.empty(n_h, d_mid))
            self.b_o = nn.Parameter(torch.empty(d_out))

        if self.do_init:
            self.init_weights()


    def init_weights(self):
        nn.init.xavier_normal_(self.w_q)
        nn.init.xavier_normal_(self.w_k)
        nn.init.xavier_normal_(self.w_v)
        nn.init.xavier_normal_(self.w_o)
        if self.bias:
            nn.init.zeros_(self.b_q)
            nn.init.zeros_(self.b_k)
            nn.init.zeros_(self.b_v)
            nn.init.zeros_(self.b_o)


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

        # batch matrix multiplication for each batch and head
        einsum_str = "b i k, h k j -> b h i j"
        if self.bias:
            q = torch.einsum(einsum_str, x, self.w_q) + self.b_q[None, :, None, :]
            k = torch.einsum(einsum_str, z, self.w_k) + self.b_k[None, :, None, :]
            v = torch.einsum(einsum_str, z, self.w_v) + self.b_v[None, :, None, :]
        else:
            q = torch.einsum(einsum_str, x, self.w_q)
            k = torch.einsum(einsum_str, z, self.w_k)
            v = torch.einsum(einsum_str, z, self.w_v)

        assert q.shape == (b, self.n_h, l_x, self.d_attn)
        assert k.shape == (b, self.n_h, l_z, self.d_attn)
        assert v.shape == (b, self.n_h, l_z, self.d_mid)

        # this is batch matrix multiplication with k transposed
        score = torch.einsum("b h i k, b h j k -> b h i j", q, k) * self.scale
        assert score.shape == (b, self.n_h, l_x, l_z)

        # combine and expand x_mask [b, l_x] and z_mask [b, l_z]
        # [b, l_x, 1] @ [b, 1, l_z] = [b, l_x, l_z]
        mask = x_mask[:, :, None] @ z_mask[:, None, :]
        assert mask.shape == (b, l_x, l_z)

        # create [b, 1, l_x, l_z] which is broadcastable to [b, h, l_x, l_z]
        emask = mask[:, None, :, :]
        bmask = emask.to(torch.bool)
        assert emask.shape == bmask.shape == (b, 1, l_x, l_z)
        score = score.masked_fill(~bmask, torch.finfo(score.dtype).min)
        # multiplying by mask below is not required but ensures
        # attention is 0 where mask is 0
        attention = torch.softmax(score, dim=-1) * emask
        assert attention.shape == (b, self.n_h, l_x, l_z)

        # yh [b, h, l_x, d_mid]
        yh = attention @ v
        assert yh.shape == (b, self.n_h, l_x, self.d_mid)

        # y [b, l_x, h * d_mid]
        y = einops.rearrange(yh, "b h l d -> b l (h d)")
        assert y.shape == (b, l_x, self.n_h * self.d_mid)

        if self.bias:
            vtilde = torch.einsum("b l k, k d -> b l d", y, self.w_o) + self.b_o
        else:
            vtilde = torch.einsum("b l k, k d -> b l d", y, self.w_o)

        assert vtilde.shape == (b, l_x, self.d_out)

        return {
            "q": q,
            "k": k,
            "v": v,
            "score": score,
            "mask": mask,
            "yh": yh,
            "y": y,
            "attention": attention,
            "vtilde": vtilde,
        }


    def extra_repr(self):
        return "d_x={}, d_z={}, d_attn={}, d_mid={}, n_h={}, d_out={}, bias={}".format(
            self.d_x, self.d_z, self.d_attn, self.d_mid, self.n_h, self.d_out, self.bias)


if __name__ == "__main__":

    b = 2
    l_x = 3
    l_z = 4
    d_x = 5
    d_z = 6
    d_out = 7
    d_attn = 8
    d_mid = 9
    n_h = 4

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

    mha = MultiHeadAttention(d_x, d_z, d_attn, d_mid, n_h, d_out)
    out = mha(x, z, x_mask, z_mask)
