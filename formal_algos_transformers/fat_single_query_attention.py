"""
Implements Algorithm 3: Basic single-query attention.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class SingleQueryAttention(nn.Module):

    """Contextualize single input embedding by attending over context embeddings.

    Args:
        d_x (int): size of primary token embedding
        d_z (int): size of each token emebdding in context sequence
        d_attn (int): query-key projection space has size d_attn
        d_out (int): size of contextualized token embedding
        bias (bool): if true, use bias terms for q,k,v
        do_init (bool): can set to False to leave tensors unintialized

    Input:
        x1 (tensor) [d_x]: single token embedding to be contextualized
        z (tensor) [l_z, d_z]: sequence of context token embeddings
        x1_mask (tensor) []: torch.tensor(0) or torch.tensor(1)
        z_mask (tensor) [l_z]: context attention mask

    Output:
        q (tensor) [d_attn]: query vector for x1
        k (tensor) [l_z, d_attn]: key vectors for z
        v (tensor) [l_z, d_out]: value vectors for z
        mask (tensor): [l_z] mask[i] = 0 if x1_mask is 0 or z_mask[i] = 0 else 1
        score (tensor) [l_z]: score = (q @ k^T) / sqrt(d_attn)
            where mask = 1 else minimum value for score tensor dtype
        attention (tensor) [l_z]: attention weights
            explicitly set to 0 where mask = 0
        vtilde (tensor) [d_out]: contextualized representation of x1

    Attributes:
        w_q (tensor) [d_x, d_attn]: query weight tensor
        w_k (tensor) [d_z, d_attn]: key weight tensor
        w_v (tensor) [d_z, d_out]: value weight tensor

        b_q (tensor) [d_attn]: query bias tensor
        b_k (tensor) [d_attn]: key projection tensor
        b_v (tensor) [d_out]: value projection tensor

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
        self.d_attn = d_attn
        self.d_out = d_out
        self.bias = bias
        self.do_init = do_init
        self.scale = 1 / math.sqrt(d_attn)

        self.w_q = nn.Parameter(torch.empty(d_x, d_attn))
        self.w_k = nn.Parameter(torch.empty(d_z, d_attn))
        self.w_v = nn.Parameter(torch.empty(d_z, d_out))

        if self.bias:
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
        x1: Tensor,
        z: Tensor,
        x1_mask: Tensor,
        z_mask: Tensor,
    ):

        assert x1.dim() == 1
        assert z.dim() == 2
        assert x1_mask.dim() == 0
        assert z_mask.dim() == 1

        (d_x,) = x1.shape
        (l_z, d_z) = z.shape

        assert d_x == self.d_x
        assert d_z == self.d_z
        assert x1_mask.shape == ()
        assert z_mask.shape == (l_z,)

        if self.bias:
            q = x1 @ self.w_q + self.b_q
            k = z @ self.w_k + self.b_k
            v = z @ self.w_v + self.b_v
        else:
            q = x1 @ self.w_q
            k = z @ self.w_k
            v = z @ self.w_v

        assert q.shape == (self.d_attn,)
        assert k.shape == (l_z, self.d_attn)
        assert v.shape == (l_z, self.d_out)

        score = q @ k.T * self.scale
        mask = x1_mask * z_mask
        score = score.masked_fill(~mask.to(torch.bool), torch.finfo(score.dtype).min)
        # multiplying by mask below is not required but ensures
        # attention is 0 where mask is 0
        attention = torch.softmax(score, dim=-1) * mask
        assert score.shape == mask.shape == attention.shape == (l_z,)

        vtilde = attention @ v
        assert vtilde.shape == (self.d_out,)

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

    d_x = 2
    d_z = 3
    d_attn = 4
    d_out = 5
    l_z = 6

    x1 = torch.rand(d_x)
    z = torch.rand(l_z, d_z)
    x1_mask = torch.tensor(0)
    z_mask = torch.tensor([1, 1, 1, 0, 1, 1])

    sqa = SingleQueryAttention(d_x, d_z, d_attn, d_out)
    out = sqa(x1, z, x1_mask, z_mask)
