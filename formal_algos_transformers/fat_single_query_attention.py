"""
Implements Algorithm 3: Basic single-query attention.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class SingleQueryAttention(nn.Module):

    def __init__(
        self,
        d_x: int,
        d_z: int,
        d_out: int,
        d_attn: int,
        bias: bool = True,
        do_init: bool = True,
    ):

        """Contextualize input embedding by attending over a sequence of context embeddings.

        Args:
            d_x (int): size of primary token embedding
            d_z (int): size of each token emebdding in context sequence
            d_out (int): size of contextualized token embedding
            d_attn (int): query-key projection space has size d_attn
            bias (bool): if true, use bias terms in q,k,v
            do_init (bool): can set to False to leave tensors unintialized

        Attributes:
            w_q (tensor): [d_x, d_attn] query weight tensor
            w_k (tensor): [d_z, d_attn] key weight tensor
            w_v (tensor): [d_z, d_out] value weight tensor

            b_q (tensor): [d_attn] query bias tensor
            b_k (tensor): [d_attn] key projection tensor
            b_v (tensor): [d_out] value projection tensor

        """

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
        x: Tensor,
        zs: Tensor,
        x_mask: Tensor,
        zs_mask: Tensor,
    ):
        """
        Args:
            x (tensor): [d_x] single token embedding to be contextualized
            zs (tensor): [l_z, d_z] sequence of context token embeddings
            x_mask (tensor): [] rank zero tensor mask i.e. torch.tensor(0) or torch.tensor(1)
            zs_mask (tensor): [l_z] context attention mask

        Output:
            q (tensor): [1, d_attn] query vector for x
            k (tensor): [l_z, d_attn] key vectors for zs
            v (tensor): [l_z, d_out] value vectors for zs
            score (tensor): [l_z] (q k^T) / sqrt(d_attn)
            mask (tensor): [l_z] mask[i] = 0 if x_mask is 0 or zs_mask[i] = 0 else 1
            bmask (tensor): [l_z] bmask[i] = False if x_mask is 0  or zs_mask[i] = 0 else True
            masked_score (tensor): [l_z] masked_score[i] = score[i] where mask[i] = 1
                else minimum value for score tensor dtype

            attention (tensor): [l_z] attention weights. attention[i] is explicitly set to 0
                if either x mask is 0 or zs_mask[i] is 0
            vt (tensor): [d_out] updated representation of the token embedding x, folding in
                information from tokens in zs

        """
        assert len(x.shape) == 1
        assert len(zs.shape) == 2
        assert len(x_mask.shape) == 0
        assert len(zs_mask.shape) == 1

        (d_x,) = x.shape
        (l_z, d_z) = zs.shape

        assert d_x == self.d_x
        assert d_z == self.d_z
        assert x_mask.shape == ()
        assert zs_mask.shape == (l_z,)

        # q = x @ w_q + b_q: [d_x] @ [d_x, d_attn] + [d_attn] = [d_attn]
        # k = zs @ w_k + b_k: [l_z, d_z] @ [d_z, d_attn] + [d_attn] = [l_z, d_attn]
        # v = zs @ w_v + b_v: [l_z, d_z] @ [d_z, d_out] + [d_out] = [l_z, d_out]
        if self.bias:
            q = x @ self.w_q + self.b_q
            k = zs @ self.w_k + self.b_k
            v = zs @ self.w_v + self.b_v
        else:
            q = x @ self.w_q
            k = zs @ self.w_k
            v = zs @ self.w_v

        assert q.shape == (self.d_attn,)
        assert k.shape == (l_z, self.d_attn)
        assert v.shape == (l_z, self.d_out)

        # score = (q @ k.T) / sqrt(d_attn): [d_attn] @ [d_attn, l_z] = [l_z]
        score = q @ k.T * self.scale

        # combine x_mask and zs_mask
        mask = x_mask * zs_mask
        bmask = mask.to(torch.bool)
        masked_score = score.masked_fill(~bmask, torch.finfo(score.dtype).min)
        assert score.shape == mask.shape == bmask.shape == masked_score.shape == (l_z,)

        # why do we multiply by mask at the end?
        # if x_mask is 0 then masked_score will be all 0s
        # and the softmax over masked_score will be 1/l_z for each entry.
        # this doesn't matter b/c you will not use contextualized representations
        # of masked tokens BUT it makes more sense to me to have attention = 0
        # for these tokens
        attention = torch.softmax(masked_score, dim=-1) * mask
        assert attention.shape == (l_z,)

        # each row of v is an output token representation
        # attention is telling us the weights to use when combining
        vt = torch.zeros(self.d_out)
        for tok in range(l_z):
            vt += attention[tok] * v[tok, :]
        assert vt.shape == (self.d_out,)

        return {
            "q": q,
            "k": k,
            "v": v,
            "score": score,
            "mask": mask,
            "bmask": bmask,
            "masked_score": masked_score,
            "attention": attention,
            "vt": vt,
        }


    def extra_repr(self):
        return "d_x={}, d_z={}, d_out={}, d_attn={}, bias={}".format(
            self.d_x, self.d_z, self.d_out, self.d_attn, self.bias)
