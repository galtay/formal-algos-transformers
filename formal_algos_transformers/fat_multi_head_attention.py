"""
Implements Algorithm 5: multi head attention.
"""

import math

import einops
import torch
import torch.nn as nn
from torch import Tensor


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_x: int,
        d_z: int,
        d_out: int,
        d_attn: int,
        d_mid: int,
        n_h: int,
        bias: bool = True,
        do_init: bool = True,
    ):

        """Applies masked self or cross multi-head attention.

        Args:
            d_x (int): size of each token embedding in primary sequence
            d_z (int): size of each token emebdding in context sequence
            d_out (int): size of each embedding in output sequence
            d_attn (int): query-key projection space has size d_attn
            d_mid (int): value projection space has size d_mid
            n_h (int): number of attention heads
            bias (bool): if true, use bias terms in q,k,v,o transforms
            do_init (bool): can set to False to leave tensors unintialized

        Attributes:
            w_q (tensor): [n_h, d_x, d_attn] query weight tensor
            w_k (tensor): [n_h, d_z, d_attn] key weight tensor
            w_v (tensor): [n_h, d_z, d_mid] value weight tensor
            w_o (tensor): [n_h * d_mid, d_out] output weight tensor

            b_q (tensor): [n_h, d_attn] query bias tensor
            b_k (tensor): [n_h, d_attn] key bias tensor
            b_v (tensor): [n_h, d_mid] value bias tensor
            b_o (tensor): [d_out] output bias tensor

        """

        super().__init__()
        self.d_x = d_x
        self.d_z = d_z
        self.d_out = d_out
        self.d_attn = d_attn
        self.d_mid = d_mid
        self.n_h = n_h
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
        """
        Args:
            x (tensor): [b, l_x, d_x] token embeddings of primary sequence
            z (tensor): [b, l_z, d_z] token embeddings of context sequence
            x_mask (tensor): [b, l_x] primary sequence attention mask (1=attend, 0=dont)
            z_mask (tensor): [b, l_z] context sequence attention mask (1=attend, 0=dont)

        Output:
            q (tensor): [b, n_h, l_x, d_attn] query vectors for x
            k (tensor): [b, n_h, l_z, d_attn] key vectors for z
            v (tensor): [b, n_h, l_z, d_mid] value vectors for z

            mask (tensor): [b, l_x, l_z] mask[b,x,z] = 0 if x_mask[b,x] = 0 or
                z_mask[b,z] = 0 else 1
            bmask (tensor): [b, l_x, l_z] bmask[b,x,z] = False if mask[b,x,z] = 0 else True

            score (tensor): [b, n_h, l_x, l_z] (q k^T) / sqrt(d_attn) for each batch and head
            masked_score (tensor): [b, n_h, l_x, l_z] masked_score[b,h,x,z] = score[b,h,x,z] where
                mask[b,h,x,z] = 1 else minimum value for score tensor dtype
            attention (tensor): [b, n_h, l_x, l_z] attention weights. attention[b,h,x,z] is explicitly
                set to 0 if either x mask[b,x] = 0 or z_mask[b,z] = 0

            yh (tensor): [b, n_h, l_x, d_mid] output vectors for each head
            y (tensor): [b, l_x, n_h * d_mid] rearrangement of yh

            vt (tensor): [b, l_x, d_out] updated representation of the tokens in x, folding in
                information from tokens in z

        """
        b_x, l_x, d_x = x.shape
        b_z, l_z, d_z = z.shape

        assert b_x == b_z
        b = b_x

        assert d_x == self.d_x
        assert d_z == self.d_z
        assert x_mask.shape == (b, l_x)
        assert z_mask.shape == (b, l_z)

        # for each batch and each head
        # q = x @ w_q + b_q: [l_x, d_x] @ [d_x, d_attn] + [d_attn] = [l_x, d_attn]
        # k = z @ w_k + b_k: [l_z, d_z] @ [d_z, d_attn] + [d_attn] = [l_z, d_attn]
        # v = z @ w_v + b_v: [l_z, d_z] @ [d_z, d_out]  + [d_out]  = [l_z, d_out]

        einsum_str = "b i k, h k j -> b h i j"
        if self.bias:
            q = torch.einsum(einsum_str, x, self.w_q) + self.b_q[:, None, :]
            k = torch.einsum(einsum_str, z, self.w_k) + self.b_k[:, None, :]
            v = torch.einsum(einsum_str, z, self.w_v) + self.b_v[:, None, :]
        else:
            q = torch.einsum(einsum_str, x, self.w_q)
            k = torch.einsum(einsum_str, z, self.w_k)
            v = torch.einsum(einsum_str, z, self.w_v)

        assert q.shape == (b, self.n_h, l_x, self.d_attn)
        assert k.shape == (b, self.n_h, l_z, self.d_attn)
        assert v.shape == (b, self.n_h, l_z, self.d_mid)

        # for each batch and each head
        # score = (q @ k.T) / sqrt(d_attn): [l_x, d_attn] @ [d_attn, l_z] = [l_x, l_z]
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

        masked_score = score.masked_fill(~bmask, torch.finfo(score.dtype).min)
        assert masked_score.shape == (b, self.n_h, l_x, l_z)

        # the final multiplication by mask in attention is not required
        # but it makes the final attention tensor more explicit
        attention = torch.softmax(masked_score, dim=-1) * emask
        assert attention.shape == (b, self.n_h, l_x, l_z)

        # yh [b, h, l_x, d_mid]
        yh = attention @ v
        assert yh.shape == (b, self.n_h, l_x, self.d_mid)

        # y [b, l_x, h * d_mid]
        y = einops.rearrange(yh, "b h l d -> b l (h d)")
        assert y.shape == (b, l_x, self.n_h * self.d_mid)

        # y [b, l_x, h * d_mid],  w_o [h * d_mid, d_out]
        if self.bias:
            vt = torch.einsum("b l k, k d -> b l d", y, self.w_o) + self.b_o
        else:
            vt = torch.einsum("b l k, k d -> b l d", y, self.w_o)

        assert vt.shape == (b, l_x, self.d_out)

        return {
            "q": q,
            "k": k,
            "v": v,
            "score": score,
            "mask": mask,
            "bmask": bmask,
            "masked_score": masked_score,
            "yh": yh,
            "y": y,
            "attention": attention,
            "vt": vt,
        }


    def extra_repr(self):
        return "d_x={}, d_z={}, d_out={}, d_attn={}, d_mid={}, n_h={}, bias={}".format(
            self.d_x, self.d_z, self.d_out, self.d_attn, self.d_mid, self.n_h, self.bias)
