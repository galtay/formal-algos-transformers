import math
from types import SimpleNamespace
import unittest

import torch
from torch import Tensor
from torch.nn.utils.stateless import functional_call

from formal_algos_transformers.fat_single_query_attention import SingleQueryAttention
from .utils import allclose


CHECK_KEYS = ["vt", "attention", "q", "k", "v", "score", "mask", "bmask", "masked_score"]


def single_query_attention_gold(
    x: Tensor,
    zs: Tensor,
    x_mask: Tensor,
    zs_mask: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Tensor,
    b_k: Tensor,
    b_v: Tensor,
):

    """This is the foundation code we have to trust for tests to be valid

    Args:
        x (tensor): [d_x] single token embedding to be contextualized
        zs (tensor): [l_z, d_z] sequence of context token embeddings
        x_mask (tensor): [] rank zero tensor mask i.e. torch.tensor(0) or torch.tensor(1)
        zs_mask (tensor): [l_z] context attention mask
        w_q (tensor): [d_x, d_attn] query weight tensor
        w_k (tensor): [d_z, d_attn] key weight tensor
        w_v (tensor): [d_z, d_out] value weight tensor
        b_q (tensor): [d_attn] query bias tensor
        b_k (tensor): [d_attn] key projection tensor
        b_v (tensor): [d_out] value projection tensor

    Output:
        q (tensor): [1, d_attn] query vector for x
        k (tensor): [l_z, d_attn] key vectors for zs
        v (tensor): [l_z, d_out] value vectors for zs
        score (tensor): [l_z] (q k^T) / sqrt(d_attn)
        mask (tensor): [l_z] mask[i] = 0 if x_mask is 0 or zs_mask[i] = 0 else 1
        bmask (tensor): [l_z] bmask[i] = False if x_mask is 0  or zs_mask[i] = 0 else True
        masked_score (tensor): [l_z] masked_score[i] = score[i] where bmask[i] = 1
            else minimum value for score tensor dtype

        attention (tensor): [l_z] attention weights. attention[i] is explicitly set to 0
            if either x mask is 0 or zs_mask[i] is 0
        vt (tensor): [d_out] updated representation of the token embedding x, folding in
            information from tokens in zs

    """

    # check all the shapes

    assert len(x.shape) == 1
    assert len(zs.shape) == 2
    assert len(x_mask.shape) == 0
    assert len(zs_mask.shape) == 1
    assert len(w_q.shape) == len(w_k.shape) == len(w_v.shape) == 2
    assert len(b_q.shape) == len(b_k.shape) == len(b_v.shape) == 1

    (d_x,) = x.shape
    (l_z, d_z) = zs.shape

    assert x_mask.shape == ()
    assert zs_mask.shape == (l_z,)

    assert w_q.shape[0] == d_x
    d_attn = w_q.shape[1]
    assert w_k.shape == (d_z, d_attn)
    assert w_v.shape[0] == d_z
    d_out = w_v.shape[1]

    assert b_q.shape[0] == d_attn
    assert b_k.shape[0] == d_attn
    assert b_v.shape[0] == d_out

    # lets make the row vector nature of x more explicit
    xrow = x[None, :]
    assert xrow.shape == (1, d_x)

    # q = xrow @ w_q + b_q: [1, d_x] @ [d_x, d_attn] + [d_attn] = [1, d_attn]
    # k = zs @ w_k + b_k: [l_z, d_z] @ [d_z, d_attn] + [d_attn] = [l_z, d_attn]
    # v = zs @ w_v + b_v: [l_z, d_z] @ [d_z, d_out] + [d_out] = [l_z, d_out]
    q = xrow @ w_q + b_q
    k = zs @ w_k + b_k
    v = zs @ w_v + b_v

    # score_row = (q @ k.T) / sqrt(d_attn): [1, d_attn] @ [d_attn, l_z] = [1, l_z]
    score_row = q @ k.T / math.sqrt(d_attn)
    assert score_row.shape == (1, l_z)

    score = torch.squeeze(score_row, dim=0)
    assert score.shape == (l_z,)

    mask = x_mask * zs_mask
    bmask = mask.to(torch.bool)
    assert mask.shape == bmask.shape == (l_z,)

    masked_score = score.masked_fill(~bmask, torch.finfo(score.dtype).min)
    attention = torch.softmax(masked_score, dim=-1) * mask
    assert masked_score.shape == attention.shape == (l_z,)

    # each row of v is a token embedding representation
    # each value in attention is a weight
    # vt = attention[0] * v[0,:] + attention[1] * v[1,:] + ...
    vt = torch.zeros(d_out)
    for tok in range(l_z):
        vt += attention[tok] * v[tok, :]
    assert vt.shape == (d_out,)

    return {
        "vt": vt,
        "attention": attention,
        "q": q,
        "k": k,
        "v": v,
        "score": score,
        "mask": mask,
        "bmask": bmask,
        "masked_score": masked_score,
    }


class TestSingleQueryAttention(unittest.TestCase):

    """Test SingleQueryAttention

    This is the foundation of the testing ladder.
    We don't have ground truth for this output, but if you
    trust this implementation then you can trust the implementations
    that depend on it.
    """

    def setUp(self):
        self.config = SimpleNamespace(
            l_z = 4,
            d_x = 5,
            d_z = 6,
            d_out = 7,
            d_attn = 8,
        )

    def test_output(self):

        w_q = torch.randn(self.config.d_x, self.config.d_attn)
        w_k = torch.randn(self.config.d_z, self.config.d_attn)
        w_v = torch.randn(self.config.d_z, self.config.d_out)

        b_q = torch.randn(self.config.d_attn)
        b_k = torch.randn(self.config.d_attn)
        b_v = torch.randn(self.config.d_out)

        x = torch.randn(self.config.d_x)
        zs = torch.randn(self.config.l_z, self.config.d_z)

        # single token x_mask can only be 0 or 1
        x_masks = [
            torch.tensor(0, dtype=torch.int64),
            torch.tensor(1, dtype=torch.int64),
        ]

        # create a few zs_masks
        zs_masks = [
            torch.tensor([1,1,1,1], dtype=torch.int64),
            torch.tensor([1,1,1,1], dtype=torch.int32),
            torch.tensor([1,1,1,0], dtype=torch.int32),
            torch.tensor([1,1,0,0], dtype=torch.int32),
            torch.tensor([1,0,1,0], dtype=torch.int32),
            torch.tensor([0,1,0,0], dtype=torch.int32),
        ]

        for bias in [True, False]:
            for x_mask in x_masks:
                for zs_mask in zs_masks:

                    single_query_attention = SingleQueryAttention(
                        d_x = self.config.d_x,
                        d_z = self.config.d_z,
                        d_out = self.config.d_out,
                        d_attn = self.config.d_attn,
                        bias = bias,
                    )

                    # multiply biases by 0 if bias = False
                    bias_mult = int(bias)
                    expected_output = single_query_attention_gold(
                        x,
                        zs,
                        x_mask,
                        zs_mask,
                        w_q,
                        w_k,
                        w_v,
                        b_q * bias_mult,
                        b_k * bias_mult,
                        b_v * bias_mult,
                    )

                    # pass input thru single_query_attention with set weights and biases
                    params_and_buffers = {
                        "w_q": w_q, "w_k": w_k, "w_v": w_v,
                        "b_q": b_q, "b_k": b_k, "b_v": b_v,
                    }

                    actual_output = functional_call(
                        single_query_attention,
                        params_and_buffers,
                        (x, zs, x_mask, zs_mask),
                    )

                    for key in CHECK_KEYS:
                        self.assertTrue(allclose(
                            expected_output[key], actual_output[key]))
