import math
from types import SimpleNamespace
from typing import List
import unittest

import torch
from torch import Tensor
from torch.nn.utils.stateless import functional_call

from formal_algos_transformers.fat_single_query_attention import SingleQueryAttention
from .utils import allclose


CHECK_KEYS = ["q", "k", "v", "score", "mask", "attention", "vtilde"]


def single_query_attention_gold(
    x1: Tensor,
    z: List[Tensor],
    x1_mask: Tensor,
    z_mask: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Tensor,
    b_k: Tensor,
    b_v: Tensor,
):

    """This is the foundation code we have to trust for tests to be valid

    Args:
        x1 (tensor) [d_x]: single token embedding to be contextualized
        z (List[tensor]) list of [d_z]: sequence of context token embeddings
        x1_mask (tensor) []: torch.tensor(0) or torch.tensor(1)
        z_mask (tensor) [l_z]: context attention mask

        w_q (tensor) [d_x, d_attn]: query weight tensor
        w_k (tensor) [d_z, d_attn]: key weight tensor
        w_v (tensor) [d_z, d_out]: value weight tensor

        b_q (tensor) [d_attn]: query bias tensor
        b_k (tensor) [d_attn]: key projection tensor
        b_v (tensor) [d_out]: value projection tensor


    Output:
        q (tensor) [d_attn]: query vector for x1
        k (tensor) [l_z, d_attn]: key vectors for z
        v (tensor) [l_z, d_out]: value vectors for z
        mask (tensor): [l_z] mask[i] = 0 if x1_mask is 0 or z_mask[i] = 0 else 1
        score (tensor) [l_z]: score = (q @ k^T) / sqrt(d_attn)
            score[i] = score[i] where mask[i] = 1
            else minimum value for score tensor dtype
        attention (tensor) [l_z]: attention weights
            explicitly set to 0 where mask = 0
        vtilde (tensor) [d_out]: contextualized representation of x1

    """

    # check all the shapes

    assert x1.dim() == 1
    (d_x,) = x1.shape

    l_z = len(z)
    assert all([z1.dim() == 1 for z1 in z])
    (d_z,) = z[0].shape
    assert all([z1.shape == (d_z,) for z1 in z])

    assert x1_mask.dim() == 0
    assert x1_mask.shape == ()

    assert z_mask.dim() == 1
    assert z_mask.shape == (l_z,)

    assert w_q.dim() == w_k.dim() == w_v.dim() == 2
    assert w_q.shape[0] == d_x
    d_attn = w_q.shape[1]
    assert w_k.shape == (d_z, d_attn)
    assert w_v.shape[0] == d_z
    d_out = w_v.shape[1]

    assert b_q.shape == b_k.shape == (d_attn,)
    assert b_v.shape == (d_out,)

    # https://pytorch.org/docs/stable/generated/torch.matmul.html
    # torch.matmul(input, other, *, out=None) → Tensor
    # If the first argument is 1-dimensional and the second argument is 2-dimensional,
    # a 1 is prepended to its dimension for the purpose of the matrix multiply.
    # After the matrix multiply, the prepended dimension is removed.

    q = torch.matmul(x1, w_q) + b_q
    ks = [torch.matmul(z1, w_k) + b_k for z1 in z]
    vs = [torch.matmul(z1, w_v) + b_v for z1 in z]

    # score = q . ks[t] / sqrt(d_attn) for t in range(l_z)
    score = torch.tensor([
        torch.dot(q, ks[t]) / math.sqrt(d_attn)
        for t in range(l_z)
    ])
    assert score.shape == (l_z,)

    mask = x1_mask * z_mask
    bmask = mask.to(torch.bool)
    assert mask.shape == bmask.shape == (l_z,)

    masked_score = score.masked_fill(~bmask, torch.finfo(score.dtype).min)
    attention = torch.softmax(masked_score, dim=0) * mask
    assert masked_score.shape == attention.shape == (l_z,)

    vtilde = torch.zeros(d_out)
    for tok in range(l_z):
        vtilde += attention[tok] * vs[tok]
    assert vtilde.shape == (d_out,)

    return {
        "q": q,
        "k": torch.stack(ks),
        "v": torch.stack(vs),
        "score": masked_score,
        "mask": mask,
        "attention": attention,
        "vtilde": vtilde,
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

        x1 = torch.randn(self.config.d_x)
        z = [torch.randn(self.config.d_z) for _ in range(self.config.l_z)]

        # single token x_mask can only be 0 or 1
        x1_masks = [
            torch.tensor(0, dtype=torch.int64),
            torch.tensor(1, dtype=torch.int64),
        ]

        # create a few zs_masks
        z_masks = [
            torch.tensor([1,1,1,1], dtype=torch.int64),
            torch.tensor([1,1,1,1], dtype=torch.int32),
            torch.tensor([1,1,1,0], dtype=torch.int32),
            torch.tensor([1,1,0,0], dtype=torch.int32),
            torch.tensor([1,0,1,0], dtype=torch.int32),
            torch.tensor([0,1,0,0], dtype=torch.int32),
        ]

        for bias in [True, False]:
            for x1_mask in x1_masks:
                for z_mask in z_masks:

                    single_query_attention = SingleQueryAttention(
                        d_x = self.config.d_x,
                        d_z = self.config.d_z,
                        d_attn = self.config.d_attn,
                        d_out = self.config.d_out,
                        bias = bias,
                    )

                    # multiply biases by 0 if bias = False
                    bias_mult = int(bias)
                    expected_output = single_query_attention_gold(
                        x1,
                        z,
                        x1_mask,
                        z_mask,
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
                        (x1, torch.stack(z), x1_mask, z_mask),
                    )

                    for key in CHECK_KEYS:
#                        print(key)
#                        print(expected_output[key])
#                        print(actual_output[key])
                        self.assertTrue(allclose(
                            expected_output[key], actual_output[key]))
