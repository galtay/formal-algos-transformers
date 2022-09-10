import math
from types import SimpleNamespace
import unittest

import torch
from torch import Tensor
from torch.nn.utils.stateless import functional_call

from formal_algos_transformers.fat_single_query_attention import SingleQueryAttention
from formal_algos_transformers.fat_single_head_attention import SingleHeadAttention
from .utils import allclose


class TestSingleHeadAttention(unittest.TestCase):

    def setUp(self):
        self.config = SimpleNamespace(
            b = 2,
            l_x = 3,
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

        x = torch.randn(self.config.b, self.config.l_x, self.config.d_x)
        z = torch.randn(self.config.b, self.config.l_z, self.config.d_z)

        # create masks by hand
        x_masks = [
            torch.tensor([
                [1,1,1],
                [1,1,1],
            ], dtype=torch.int64),
            torch.tensor([
                [1,1,1],
                [1,1,0],
            ], dtype=torch.int64)
        ]

        z_masks = [
            torch.tensor([
                [1,1,1,1],
                [1,1,1,1],
            ], dtype=torch.int64),
            torch.tensor([
                [1,1,0,0],
                [1,1,1,0],
            ], dtype=torch.int64)
        ]

        for bias in [True, False]:
            for x_mask in x_masks:
                for z_mask in z_masks:

                    # we will use this as expected output
                    single_query_attention = SingleQueryAttention(
                        d_x = self.config.d_x,
                        d_z = self.config.d_z,
                        d_out = self.config.d_out,
                        d_attn = self.config.d_attn,
                        bias = bias,
                    )

                    # compare actual output of this to expected output
                    attention = SingleHeadAttention(
                        d_x = self.config.d_x,
                        d_z = self.config.d_z,
                        d_out = self.config.d_out,
                        d_attn = self.config.d_attn,
                        bias = bias,
                    )

                    params_and_buffers = {
                        "w_q": w_q, "w_k": w_k, "w_v": w_v,
                        "b_q": b_q, "b_k": b_k, "b_v": b_v,
                    }
                    actual_output = functional_call(
                        attention,
                        params_and_buffers,
                        (x, z, x_mask, z_mask),
                    )

                    for batch in range(self.config.b):
                        for tok in range(self.config.l_x):

                            # get single query attention results

                            x1 = x[batch, tok, :]
                            zs = z[batch, :, :]
                            expected_output = functional_call(
                                single_query_attention,
                                params_and_buffers,
                                (x1, zs, x_mask[batch, tok], z_mask[batch, :]),
                            )

                            self.assertTrue(allclose(
                                expected_output["k"],
                                actual_output["k"][batch, :, :]))

                            self.assertTrue(allclose(
                                expected_output["v"],
                                actual_output["v"][batch, :, :]))

                            for check_key in ["q", "score", "mask", "bmask", "masked_score", "attention", "vt"]:

                                self.assertTrue(allclose(
                                    expected_output[check_key],
                                    actual_output[check_key][batch, tok, :]))
