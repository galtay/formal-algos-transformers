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
        masks = [
            torch.tensor([[
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],[
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]]),
            torch.tensor([[
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
            ],[
                [1, 1, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 0],
            ]]),
        ]


        for bias in [True, False]:
            for mask in masks:

                # we will use this as expected output
                single_query_attention = SingleQueryAttention(
                    d_x = self.config.d_x,
                    d_z = self.config.d_z,
                    d_attn = self.config.d_attn,
                    d_out = self.config.d_out,
                    bias = bias,
                )

                # compare actual output of this to expected output
                single_head_attention = SingleHeadAttention(
                    d_x = self.config.d_x,
                    d_z = self.config.d_z,
                    d_attn = self.config.d_attn,
                    d_out = self.config.d_out,
                    bias = bias,
                )

                params_and_buffers = {
                    "w_q": w_q, "w_k": w_k, "w_v": w_v,
                    "b_q": b_q, "b_k": b_k, "b_v": b_v,
                }
                actual_output = functional_call(
                    single_head_attention,
                    params_and_buffers,
                    (x, z, mask),
                )

                for batch in range(self.config.b):
                    for tok in range(self.config.l_x):

                        # get single query attention results

                        x1 = x[batch, tok, :]
                        zb = z[batch, :, :]
                        expected_output = functional_call(
                            single_query_attention,
                            params_and_buffers,
                            (x1, zb, mask[batch, tok, :]),
                        )

                        self.assertTrue(allclose(
                            expected_output["k"],
                            actual_output["k"][batch, :, :]))

                        self.assertTrue(allclose(
                            expected_output["v"],
                            actual_output["v"][batch, :, :]))

                        for check_key in ["q", "score", "attention", "vtilde"]:

                            self.assertTrue(allclose(
                                expected_output[check_key],
                                actual_output[check_key][batch, tok, :]))
