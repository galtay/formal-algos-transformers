import math
from types import SimpleNamespace
import unittest

import torch

from formal_algos_transformers.embeddings import ContentEmbeddings
from formal_algos_transformers.embeddings import PositionEmbeddings
from formal_algos_transformers.embeddings import PositionEncodings


class TestEmbeddings(unittest.TestCase):

    def setUp(self):
        self.config = SimpleNamespace(
            n_v = 4,
            d_e = 8,
            l_max = 3,
        )

    def test_content_embeddings(self):

        ce = ContentEmbeddings(self.config.n_v, self.config.d_e)
        input_ids = torch.tensor([
            [0, 1, 0],
            [3, 2, 1],
        ])
        b, s = input_ids.shape
        embeddings = ce(input_ids)
        self.assertTrue(embeddings.shape == (b, s, self.config.d_e))
        for b in range(input_ids.shape[0]):
            for t in range(input_ids.shape[1]):
                input_id = input_ids[b,t]
                self.assertTrue(all(
                    embeddings[b, t, :] == ce.embedding.weight[input_id]
                ))


    def test_position_embeddings(self):

        pe = PositionEmbeddings(self.config.l_max, self.config.d_e)
        input_ids = torch.tensor([
            [0, 0, 0],
            [0, 0, 0],
        ])
        b, s = input_ids.shape
        embeddings = pe(input_ids)
        self.assertTrue(embeddings.shape == (1, s, self.config.d_e))
        for t in range(input_ids.shape[1]):
            self.assertTrue(all(
                embeddings[0, t, :] == pe.embedding.weight[t]
            ))


    def test_position_encodings(self):

        pe = PositionEncodings(self.config.l_max, self.config.d_e)
        input_ids = torch.tensor([
            [0, 0, 0],
            [0, 0, 0],
        ])
        b, s = input_ids.shape
        embeddings = pe(input_ids)
        self.assertTrue(embeddings.shape == (1, s, self.config.d_e))
        for t in range(input_ids.shape[1]):
            self.assertTrue(all(
                embeddings[0, t, :] == pe.pos_enc[0, t, :]
            ))
