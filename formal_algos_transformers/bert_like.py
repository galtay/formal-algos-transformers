"""

"""
import math
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn

from formal_algos_transformers.fat_multi_head_attention import MultiHeadAttention
from formal_algos_transformers.embeddings import ContentEmbeddings
from formal_algos_transformers.embeddings import PositionEncodings


class Embeddings(nn.Module):

    def __init__(self, content: ContentEmbeddings, position: PositionEncodings):
        """Combined content and position encodings"""
        super().__init__()
        self.content = content
        self.position = position

    def forward(self, input_ids):
        return self.content(input_ids) + self.position(input_ids)


class EncoderBlock(nn.Module):

    def __init__(self, d_ff: int, mha: MultiHeadAttention):

        """Apply an encoder only transformer block.

        Note: this implementation does not contain dropout

        Args:
            d_ff (int): size of ff layer
            mha (nn.Module): multi-head attention module
        """
        super().__init__()
        self.d_ff = d_ff
        self.mha = mha
        self.d_out = mha.d_out

        self.norm_1 = nn.LayerNorm(mha.d_out)
        self.norm_2 = nn.LayerNorm(mha.d_out)
        self.linear = nn.Sequential(
            nn.Linear(mha.d_out, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, mha.d_out),
        )

    def forward(self, x, x_mask):
        out = x + self.mha(x, x, x_mask, x_mask)["vt"]
        out = self.norm_1(out)
        out = out + self.linear(out)
        out = self.norm_2(out)
        return out


class Encoder(nn.Module):

    def __init__(self, encoder_blocks: nn.ModuleList):
        super().__init__()
        self.encoder_blocks = encoder_blocks

    def forward(self, x, x_mask):
        "Pass the input (and mask) through each layer in turn."
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, x_mask)
        return x


class EncoderTransformer(nn.Module):

    def __init__(
        self,
        embeddings: Embeddings,
        encoder: Encoder,
    ):
        super().__init__()
        self.embeddings = embeddings
        self.encoder = encoder

    def forward(self, x_input_ids, x_mask):
        "Input token IDs to last hidden states"
        x = self.embeddings(x_input_ids)
        x = self.encoder(x, x_mask)
        return x


class EncoderMlmHead(nn.Module):

    def __init__(self, d_out: int, n_v: int):

        """Returns language modeling prediction logits"""

        super().__init__()
        self.d_out = d_out
        self.n_v = n_v

        self.linear_w_f = nn.Sequential(
            nn.Linear(d_out, d_out),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(d_out)
        self.linear_w_u = nn.Linear(d_out, n_v)

    def forward(self, x):
        x = self.linear_w_f(x)
        x = self.norm(x)
        x = self.linear_w_u(x)
        # the softmax is in the psuedo code
        # but its better to bundle it with the CrossEntropyLoss
        # x = torch.softmax(x, dim=-1)
        return x


class EncoderTransformerMlm(nn.Module):

    def __init__(
        self,
        embeddings: Embeddings,
        encoder: Encoder,
        mlm_head: EncoderMlmHead,
    ):
        super().__init__()
        self.embeddings = embeddings
        self.encoder = encoder
        self.mlm_head = mlm_head

    def forward(self, x_input_ids, x_mask):
        "Input token IDs to last hidden state"
        x = self.embeddings(x_input_ids)
        x = self.encoder(x, x_mask)
        x = self.mlm_head(x)
        return x


def make_bert_base_encoder_transformer():

    embd_size = 768
    d_x = embd_size
    d_z = embd_size
    d_out = embd_size

    n_h = 12
    n_layers = 12

    d_attn = embd_size // n_h
    d_mid = embd_size // n_h
    bias = True
    d_ff = 3_072

    n_v = 30_522
    l_max = 512

    content_embeddings = ContentEmbeddings(n_v, embd_size)
    position_encodings = PositionEncodings(l_max, embd_size)
    embeddings = Embeddings(content_embeddings, position_encodings)

    encoder_blocks = nn.ModuleList([
        EncoderBlock(
            d_ff,
            MultiHeadAttention(d_x, d_z, d_out, d_attn, d_mid, n_h, bias),
        ) for _ in range(n_layers)
    ])
    encoder = Encoder(encoder_blocks)

    encoder_transformer = EncoderTransformer(embeddings, encoder)

    return encoder_transformer


if __name__ == "__main__":

    model = make_bert_base_encoder_transformer()
    input_ids = torch.tensor([
        [4, 900, 72, 0, 0],
        [9287, 12, 726, 23107, 82],
    ])
    x_mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
    ])
    out = model(input_ids, x_mask)
