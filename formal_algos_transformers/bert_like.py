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


def get_pad_mask(x_mask, z_mask):
    """Build 2-D padding mask from two 1-D padding masks.

    Input:
        x_mask (tensor) [b, l_x]: primary sequence attention mask (1=attend, 0=dont)
        z_mask (tensor) [b, l_z]: context sequence attention mask (1=attend, 0=dont)

    Output:
        mask (tensor) [b, l_x, l_z]: attention mask (1=attend, 0=dont)

    """

    # combine and expand x_mask [b, l_x] and z_mask [b, l_z]
    # [b, l_x, 1] @ [b, 1, l_z] = [b, l_x, l_z]
    mask = x_mask[:, :, None] @ z_mask[:, None, :]
    return mask


class Embeddings(nn.Module):

    def __init__(self, content: ContentEmbeddings, position: PositionEncodings):
        """Combined content and position encodings"""
        super().__init__()
        self.content = content
        self.position = position

    def forward(self, input_ids):
        return self.content(input_ids) + self.position(input_ids)


class EncoderBlock(nn.Module):

    """Apply an encoder only transformer block.

    Note: this implementation does not contain dropout

    Args:
        d_ff (int): size of feed forward layer
        mha (nn.Module): multi-head attention module
        prenorm (bool): if True, out = x + Module[LN(x)]
            else out = LN[x + Module(x)]

    Input:
        x (tensor) [b, l_x, d_x|d_out]: token embeddings of primary sequence
        mask (tensor) [b, l_x, l_x]: attention mask (1=attend, 0=dont)

    Output:
        out (tensor) [b, l_x, d_out]: token embeddings after MHA and feed forward
    """

    def __init__(self, d_ff: int, mha: MultiHeadAttention, prenorm=True):

        super().__init__()
        self.d_ff = d_ff
        self.mha = mha
        self.prenorm = prenorm
        self.d_out = mha.d_out

        self.norm_1 = nn.LayerNorm(mha.d_out)
        self.norm_2 = nn.LayerNorm(mha.d_out)
        self.linear = nn.Sequential(
            nn.Linear(mha.d_out, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, mha.d_out),
        )

    def forward(self, x, mask):
        if self.prenorm:
            out = x + self.norm_1(self.mha(x, x, mask)["vtilde"])
            out = out + self.norm_2(self.linear(out))
        else:

            out = self.norm_1(x + self.mha(x, x, mask)["vtilde"])
            out = self.norm_2(out + self.linear(out))
        return out


class Encoder(nn.Module):

    def __init__(self, encoder_blocks: nn.ModuleList):
        super().__init__()
        self.encoder_blocks = encoder_blocks

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, mask)
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

    def forward(self, x_input_ids, mask):
        "Input token IDs to last hidden states"
        x = self.embeddings(x_input_ids)
        x = self.encoder(x, mask)
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
    prenorm = False

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
            MultiHeadAttention(d_x, d_z, d_attn, d_mid, n_h, d_out, bias),
            prenorm=prenorm,
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
    mask_1d = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
    ])
    mask = get_pad_mask(mask_1d, mask_1d)
    out = model(input_ids, mask)
