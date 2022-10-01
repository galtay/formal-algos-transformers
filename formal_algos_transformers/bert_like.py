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

    """Transform vocab input ids to embeddings.

    Note: formal algos does not include layernorm or dropout.

    Args:
        content (ContentEmbeddings): content token embeddings
        position (PositionEncodings): position token embeddings
        do_layer_norm (bool): do layer norm
        dropout_proba (float): dropout probability

    Input:
        input_ids (tensor) [b, l]: input ids

    Output:
        out (tensor) [b, l, d]: token embeddings
    """

    def __init__(
        self,
        content: ContentEmbeddings,
        position: PositionEncodings,
        do_layer_norm: bool = True,
        dropout_proba: float = 0.1,
    ):
        """Combined content and position encodings"""
        super().__init__()
        self.content = content
        self.position = position
        self.do_layer_norm = do_layer_norm
        self.dropout_proba = dropout_proba
        if do_layer_norm:
            self.norm = nn.LayerNorm(content.d_e)
        self.drop = nn.Dropout(dropout_proba)

    def forward(self, input_ids):
        embeddings = self.content(input_ids) + self.position(input_ids)
        if self.do_layer_norm:
            embeddings = self.norm(embeddings)
        embeddings = self.drop(embeddings)
        return embeddings


class PointwiseFeedForward(nn.Module):

    """Apply a multilayer perceptron to each token.

    Args:
        d_ff (int): size of feed forward layer
        d_out (int): size of MHA output
        dropout_proba (float): dropout probability

    Input:
        x (tensor) [b, l_x, d_out]: MHA token embeddings

    Output:
        out (tensor) [b, l_x, d_out]: token embeddings after feed forward
    """

    def __init__(self, d_ff: int, d_out: int, dropout_proba=0.1):

        super().__init__()
        self.d_ff = d_ff
        self.d_out = d_out
        self.dropout_proba = dropout_proba
        self.ff = nn.Sequential(
            nn.Linear(d_out, d_ff),
            nn.GELU(),
            nn.Dropout(dropout_proba),
            nn.Linear(d_ff, d_out),
        )

    def forward(self, x):
        return self.ff(x)


class EncoderBlock(nn.Module):

    """Apply an encoder only transformer block.

    Args:
        mha (nn.Module): multi-head attention module
        ff (nn.Module): pointwise feed forward module
        prenorm (bool): if True, out = x + Module[LN(x)]
            else out = LN[x + Module(x)]
        mha_dropout_proba (float): dropout probability for multihead attention
        ff_dropout_proba (float): dropout probability for feed forward

    Input:
        x (tensor) [b, l_x, d_x|d_out]: token embeddings of primary sequence
        mask (tensor) [b, l_x, l_x]: attention mask (1=attend, 0=dont)

    Output:
        out (tensor) [b, l_x, d_out]: token hidden states
    """

    def __init__(
        self,
        d_ff: int,
        mha: MultiHeadAttention,
        ff: PointwiseFeedForward,
        prenorm=True,
        mha_dropout_proba=0.1,
        ff_dropout_proba=0.1,
    ):

        super().__init__()
        self.mha = mha
        self.ff = ff

        self.prenorm = prenorm
        self.mha_dropout_proba = mha_dropout_proba
        self.ff_dropout_proba = ff_dropout_proba
        self.d_out = mha.d_out

        self.norm_mha = nn.LayerNorm(mha.d_out)
        self.norm_ff = nn.LayerNorm(mha.d_out)
        self.drop_mha = nn.Dropout(mha_dropout_proba)
        self.drop_ff = nn.Dropout(ff_dropout_proba)

    def forward(self, x, mask):
        # pre-norm
        if self.prenorm:
            x_norm = self.norm_mha(x)
            out = x + self.drop_mha(self.mha(x_norm, x_norm, mask)["vtilde"])
            out = out + self.drop_ff(self.norm_ff(self.ff(out)))
        # post-norm
        else:
            out = self.drop_mha(self.norm_mha(x + self.mha(x, x, mask)["vtilde"]))
            out = self.drop_ff(self.norm_ff(out + self.ff(out)))
        return out


class EncoderStack(nn.Module):

    """Apply a stack of encoder blocks to token embeddings.

    Args:
        encoder_blocks (nn.ModuleList): a stack of encoder blocks

    Input:
        x (tensor) [b, l_x, d_x|d_out]: token embeddings of primary sequence
        mask (tensor) [b, l_x, l_x]: attention mask (1=attend, 0=dont)

    Output:
        out (tensor) [b, l_x, d_out]: token hidden states
    """

    def __init__(self, encoder_blocks: nn.ModuleList):
        super().__init__()
        self.encoder_blocks = encoder_blocks
        self.norm = nn.LayerNorm(encoder_blocks[-1].d_out)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, mask)
        return self.norm(x)


class EncoderHeadless(nn.Module):

    """Apply a stack of encoder blocks to input ids.

    Args:
        embeddings (Embeddings): content and position embeddings
        encoder_blocks (nn.ModuleList): a stack of encoder blocks

    Input:
        x_input_ids (tensor) [b, l_x]: input_ids of primary sequences
        mask (tensor) [b, l_x, l_x]: attention mask (1=attend, 0=dont)

    Output:
        out (tensor) [b, l_x, d_out]: token hidden states
    """

    def __init__(
        self, embeddings: Embeddings, encoder_stack: EncoderStack,
    ):
        super().__init__()
        self.embeddings = embeddings
        self.encoder_stack = encoder_stack

    def forward(self, x_input_ids, mask):
        "Input token IDs to last hidden states"
        x = self.embeddings(x_input_ids)
        x = self.encoder_stack(x, mask)
        return x


class EncoderMlmHead(nn.Module):

    """Apply a masked language modeling head to encoder hidden states.

    Args:
        d_out (int): size of each token hidden state
        n_v (int): size of vocabulary

    Input:
        x (tensor) [b, l_x, d_out]: token hidden states

    Output:
        out (tensor) [b, l_x, n_v]: pre softmax vocab logits
    """

    def __init__(self, d_out: int, n_v: int):
        super().__init__()
        self.d_out = d_out
        self.n_v = n_v

        self.linear_w_f = nn.Sequential(nn.Linear(d_out, d_out), nn.GELU())
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


class EncoderForMlm(nn.Module):
    def __init__(
        self, encoder_headless: EncoderHeadless, mlm_head: EncoderMlmHead,
    ):
        super().__init__()
        self.encoder_headless = encoder_headless
        self.mlm_head = mlm_head

    def forward(self, x_input_ids, x_mask):
        "Input token IDs to last hidden state"
        x = self.encoder_headless(x_input_ids, x_mask)
        x = self.mlm_head(x)
        return x


def make_bert_base_encoder():

    embd_size = 768
    d_x = embd_size
    d_z = embd_size
    d_out = embd_size

    embd_do_layer_norm = True
    embd_dropout_proba = 0.1

    mha_attn_dropout_proba = 0.1
    ff_internal_dropout_proba = 0.1

    mha_sub_dropout_proba = 0.1
    ff_sub_dropout_proba = 0.1

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
    embeddings = Embeddings(
        content_embeddings,
        position_encodings,
        do_layer_norm=embd_do_layer_norm,
        dropout_proba=embd_dropout_proba,
    )

    encoder_blocks = nn.ModuleList(
        [
            EncoderBlock(
                d_ff,
                MultiHeadAttention(
                    d_x,
                    d_z,
                    d_attn,
                    d_mid,
                    n_h,
                    d_out,
                    bias,
                    dropout_proba=mha_attn_dropout_proba,
                ),
                PointwiseFeedForward(
                    d_ff, d_out, dropout_proba=ff_internal_dropout_proba
                ),
                prenorm=prenorm,
                mha_dropout_proba=mha_sub_dropout_proba,
                ff_dropout_proba=ff_sub_dropout_proba,
            )
            for _ in range(n_layers)
        ]
    )
    encoder_stack = EncoderStack(encoder_blocks)
    encoder_headless = EncoderHeadless(embeddings, encoder_stack)
    encoder_mlm_head = EncoderMlmHead(d_out, n_v)
    encoder_for_mlm = EncoderForMlm(encoder_headless, encoder_mlm_head)

    return encoder_for_mlm


if __name__ == "__main__":

    encoder_for_mlm = make_bert_base_encoder()
    input_ids = torch.tensor([[4, 900, 72, 0, 0], [9287, 12, 726, 23107, 82],])
    mask_1d = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1],])
    mask = get_pad_mask(mask_1d, mask_1d)
    mlm_logits = encoder_for_mlm(input_ids, mask)
