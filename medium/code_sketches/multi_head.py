class MultiHeadAttention(nn.Module):

    def __init__(self, d_x: int, d_z: int, d_attn: int, d_mid: int, n_h: int, d_out: int):

        self.w_q = nn.Parameter(torch.empty(n_h, d_x, d_attn))
        self.w_k = nn.Parameter(torch.empty(n_h, d_z, d_attn))
        self.w_v = nn.Parameter(torch.empty(n_h, d_z, d_mid))
        self.w_o = nn.Parameter(torch.empty(n_h * d_mid, d_out))
        self.b_q = nn.Parameter(torch.empty(n_h, d_attn))
        self.b_k = nn.Parameter(torch.empty(n_h, d_attn))
        self.b_v = nn.Parameter(torch.empty(n_h, d_mid))
        self.b_o = nn.Parameter(torch.empty(d_out))

    def forward(self, x: Tensor, z: Tensor, x_mask: Tensor, z_mask: Tensor):

        b_x, l_x, d_x = x.shape
        b_z, l_z, d_z = z.shape
        assert b_x == b_z; b = b_x
        assert x_mask.shape == (b, l_x)
        assert z_mask.shape == (b, l_z)

        einsum_str = "b i k, h k j -> b h i j"
        q = torch.einsum(einsum_str, x, self.w_q) + self.b_q[None, :, None, :]
        k = torch.einsum(einsum_str, z, self.w_k) + self.b_k[None, :, None, :]
        v = torch.einsum(einsum_str, z, self.w_v) + self.b_v[None, :, None, :]

        assert q.shape == (b, self.n_h, l_x, self.d_attn)
        assert k.shape == (b, self.n_h, l_z, self.d_attn)
        assert v.shape == (b, self.n_h, l_z, self.d_mid)

        # combine and expand x_mask [b, l_x] and z_mask [b, l_z]
        # [b, l_x, 1] @ [b, 1, l_z] = [b, l_x, l_z]
        mask = x_mask[:, :, None] @ z_mask[:, None, :]
        assert mask.shape == (b, l_x, l_z)

        # create [b, 1, l_x, l_z] which is broadcastable to [b, h, l_x, l_z]
        emask = mask[:, None, :, :]
        bmask = emask.to(torch.bool)
        assert emask.shape == bmask.shape == (b, 1, l_x, l_z)

        einsum_str = "b h i k, b h j k -> b h i j", q, k
        score = torch.einsum(einsum_str, q, k) * self.scale
        score = score.masked_fill(~bmask, torch.finfo(score.dtype).min)
        attention = torch.softmax(score, dim=-1) * emask
        assert score.shape == attention.shape == (b, self.n_h, l_x, l_z)

        yh = torch.matmul(attention, v)
        assert yh.shape == (b, self.n_h, l_x, self.d_mid)

        y = einops.rearrange(yh, "b h l d -> b l (h d)")
        assert y.shape == (b, l_x, self.n_h * self.d_mid)

        vtilde = torch.einsum("b l k, k d -> b l d", y, self.w_o) + self.b_o
        assert vtilde.shape == (b, l_x, self.d_out)
