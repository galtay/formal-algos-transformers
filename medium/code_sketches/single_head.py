class SingleHeadAttention(nn.Module):

    def __init__(self, d_x: int, d_z: int, d_attn: int, d_out: int):

        self.w_q = nn.Parameter(torch.empty(d_x, d_attn))
        self.w_k = nn.Parameter(torch.empty(d_z, d_attn))
        self.w_v = nn.Parameter(torch.empty(d_z, d_out))
        self.b_q = nn.Parameter(torch.empty(d_attn))
        self.b_k = nn.Parameter(torch.empty(d_attn))
        self.b_v = nn.Parameter(torch.empty(d_out))

    def forward(self, x: Tensor, z: Tensor, x_mask: Tensor, z_mask: Tensor):

        b_x, l_x, d_x = x.shape
        b_z, l_z, d_z = z.shape
        assert b_x == b_z; b = b_x
        assert x_mask.shape == (b, l_x)
        assert z_mask.shape == (b, l_z)

        einsum_str = "b i k, k j -> b i j"
        q = torch.einsum(einsum_str, x, self.w_q) + self.b_q
        k = torch.einsum(einsum_str, z, self.w_k) + self.b_k
        v = torch.einsum(einsum_str, z, self.w_v) + self.b_v

        assert q.shape == (b, l_x, self.d_attn)
        assert k.shape == (b, l_z, self.d_attn)
        assert v.shape == (b, l_z, self.d_out)

        # combine and expand x_mask [b, l_x] and z_mask [b, l_z]
        # [b, l_x, 1] @ [b, 1, l_z] = [b, l_x, l_z]
        mask = x_mask[:, :, None] @ z_mask[:, None, :]
        bmask = mask.to(torch.bool)

        einsum_str = "b i k, b j k -> b i j"
        score = torch.einsum(einsum_str, q, k) / math.sqrt(self.d_attn)
        score = score.masked_fill(~bmask, torch.finfo(score.dtype).min)

        # multiplying by mask below is not required but ensures
        # attention is 0 where mask is 0
        attention = torch.softmax(score, dim=-1) * mask
        assert mask.shape == score.shape == attention.shape == (b, l_x, l_z)

        vtilde = attention @ v
        assert vtilde.shape == (b, l_x, self.d_out)
