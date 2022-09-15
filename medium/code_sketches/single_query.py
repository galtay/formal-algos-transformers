class SingleQueryAttention(nn.Module):

    def __init__(self, d_x: int, d_z: int, d_attn: int, d_out: int):

        self.w_q = nn.Parameter(torch.empty(d_x, d_attn))
        self.w_k = nn.Parameter(torch.empty(d_z, d_attn))
        self.w_v = nn.Parameter(torch.empty(d_z, d_out))
        self.b_q = nn.Parameter(torch.empty(d_attn))
        self.b_k = nn.Parameter(torch.empty(d_attn))
        self.b_v = nn.Parameter(torch.empty(d_out))

    def forward(self, x1: Tensor, z: Tensor):

        (d_x,) = x.shape
        (l_z, d_z) = zs.shape

        q = x1 @ self.w_q + self.b_q
        k = z @ self.w_k + self.b_k
        v = z @ self.w_v + self.b_v

        assert q.shape == (self.d_attn,)
        assert k.shape == (l_z, self.d_attn)
        assert v.shape == (l_z, self.d_out)

        score = q @ k.T / math.sqrt(self.d_attn)
        attention = torch.softmax(score, dim=-1)

        assert score.shape == attention.shape == (l_z,)

        vtilde = torch.zeros(self.d_out)
        for tok in range(l_z):
            vtilde += attention[tok] * v[tok, :]
