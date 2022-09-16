class SingleQueryAttention(nn.Module):

    def __init__(self, d_x: int, d_z: int, d_attn: int, d_out: int):

        self.w_q = nn.Parameter(torch.empty(d_x, d_attn))
        self.w_k = nn.Parameter(torch.empty(d_z, d_attn))
        self.w_v = nn.Parameter(torch.empty(d_z, d_out))
        self.b_q = nn.Parameter(torch.empty(d_attn))
        self.b_k = nn.Parameter(torch.empty(d_attn))
        self.b_v = nn.Parameter(torch.empty(d_out))

    def forward(self, x: Tensor, z: List[Tensor]):

        assert x.shape == (self.d_x,)
        l_z = len(z)
        assert all([zt.shape == (self.d_z,) for zt in z])

        q = torch.matmul(x, self.w_q) + self.b_q
        k = [torch.matmul(zt, self.w_k) + self.b_k for zt in z]
        v = [torch.matmul(zt, self.w_v) + self.b_v for zt in z]

        assert q.shape == (self.d_attn,)
        assert all([kt.shape == (self.d_attn,) for kt in k])
        assert all([vt.shape == (self.d_out,) for vt in v])

        score = torch.tensor([
            torch.dot(q, kt) / math.sqrt(self.d_attn)
            for kt in k
        ])
        attention = torch.softmax(score, dim=-1)
        assert score.shape == attention.shape == (l_z,)

        vtilde = torch.zeros(self.d_out)
        for t in range(l_z):
            vtilde += attention[t] * v[t]
