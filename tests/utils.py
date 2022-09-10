import torch

ATOL = 1e-4
RTOL = 1e-4

def allclose(x, y, atol=ATOL, rtol=RTOL):
    """Wraps torch.allclose with new default tolerances.

    https://pytorch.org/docs/stable/generated/torch.allclose.html
    """
    return torch.allclose(x, y, atol=atol, rtol=rtol)
