import math
import torch
import torch.nn as nn
from einops import repeat


# ============================
# General Utilities
# ============================
def _getchk(tch_tensor, s:int, t:int, batch_size:int):
    if tch_tensor is None:
        return None
    if tch_tensor.size(0) == batch_size:
        return tch_tensor[s:t]
    else:
        n_folds = batch_size // tch_tensor.size(0)
        return repeat(tch_tensor, 'b ... -> (b n) ...', n=n_folds)[s:t]  # [N, ...]


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


# ============================
# FACTS Embedding Modules
# ============================
# >> Basic Embedder <<
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [B, T, D]
        self.register_buffer('pe', pe)

    def forward(self, start=0, end=None, ids: list=None):
        """Forward function for the positional embedding."""
        if ids is not None:
            return self.pe[:, ids]
        else:
            return self.pe[:, start:end]
