"""
Multivariate Time Series (MTS) Modelling Modules for FACTS.

Any use of this code should cite the following paper:
    FACTS: A Factored State-Space Framework For World Modelling

Author: Nanbo Li (linanbo2008@gmail.com)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from facts_ssm.utils import RMSNorm
from typing import Dict, List


# ============================
# FACTS Encoding Modules
# ============================
# >> Learnable Sequence Decomposition <<
class Conv2dDecomp(nn.Module):
    """
    Learnable decomposition block
    """
    def __init__(self, in_channels, out_channels, kernel_size=24, stride=1):
        super(Conv2dDecomp, self).__init__()
        self.look_back = kernel_size
        self.ld_conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=(kernel_size, 1),
            bias=False
        )
        self.stack_dim=out_channels

    def forward(self, x, trend_stack=False):
        # x: [B, T, C]
        x = F.pad(x, (0,0,self.look_back-1,0), value=0).unsqueeze(1)  # [B, 1, T, C]
        x = self.ld_conv(x) # [B, D, T, C]
        return x


class MSConv2dDecomp(nn.Module):
    """
    Learnable Multi-Scale Decomposition Block
    """
    def __init__(self, in_channels, kernels: List[int], max_ws=48, stride:int=1):
        super(MSConv2dDecomp, self).__init__()
        self.c_in = in_channels  # in_factors
        assert max(kernels) <= max_ws, "The maximum kernel size should be less than the maximum window size."
        self.stack_dim = len(kernels)# if not bidirect else 2*len(kernels)
        self.kernels = kernels
        self.convs=nn.ModuleList()
        for ks in kernels:
            self.convs.append(
                nn.Conv1d(
                    in_channels, 
                    in_channels, 
                    kernel_size=ks, 
                    stride=stride, 
                    bias=False)
            )

    def forward(self, x, trend_stack=False):
        # x: [B, T, C]
        x = x.transpose(1, 2)  # [B, C, T]
        feats = []
        for (ks, conv) in zip(self.kernels, self.convs):
            feats.append(conv(F.pad(x, (ks-1, 0), value=0)))
        feats = torch.stack(feats, dim=1).transpose(-1, -2)  # [B, D, T, C]
        return feats


# >> Set Embedder <<
class SetEmbedder(nn.Module):
    def __init__(self, configs):
        super(SetEmbedder, self).__init__()
        assert configs.decomp_method in ["dft", "ms_conv2d", "conv2d"], "Invalid decoder method."
        if configs.decomp_method == "dft":
            raise NotImplementedError
        elif configs.decomp_method == "conv2d":
            self.decomposer = Conv2dDecomp(
                1, configs.slot_size  # // 2
            )
        else:
            self.decomposer = MSConv2dDecomp(
                in_channels=configs.enc_in,
                kernels=[2, 4, 8, 16, 32],
                max_ws=64
            )
        decomp_dim = self.decomposer.stack_dim
        self.feat_proj = nn.Linear(decomp_dim, configs.slot_size, bias=False)  
        self.feat_dropout = nn.Dropout(configs.dropout)
        self.feat_norm = RMSNorm(configs.slot_size)

    def forward(self, x):
        # x: [B, T, M] -> [B, D, T, M]
        # Summarize spatial-temporal features as ST graphs
        x_emb = self.decomposer(x, trend_stack=True)  # ([B, 1, T, M], [B, D-1, T, M])
        x_emb = torch.cat(x_emb, dim=1) if isinstance(x_emb, tuple) else x_emb # [B, D, T, M]
        x_emb = rearrange(x_emb, "b d t m -> b t m d")  # [B, T, M, D]
        x_emb = self.feat_dropout(self.feat_norm(self.feat_proj(x_emb)))  # [B, T, M, D]
        return x_emb


class EncoderLayer(nn.Module):
    def __init__(self, 
                 facts_module, 
                 d_model=None,
                 expand: int=2,
                 num_heads: int=1, 
                 dropout: float=0., 
                 activation="relu",
                 norm_first: bool=True,
                 mode: str="normal"):
        super(EncoderLayer, self).__init__()
        self.facts = facts_module
        self.M = facts_module.M
        self.K = facts_module.K
        self.slot_size = facts_module.slot_size
        assert mode in ['factor', 'normal'], f"Invalid mode: {mode}, has to be in ['factor', 'normal']"
        self.mode = mode

        self.norm_in = nn.LayerNorm(self.slot_size) if norm_first else nn.Identity()
        self.norm1 = nn.LayerNorm(self.slot_size)
        self.norm2 = nn.LayerNorm(self.slot_size) if mode=="full" else nn.Identity()
        self.norm_out = nn.Identity() if norm_first else nn.LayerNorm(self.slot_size)

        if mode == "normal":
            d_model = d_model or self.slot_size * expand
            self.mlp = nn.Sequential(
                nn.Linear(self.slot_size, d_model),
                nn.GELU() if activation == "gelu" else nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, self.slot_size),
                nn.Dropout(dropout)
            )

    def forward(self, x, z: torch.Tensor=None, attn_mask=None, tau=None):
        # x: [B, L, M, D], Z: [B, L, K, D]
        if attn_mask is not None:
            assert attn_mask.size(-1)==self.M, f"Expected attn_mask.size(-1)=={self.M}, got {attn_mask.size(-1)}"
            assert attn_mask.size(-2)==self.K, f"Expected attn_mask.size(-2)=={self.K}, got {attn_mask.size(-2)}"

        if z is not None:
            assert z.ndim == 4, f"Expected 4D tensor, got z.shape={z.shape}."
            # z = z[:, -1:]  # takes in only the latest memory [B, 1, K, D]

        B, L, _, _ = x.size()

        # FACTS layer
        x = self.norm_in(x)  # [B, L, M, D]
        z = z if self.facts.init_method is None else None
        new_x, z = self.facts(x, z=z, mask=attn_mask)  # [B, L, K, D], [B, L, K, D]
        
        # factor mode - output directly
        if self.mode == "factor":
            return new_x, z
        
        # normal (standard) mode - no mixer
        new_x = x + new_x if self.M == self.K else new_x  # [B, L, K, D]
        new_x = self.norm1(new_x)  # [B, L, K, D]
        new_x = self.norm2(new_x + self.mlp(new_x))
        
        return self.norm_out(new_x), z  # [B, L, K, D], [B, L, K, D]


class Encoder(nn.Module):
    def __init__(self, facts_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.facts_layers = nn.ModuleList(facts_layers)
        self.norm_out = norm_layer

    def forward(self, x, z_mem:torch.Tensor=None, attn_mask:torch.Tensor=None, tau=None):
        """Forward function for the encoder.
        
        Args:
            x (torch.Tensor): [B, L, M, D], input tensor.
            z_mem (torch.Tensor): [B, L, K, D], memory tensor.
            attn_mask (torch.Tensor): [B, M, M] or [B, L, M, M], attention mask.
            tau (float): temperature for the routing.
        """
        # x: [B, L, K, D], 
        # # z_stacks: ([B, L, K, D]) or None
        for fi, f_layer in enumerate(self.facts_layers):
            x, z_mem = f_layer(x, z=z_mem, attn_mask=attn_mask, tau=tau) # [B, L, K, D], [B, L, K, D]

        if self.norm_out is not None:
            x = self.norm_out(x)
        return x, z_mem


# ============================
# FACTS Decoding Modules
# ============================
# >> Factor Graph Decoder <<
class FactorGraphDecoder(nn.Module):
    """Permutation-invariant Predictor: 
    takes in: [B, T, K, D] -> map to: [B, M+1, T, K] -> perm-inv op: [B, T, M]
    """

    def __init__(self, 
                 slot_size: int,
                 out_factors: int,
                 out_dim: int=1, 
                 d_model: int=None,
                 elem_wise_alpha: bool=False):
        super().__init__()
        self.module_id='fgd'
        self.D = slot_size  # input slot size
        self.M = out_factors  # output factors
        self.alpha_dim = out_factors if elem_wise_alpha else 1

        d_model = d_model or (out_factors+self.alpha_dim)*2
        self.dec_layer = nn.Sequential(
            nn.Linear(slot_size, d_model, bias=False),
            nn.ReLU(),
            nn.Linear(d_model, out_factors+self.alpha_dim)
        )

    def forward(self, slots):
        # slots: [B x t x K x D]
        bs,t,k,d = slots.shape
        N = bs * t
        y = torch.empty((N, self.M), device=slots.device)  # [N, M]

        slots = rearrange(slots, 'b t k d -> (b t) k d')  # [N, K, D]

        # decoding operations (automatically batched to avoid OOM)
        chunk = slots.shape[0]
        while chunk > 0:
            try:
                for i in range(0, N, chunk):
                    y[i:i+chunk] = self._decode(slots[i:i+chunk])
                return rearrange(y, '(b t) m -> b t m', b=bs, t=t)
            except RuntimeError as e:
                chunk = chunk // 2
                continue
        raise RuntimeError("Decoding failed")
    
    def _decode(self, z):
        # z: [N, K, D]
        y, a_k = self.dec_layer(z).split([self.M, self.alpha_dim], dim=-1)  # [N, K, M], [N, K, alpha_dim]
        return (torch.softmax(a_k, dim=1) * y).sum(dim=1)  # [N, M]