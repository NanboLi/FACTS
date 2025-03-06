"""
Attention-based (LPI &. RPE) routers for the FACTS model.

Included routers:
    - SoftmaxAttentionRouter: The attention-based router with softmax attention.
    - # LinearAttentionRouter: The attention-based router with linear attention.
    - SlotAttentionRouter: The attention-based router with slot attention.

Any use of this code should cite the following paper:
    FACTS: A Factored State-Space Framework For World Modelling

Author: Nanbo Li (linanbo2008@gmail.com)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from facts_ssm.utils import _getchk
# import pdb


class SoftmaxAttentionRouter(nn.Module):
    """Softmax Attention Router for FACTS. (LPI & RPE)"""

    def __init__(
            self,
            num_slots: int, 
            dim: int,
            aug_dim: int=0,
            to_q: nn.Module=None,
            to_k: nn.Module=None,
            to_v: nn.Module=None,
            to_v_aug: nn.Module=None,
            n_heads:int=1,
            head_dim: int=None,
            norm_inputs:bool=True,
            proj_bias:bool=False,
            dropout:float=0., 
            eps:float=1e-8):
        super().__init__()
        self.K = num_slots
        self.D = dim
        self.n_heads = n_heads
        self.head_dim = head_dim or dim
        self.aug_dim = aug_dim
        assert dim % self.head_dim == 0, f"dim={dim} must be multiples of head_dim, but head_dim={head_dim} now"
        assert aug_dim % dim == 0, f"aug_dim={aug_dim} must be multiples of dim, but dim={dim} now"
        self.attn_scale = self.head_dim ** -0.5
        self.dropout_p = dropout
        self.eps = eps

        # norm inputs and queries
        self.norm_inputs = nn.LayerNorm(dim) if norm_inputs else nn.Identity()
        self.norm_slots = nn.LayerNorm(dim)

        # Q,K,V (optional: V_aug) projections (aug contains auxiliary vars, e.g. deltas)
        self.to_q = to_q or nn.Linear(dim, self.head_dim*n_heads, bias=proj_bias)
        self.to_k = to_k or nn.Linear(dim, self.head_dim*n_heads, bias=proj_bias)
        self.to_v = to_v or nn.Linear(dim, self.head_dim*n_heads, bias=proj_bias)

        n_groups = aug_dim // dim  # assuming dt_rank=B_rank=C_rank=slot_size
        if aug_dim > 0:
            # n_groups = aug_dim // dim  # assuming dt_rank=B_rank=C_rank=slot_size
            self.to_v_aug = to_v_aug or nn.Sequential(
                nn.GroupNorm(num_channels=aug_dim, num_groups=n_groups),
                nn.Conv1d(
                    aug_dim, 
                    (n_groups*self.head_dim)*n_heads, 
                    kernel_size=1,
                    groups=n_groups, 
                    bias=proj_bias
                )
            )

        # output projection: [B, hD+h_aug, H, K] -> [B, hD+h_aug, 1, K]
        self.out_proj = nn.Conv2d(
            (n_groups+1)*self.head_dim, 
            (n_groups+1)*dim,
            kernel_size=(n_heads, 1),
            groups=(n_groups+1),  
            bias=False
        )

    def forward(self, inputs, slots, aug_inputs: torch.Tensor=None, mask:torch.Tensor=None):
        """Forward function for the attention-based router.
        
        Args:
            inputs (torch.Tensor): [B, M, D] The input tensor.
            slots (torch.Tensor): [B, K, D] The latent slots.
            aug_inputs (torch.Tensor, optional): [B, M, aug_dim] The time deltas. Defaults to None.
            mask (torch.Tensor, optional): [B, K, M] The mask tensor. Defaults to None.
        """
        assert slots.ndim == 3, "Slot tensor must be 4D tensor [B K D]"
        assert inputs.ndim == 3, "Input tensor must be 3D tensor (batched matrices / spatial graphs)"
        b, m, _, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        k = slots.size(1)
        h = self.n_heads

        # norm inputs and slots:
        inputs = self.norm_inputs(inputs)  # [B, M, D]
        slots = self.norm_slots(slots)  # [B, K, D]

        # perform cross-attention routing (quadratic in memory size)
        ukeys = rearrange(self.to_k(inputs), 'b m (h d)  -> b h m d', h=h, m=m)  # [B, H, K, hD]
        uvals = rearrange(self.to_v(inputs), 'b m (h d)  -> b h m d', h=h, m=m) # [B, H, K, hD]
        slots = rearrange(self.to_q(slots), 'b k (h d) -> b h k d', h=h, k=k)  # [B, H, K, hD]

        if aug_inputs is not None:
            assert aug_inputs.size(-1) == self.aug_dim, f"Augmented input dim ({aug_inputs.size(-1)}) must match the router's aug_dim ({self.aug_dim})"
            aug_inputs = rearrange(self.to_v_aug(aug_inputs.transpose(-1, -2)), 'b (h a) m -> b h m a', h=h)  # [B, H, M, aug_h_dim]
            uvals = torch.cat([uvals, aug_inputs], dim=-1)  # [B, H, M, hD+aug_h_dim]
        if mask is not None:
            mask = repeat(mask, 'b k m -> b h k m', h=h)  # [B, H, K, M]

        # # routing operations (automatically batched to avoid OOM)
        chunk = slots.size(0)
        while chunk > 0:
            try:
                v_out = torch.cat(
                    [self._rout(slots[i:i+chunk], ukeys[i:i+chunk], uvals[i:i+chunk], _getchk(mask,i,i+chunk,b)) for i in range(0,b,chunk)], 
                    dim=0
                )   # [B, H, K, D(+aug_dim)]
                break
            except RuntimeError as e:
                chunk = chunk // 2
                continue

        # output projection
        v_out = rearrange(
            self.out_proj(rearrange(v_out, 'b h k v -> b v h k')), 'b v 1 k -> b k v'
        )  # [B, K, D(+aug_dim)]

        aug_vals = None
        if v_out.size(-1) == self.D+self.aug_dim:
            v_out, aug_vals = torch.split(v_out, [self.D, self.aug_dim], dim=-1)
        return v_out, aug_vals  # [B, K, D], [B, K, aug_dim] or None
    
    def _rout(self, Q, K, V, mask=None):
        """Operation for the attention-based router.
        
        Args:
            Q (torch.Tensor):      [B, H, K, hD]
            K (torch.Tensor):      [B, H, M, hD]
            V (torch.Tensor):      [B, H, M, hD(+aug_h_dim)]
            mask (torch.Tensor):   [B, H, K, M] or None, values in [-inf, 0]

        Output:
            V (torch.Tensor):      [B, H, K, hD(+aug_h_dim)]
        """
        V = F.scaled_dot_product_attention(
            Q, K, V, 
            attn_mask=mask, 
            dropout_p=self.dropout_p, 
            is_causal=False
        )  # [B, H, K, hD(+aug_h_dim)]
        return V


class InvertedAttentionRouter(SoftmaxAttentionRouter):
    """Inverted Softmax Attention Router for FACTS. (LPI & RPE)
    
    Note that, similar attention mechanism is used in the Slot Attention.
    However, original SlotAttention involves iterative routing operations 
    (using RNNs), here we only perform one-shot routing --- moreover, 
    with simplified operations.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0. else nn.Identity()
    
    def _rout(self, Q, K, V, mask=None):
        """Operation for the attention-based router.
        
        Args:
            Q (torch.Tensor):      [B, H, K, hD]
            K (torch.Tensor):      [B, H, M, hD]
            V (torch.Tensor):      [B, H, M, hD]
            mask (torch.Tensor):   [B, H, K, M] or None, values in [-inf, 0]

        Output:
            V (torch.Tensor):      [B, H, K, hD(+aug_h_dim)]
        """
        attn = self.attn_scale * (Q @ K.transpose(-2, -1))  # [B, H, K, M]
        if mask is not None:
            # attn = attn.masked_fill(mask, -math.inf)
            # attn = attn + mask
            attn = attn * mask

        attn = attn.softmax(dim=-2) + self.eps  # [B, H, K, M]
        # normalize over M
        attn = attn / torch.sum(attn, dim=-1, keepdim=True)  # [B, H, K, M]
        attn = self.attn_dropout(attn) # [B, H, K, M]
        V = attn @ V  # [B, H, K, hD(+aug_h_dim)]
        return V


router_zoo = {
    "sfmx_attn": SoftmaxAttentionRouter,
    # "lin_attn": LinearAttentionRouter,
    "inverted": InvertedAttentionRouter,
}

def build_router(router, *args, **kwargs):
    assert router in router_zoo.keys(), \
        f"Router {router} not found in the zoo, please choose from {list(router_zoo.keys())}"
    return router_zoo[router](*args, **kwargs)