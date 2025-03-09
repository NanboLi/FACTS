"""
FACTS core modules.

Any use of this code should cite the following paper:
    FACTS: A Factored State-Space Framework For World Modelling

Author: Nanbo Li (linanbo2008@gmail.com)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
from facts_ssm.routers import build_router


# ============================
# FACTS Core Modules
# ============================
class FactsBuildProjection(nn.Module):
    """ 
    FACTS Building Projection: Row-wise projections for the FACTS input
    tensor. The output will be used for routing. 
    
    Remarks:
      - rank(dt)=1 for slim mode; rank(dt)=D for full (dim-wise) mode
      - rank(A)=1, rank(B)=D, rank(C)=D
    """
    def __init__(
            self,
            in_features: int,
            in_factors: int,
            x_dim: int,
            param_dim: int, 
            expand: int,
            d_conv: int,
            res_dim: int=0):
        super().__init__()
        self.in_factors = in_factors  # M
        self.x_dim = x_dim  # D
        self.param_dim = param_dim  # A
        self.res_dim = res_dim  # res_dim=(D or 0)
        self.expand = expand
        
        # intermediate dimensions
        self.x_param_dim = x_dim + param_dim
        
        # InProj(x) -> x_pack = (x | params | (res))
        self.in_proj = nn.Linear(
            in_features, 
            self.x_param_dim * self.expand + self.res_dim, 
            bias=False
        )

        # Conv(x_pack, conv_along='time_axis') -> x_pack
        self.xParam_conv = nn.Sequential(
            nn.ZeroPad2d((0,0,d_conv-1,0)),
            nn.Conv2d(
                in_channels=self.x_param_dim * self.expand,
                out_channels=self.x_param_dim,
                bias=True,
                kernel_size=(d_conv, 1)
            )
        )  

    def forward(self, x):
        """Forward function for the FACTS building projection.
        
        Args:
            x (torch.Tensor): [B, T, M, D], input tensor.

        Returns:
            x   (torch.Tensor):           [B, T, M, D], input tensor.
            params (torch.Tensor):        [B, T, M, A], parameters for the routing.
            res (torch.Tensor, optional): [B, T, M, D] or None, residual projection.
        """
        # input projection: x -> (dt, xBC, (res))
        x = self.in_proj(x)  # [B, T, M, x_param_dim * expand + res_dim]
        res = None
        if self.res_dim > 0:
            (x, res) = x.split(split_size=[self.x_param_dim*self.expand, self.res_dim], dim=-1)
            res = F.silu(res)  # [B, T, M, D]

        # row-wise Conv(xBCdt) -> split -> (x, B, C, dt)
        x = self.xParam_conv(rearrange(x, 'b t m p -> b p t m'))  # [B, x_param_dim, T, M]
        # split -> [B, T, M, x_dim], [B, T, M, param_dim]
        x, params = rearrange(x, 'b p t m -> b t m p').split([self.x_dim, self.param_dim], dim=-1)  

        return x, params, res
    

class FACTS(nn.Module):
    """FACTS: FACTored State-space model."""

    def __init__(self,
                 in_features: int,
                 in_factors: int,
                 num_factors: int,
                 slot_size: int,
                 expand: int=2,
                 d_conv: int=2,
                 num_heads: int=1,
                 dropout: float=0.,
                 C_rank: int=0,
                 router: str='sfmx_attn',
                 init_method: str='learnable',
                 fast_mode: bool=True,
                 slim_mode: bool=False,
                 residual: bool=True,
                 chunk_size: int=-1,
                 dt_bias: float=1e-4,
                 eps: float=1e-12):
        """
        Initializes a FACTS model.

        Args:
            in_features (int): input feature/emb dimension.
            in_factors (int): num. of input factors.  (M)
            num_factors (int): num. of output factors.  (K)
            slot_size (int): slot size/dim.  (D)
            expand (int): expansion factor for the input projection (for intermediate dims).
            d_conv (int): kernel size for convolving along the input sequential axis.
            num_heads (int): number of heads for attention-based routing.
            dropout (float): dropout rate.
            router (str): router type.
            init_method (str): memory initialisation method.
            fast_mode (bool): whether to use fast/fully-parallel mode.
            slim_mode (bool): whether to use slim mode. head_dim=slot_size//num_heads if slim_mode else head_dim=slot_size.
            C_rank (int): rank of the C projection (options: [0, 1, D]). 0 for no C-selective projection.
            residual (bool): whether to use residual connection, not supported for M != K     
            chunk_size (int): chunk size for the chunked RNN/partial parallelisation: chunk_{t} = RNN(chunk_{t-1})
            dt_bias (float): bias for the dt term, expected to be positive.
            eps (float): small value for numerical stability/precision control.    
        """
        super().__init__()
        self.M = in_factors  # M
        self.K = num_factors  # K
        self.slot_size = slot_size  # D
        self.fast_mode = fast_mode or (chunk_size==-1)  # whether to use fast mode
        self.chunk_size = chunk_size  # chunk size for the chunked RNN: chunk_{t} = RNN(chunk_{t-1})
        if init_method not in ['init_state', 'uniform', 'spherical', 'learnable', 'self']:
            init_method = None
        if init_method == 'init_state':
            assert self.M == self.K, "In 'init_state' mode, M should be equal to K."
        self.init_method = init_method
        self.slim_mode = slim_mode
        self.dt_bias = dt_bias
        self.eps = eps
        
        # the ranks of the params:
        self.dt_rank = slot_size
        self.A_rank = 1
        self.B_rank = slot_size
        self.C_rank = C_rank if C_rank==slot_size else 0
        # if use residual connection for layer stacking
        res_dim = slot_size if (residual and self.M==self.K) else 0

        # input projection for constructing FACTS
        param_dim = self.dt_rank + self.B_rank + self.C_rank
        self.facts_build_proj = FactsBuildProjection(
            in_features=in_features,
            in_factors=in_factors,
            x_dim=slot_size,
            param_dim=param_dim, 
            expand=expand,
            d_conv=d_conv,
            res_dim=res_dim
        )
        # dynamic routing for selective state-space model
        if slim_mode:
            assert slot_size % num_heads == 0, \
                f"In slim mode, slot_size (now {slot_size}) must be divisible by num_heads (now {num_heads})."
            head_dim = slot_size // num_heads
        else:
            head_dim = slot_size

        self.router = build_router(
            router,
            num_slots=num_factors, 
            dim=slot_size,
            head_dim=head_dim,
            aug_dim=param_dim,
            n_heads=num_heads,
            norm_inputs=True,
            dropout=dropout,
            eps=self.eps
        )

        if init_method == 'learnable':
            self.slots_mu = nn.Parameter(
                nn.init.normal_(torch.empty((1, self.K, self.slot_size))), requires_grad=True
            )
        elif init_method == 'uniform':
            # Factor initialisation parameters
            self.register_buffer("slots_mu", torch.zeros((1, self.K, self.slot_size)))
        elif init_method == 'spherical':
            self.register_buffer("slots_mu", torch.zeros((1, self.K,  self.slot_size)))
            self.register_buffer("slots_log_sigma", torch.zeros((1, self.K, self.slot_size)))
        else:
            pass

        self.A_log = nn.Parameter(
            torch.log(torch.ones(self.A_rank))
        )
        # self.A_log._no_weight_decay = True

    def init_memory(self, x: torch.Tensor, z: torch.Tensor=None):
        """Initialise the memory tensor.

        Args:
            x (torch.Tensor): [B, T, M, D], the input/observation tensor.
            z (optional, torch.Tensor): [B, 1 or T, M, D], the input memory.

        Returns:
            z (torch.Tensor): [B, 1, K, D], memory tensor.
        """
        bs = x.size(0)

        # if no init_method specified, we return z as it is
        if self.init_method is None:
            assert z is not None, "If init_method is not provided, memory tensor should be provided."
            return z
        
        if self.init_method == 'self':   # self-attention
            z = x.clone()  # [B, T, M, D], assume input z already normalised
        elif self.init_method == 'init_state':
            z = x[:, :1].clone().detach()  # [B, 1, K=M, D]
        elif self.init_method == 'learnable':
            z = self.slots_mu.expand(bs, -1, -1).unsqueeze(1)  # [B, 1, K, D]
        elif self.init_method == 'uniform':
            z = torch.rand_like(self.slots_mu.expand(bs, -1, -1)).unsqueeze(1)   # [B, 1, K, D]
            nn.init.xavier_uniform_(z, gain=nn.init.calculate_gain("linear")) # [B, 1, K, D]
        elif self.init_method == 'spherical':
            mu = self.slots_mu.expand(bs, -1, -1)
            sigma = self.slots_log_sigma.exp().expand(bs, -1, -1)
            z = mu + sigma * torch.randn_like(mu, device=x.device, dtype=x.dtype)  # [B, K, D]
            z = z.unsqueeze(1)  # [B, 1, K, D]
        else:
            raise ValueError(f"Invalid init_method: {self.init_method}")
        return z

    def forward(self, x, z: torch.Tensor=None, mask: torch.Tensor=None):
        """ FACTS forward function.

        Args:
            x (torch.Tensor): [B, T, M, D], input tensor.
            z (torch.Tensor, optional): [B, 1, K, D], memory tensor.
            mask (torch.Tensor, optional): [B, M, M], [B, T, M, M] or None , attention mask.

        Returns:
            z_hat (torch.Tensor): [B, T, K, D], selective t-step output.
            z (torch.Tensor): [B, T, K, D], updated t-step memory tensor.
        """
        bs = x.size(0)

        # Initialisation of the memory
        z = self.init_memory(x, z)  # [B, 1, K, D]

        # Row-wise (along the M dimension) projections of the input 
        x, pe_params, res = self.facts_build_proj(x)
        
        if self.fast_mode or self.chunk_size >= x.size(1):
            z_hat, z = self.ssm(x, z, pe_params, mask)
        else:
            assert self.chunk_size > 0, "Chunk size should be greater than 0."
            assert self.chunk_size < x.size(1), "Chunk size should be less than the sequence length."
            z_hat, z = self.ssm_chunked_rnn(x, z, pe_params, mask)

        if res is not None:
            assert res.ndim==4, f"Residual should be 4D tensor, but got {res.size()}."
            z_hat = z_hat * res

        return z_hat, z
    
    def ssm(self, x, z, pe_params, mask):
        """Parallel SSM that models the dynamic graphs with temporal selectivity.
        
        Args:
            x (torch.Tensor): [B, T, M, D], input tensor.
            z (torch.Tensor): [B, 1, K, D], memory tensor.
            pe_params (torch.Tensor): [B, T, M, dt+B+C], perm-inv params projection.
            mask (torch.Tensor, optional): [B, M, M], [B, T, M, M] or None, attention mask.
        """
        bs, t, _, _ = x.size()
        assert z.size(1)==1 or z.size(1)==t, f"Expect z.size(1)=(1 or t={t}) but got z.size={z.shape}"        
        A = -torch.exp(self.A_log.float())

        # The FACTS routing (LPI, RPE)
        u, pe_params = self.router(
            rearrange(x, 'b t m d -> (b t) m d'), 
            rearrange(z, 'b t k d -> (b t) k d') if z.size(1)==t else repeat(z[:, 0], 'b k d -> (b t) k d', t=t),
            rearrange(pe_params, 'b t m p -> (b t) m p'),
            mask=mask
        )  # [B, K, D], None
        u = rearrange(u, '(b t) k d -> b t k d', b=bs, t=t) # [B, T, K, D]
        pe_params = rearrange(pe_params, '(b t) k r -> b t k r', b=bs, t=t)

        if self.C_rank:
            dt, B, C = pe_params.split([self.dt_rank, self.B_rank, self.C_rank], dim=-1)
        else:
            dt, B = pe_params.split([self.dt_rank, self.B_rank], dim=-1)
            C = 1
        dt = F.softplus(dt).expand(-1, -1, -1, self.slot_size) + self.dt_bias  #+1e-4  # [B, T, K, D]

        # Cumsum implementation for the Fast SSM
        dA_prod = F.pad(dt * A, (0, 0, 0, 0, 0, 1)).flip(1).cumsum(1).exp().flip(1)  # [B, T+1, K, D]
        if z.size(1) == 1:
            dB_u = torch.cat((z, (dt*B)*u), dim=1)  # [B, T+1, K, D]
        elif z.size(1) == t:
            dB_u = F.pad((dt*B)*u, (0,0,0,0,1,0))  # [B, T+1, K, D]
        else:
            raise ValueError(f"Invalid z.size(1)={z.size(1)} != (1 or {t})")
        z = dB_u * dA_prod
        z = z.cumsum(1) / (dA_prod + self.eps) #1e-12)
        z = z[:, 1:]  # [B, T, K, D]
        return z * C, z

    def ssm_chunked_rnn(self, x, z, pe_params, mask):
        """Selective SSM in its RNN mode (recursively on chunks).
        
        Args:
            x (torch.Tensor): [B, T, M, D], input tensor.
            z (torch.Tensor): [B, 1, K, D], memory tensor.
            pe_params (torch.Tensor): [B, T, M, dt+B+C], perm-inv params projection.
            mask (torch.Tensor, optional): [B, M, M], [B, T, M, M] or None, attention mask.
        """
        assert z.size(1) == 1, "Input memory should of the last time step only, i.e. expect z.size(1)=1,"

        T = x.size(1)
        chunk = self.chunk_size
        z_hat = []
        zs = []
        for i in range(0, T, self.chunk_size):
            z_hat_chunk, z_chunk = self.ssm(
                x=x[:,i:i+chunk], 
                z=z, 
                pe_params=pe_params[:,i:i+chunk],
                mask=(mask[:,i:i+chunk] if mask.ndim==4 else mask) if mask is not None else None
            )
            z=z_chunk[:, -1:]  # [B, 1, K, D]
            z_hat.append(z_hat_chunk)
            zs.append(z_chunk)

        z_hat = torch.cat(z_hat, dim=1)  # [B, T, K, D]
        zs = torch.cat(zs, dim=1)  # [B, T, K, D]
        return z_hat, zs
