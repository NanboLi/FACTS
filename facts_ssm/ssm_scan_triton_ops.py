"""
Triton kernels for SSM parallel scans: 
    We are still working on the optimization of the selective scan implementation.
"""
import torch
import triton
import triton.language as tl
from einops import rearrange


# --------------- Selective Scan with Backpropagation ---------------
@triton.jit
def unpack64(merged):
    """
    Adapted from: https://github.com/sustcsonglin/mamba-triton/blob/master/triton_parallel_scan.py
    """
    tl.static_assert(merged.dtype == tl.uint64)
    b = (merged & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    a = (merged >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    return a, b


@triton.jit
def pack64(a, b):
    """
    Adapted from: https://github.com/sustcsonglin/mamba-triton/blob/master/triton_parallel_scan.py
    """
    tl.static_assert(a.dtype == tl.float32)
    tl.static_assert(b.dtype == tl.float32)
    a = a.to(dtype=tl.uint32, bitcast=True).to(tl.uint64)
    a = a << 32
    b = b.to(dtype=tl.uint32, bitcast=True).to(tl.uint64)
    return a | b


@triton.jit()
def first_order_op(l, r):
    """
    See https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf Section 1.4.1

    Adapted from: https://github.com/sustcsonglin/mamba-triton/blob/master/triton_parallel_scan.py
    """
    Al, Xl = unpack64(l)
    Ar, Xr = unpack64(r)
    return pack64(Ar*Al, Ar*Xl + Xr)


@triton.jit
def call_scanner(As, Xs, axis:tl.constexpr=0, rev:tl.constexpr=0):
    # Scan
    tuples = pack64(As, Xs)
    output_tuples_ = tl.associative_scan(tuples, axis=axis, combine_fn=first_order_op)
    _, H = unpack64(output_tuples_)
    return H


@triton.jit
def ssm_scan_kernel(
        A, X, Z, bs, dims, seq_len,
        rev:tl.constexpr=0,
        axis:tl.constexpr=0, 
        BLOCK_SIZE:tl.constexpr=256
    ):
    batch_id = tl.program_id(axis=0)
    dim_id = tl.program_id(axis=1)
    block_id = tl.program_id(axis=2)
    
    seq_ids = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask=(seq_ids < seq_len)

    x_ptrs = X + batch_id * dims * seq_len + dim_id * seq_len + seq_ids
    out_ptrs= Z + batch_id * dims * seq_len + dim_id * seq_len + seq_ids

    if not rev:
        # forward pass
        a_ptrs = A + batch_id * dims * seq_len + dim_id * seq_len + seq_ids
        alphas = tl.load(a_ptrs, mask=mask, other=float(1.))
    else:
        # back pass: shift the input [A_{t}, A_{t-1}, ..., A_{1}] to the right by one: [A_{t+1}, A_{t}, ..., A_{2}]
        a_ptrs = A + batch_id * dims * seq_len + dim_id * seq_len + seq_ids - 1
        alphas = tl.load(a_ptrs, mask=(seq_ids > 0) & mask, other=float(1.))
    xs = tl.load(x_ptrs, mask=mask, other=float(0.))

    # compute scan for each blocks:
    z_scan = call_scanner(alphas, xs, axis=axis)

    # store the result
    tl.store(out_ptrs, z_scan, mask=mask)


def ssm_scan_triton(Aa, Bu, axis:int=0, backward:bool=False, max_block_size:int=256):
    """Parallel scan kernel for computing the prefix sum of a sequence of vectors.
    
    Args:
    - Abar (torch.Tensor): [N, L, D].
    - Bu (torch.Tensor): [N, L, D].
    
    (TODO: auto-tune block_size)
    """
    N, seq_len, D = Aa.shape
    Aa = Aa.transpose(-1, -2).contiguous()
    Bu = Bu.transpose(-1, -2).contiguous()

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = min(triton.next_power_of_2(seq_len), max_block_size)
    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    # simple grid (TODO: optimise the resource allocation)
    grid = (N, D, num_blocks)

    # Allocate output
    z = torch.empty_like(Bu)  # (N, D, seq_len)

    ssm_scan_kernel[grid](
        Aa, Bu, z,
        N, D, seq_len,
        rev=int(backward),
        axis=axis, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return z.transpose(-1, -2).contiguous()


class SSMScanTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A_t, B_u):
        """Forward pass of the selective scan operation.

        Args:
        - A_t (torch.Tensor): [B, T, K, D].
        - B_u (torch.Tensor): [B, T, K, D].

        """
        bs, _, k, d = B_u.size()

        ctx.bs=bs
        ctx.k = k
        ctx.d = d

        A_t = rearrange(A_t, 'b t k d -> b t (k d)').contiguous()
        B_u = rearrange(B_u, 'b t k d -> b t (k d)').contiguous()

        Z_t = ssm_scan_triton(A_t, B_u, backward=False)  
        ctx.save_for_backward(A_t, Z_t)   # save: [B T KD], [B T KD]

        return rearrange(Z_t, 'b t (k d) -> b t k d', k=k, d=d).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of the selective scan operation."""
        dy = rearrange(grad_output, 'b t k d -> b t (k d)').contiguous()
        A_t, Z_t = ctx.saved_tensors  # (B, T, KD), (B, T, KD)
        
        dZ = ssm_scan_triton(A_t.flip(1), dy.flip(1), backward=True).flip(1)  # [B, T, KD]

        # dA_t = dZ * Z_t
        dA_t = torch.zeros_like(dZ)
        dA_t[:, 1:] = dZ[:, 1:] * Z_t[:, :-1]
        dA_t = rearrange(dA_t, 'b t (k d) -> b t k d', k=ctx.k, d=ctx.d).contiguous()
        # dB_u=dZ
        dB_u = rearrange(dZ, 'b t (k d) -> b t k d', k=ctx.k, d=ctx.d).contiguous()
        return dA_t, dB_u
    

SSM_Scan = SSMScanTriton.apply