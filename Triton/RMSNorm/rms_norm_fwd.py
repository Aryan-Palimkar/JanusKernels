import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_fwd_fused(
    X, 
    Y, 
    W,
    RSTD,
    stride,
    eps,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride

    var = 0.0
    for offs in tl.static_range(0, n_cols, BLOCK_SIZE):
        cols = offs + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        var += tl.sum(x * x, axis=0)

    var /= n_cols
    rstd = tl.math.rsqrt(var + eps)
    tl.store(RSTD + row, rstd)

    for offs in tl.static_range(0, n_cols, BLOCK_SIZE):
        cols = offs + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        y = x * rstd * w
        tl.store(Y + cols, y, mask=mask)

