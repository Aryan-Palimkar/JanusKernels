import triton
import triton.language as tl

@triton.jit
def rms_norm_bwd_s(
    DY,
    Y,
    S,
    stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    Y += row * stride
    DY += row * stride

    s = 0.0
    for off in range(0, n_cols, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        y = tl.load(Y + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
        s += tl.sum(y * dy)
    
    tl.store(S + row, s)


@triton.jit
def rms_norm_bdw_dx(
    X,
    W,
    DY,
    DX,
    DW,
    S,
    RSTD,
    stride,
    n_cols: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    DY += row * stride
    DX += row * stride
    S += row
    RSTD += row

    group_id = row % GROUP_SIZE_M
    DW += group_id * n_cols
    
    rstd = tl.load(RSTD)
    s = tl.load(S)

    for off in range(0, n_cols, BLOCK_SIZE_N):
        cols = tl.arange(0, BLOCK_SIZE_N) + off
        mask = cols < n_cols

        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask)
        dy = tl.load(DY + cols, mask=mask)

        x_hat = x * rstd
        dx = rstd * (w * dy - x_hat * (s / n_cols))
        dw_partial = dy * x * rstd

        tl.store(DX + cols, dx, mask=mask)
        tl.atomic_add(DW + cols, dw_partial, mask=mask)



@triton.jit
def rms_norm_bwd_dw(
    DW,
    DW_FINAL,
    GROUP_SIZE_M,
    n_cols,
    stride,
    BLOCK_SIZE: tl.constexpr
):
    col = tl.program_id(0)
    DW += col

    dw_accum = 0.0
    for off in range(0, GROUP_SIZE_M, BLOCK_SIZE):
        offs = tl.arange(0, BLOCK_SIZE) + off
        mask = offs < GROUP_SIZE_M

        dw_partial = tl.load(DW + offs * stride, mask=mask, other=0.0)
        dw_accum += tl.sum(dw_partial)
    
    tl.store(DW_FINAL + col, dw_accum)