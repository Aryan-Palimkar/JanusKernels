import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    O_block, l_i, m_i, Q_block, K_block_ptr, V_block_ptr, block_idx_q, scale, 
    BLOCK_SIZE_Q: tl.constexpr, 
    BLOCK_SIZE_KV: tl.constexpr, 
    STAGE: tl.constexpr, 
    offsets_q: tl.constexpr, 
    offsets_kv: tl.constexpr, 
    SEQ_LEN: tl.constexpr,
):
    if STAGE == 1: # causal attn
        low, high = 0, block_idx_q * BLOCK_SIZE_Q
    elif STAGE == 2: # causal attn diagnol rows
        low, high = block_idx_q * BLOCK_SIZE_Q, (block_idx_q + 1) * BLOCK_SIZE_Q

    else: # non-causal attn
        low, high = 0, SEQ_LEN
    
    K_block_ptr = tl.advance(K_block_ptr, (0, low))
    V_block_ptr = tl.advance(V_block_ptr, (low, 0))

    for start_kv in range(low, high, BLOCK_SIZE_KV):
  

        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)
        QK_block *= scale

        if STAGE == 2:
            mask = offsets_q[:, None] >= (start_kv + offsets_kv[None, :])
            QK_block += tl.where(mask, 0, -1.0e6)
        
        m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
        QK_block -= m_ij[:, None]
        
        P_block = tl.math.exp(QK_block)

        l_ij = tl.sum(P_block, 1)

        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        V_block = tl.load(V_block_ptr)
        
        P_block = P_block.to(tl.float16)

        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)

        m_i = m_ij

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV)) # K[HEAD_DIM, SEQ_LEN]
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0)) # V[SEQ_LEN, HEAD_DIM]

    return O_block, l_i, m_i

@triton.jit
def _attn_fwd(
    Q, 
    K, 
    V, 
    scale,
    M,
    O, 
    stride_Q_batch, 
    stride_Q_head,
    stride_Q_seq, 
    stride_Q_dim, 
    stride_K_batch, 
    stride_K_head, 
    stride_K_seq, 
    stride_K_dim, 
    stride_V_batch, 
    stride_V_head, 
    stride_V_seq, 
    stride_V_dim, 
    stride_O_batch, 
    stride_O_head, 
    stride_O_seq, 
    stride_O_dim,
    stride_M_batch,
    stride_M_head,
    stride_M_seq,
    BATCH_SIZE: tl.constexpr,
    NUMS_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
            ):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    block_idx_q = tl.program_id(0)
    batch_head_idx = tl.program_id(1)

    idx_batch = batch_head_idx // NUMS_HEADS
    idx_head = batch_head_idx % NUMS_HEADS

    q_offset = idx_batch.to(tl.int64) * stride_Q_batch + idx_head.to(tl.int64) * stride_Q_head
    kv_offset = idx_batch.to(tl.int64) * stride_K_batch + idx_head.to(tl.int64) * stride_K_head
    
    Q_block_ptr = tl.make_block_ptr(
        base = Q + q_offset,
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_Q_seq, stride_Q_dim),
        offsets = (block_idx_q * BLOCK_SIZE_Q, 0),
        block_shape = (BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base = V + kv_offset,
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_V_seq, stride_V_dim),
        offsets = (0, 0),
        block_shape = (BLOCK_SIZE_KV, HEAD_DIM),
        order=(1,0),
    )

    K_block_ptr = tl.make_block_ptr(
        base = K + kv_offset,
        shape = (HEAD_DIM, SEQ_LEN),
        strides = (stride_K_dim, stride_K_seq),
        offsets = (0, 0),
        block_shape = (HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )

    O_block_ptr = tl.make_block_ptr(
        base = O + q_offset,
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_O_seq, stride_O_dim),
        offsets = (block_idx_q * BLOCK_SIZE_Q, 0),
        block_shape = (BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    m_offset = idx_batch.to(tl.int64) * stride_M_batch + idx_head.to(tl.int64) * stride_M_head
    M_ptr = tl.make_block_ptr(
        base = M + m_offset,
        shape = (SEQ_LEN,),
        strides = (stride_M_seq,),
        offsets = (block_idx_q * BLOCK_SIZE_Q,),
        block_shape = (BLOCK_SIZE_Q,),
        order = (0,)
    )

    
    offsets_q = block_idx_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offsets_kv = tl.arange(0, BLOCK_SIZE_KV)

    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    o_i = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    Q_block = tl.load(Q_block_ptr)

    if STAGE == 1 or STAGE == 3:
        o_i, l_i, m_i = _attn_fwd_inner(
            o_i, l_i, m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_idx_q,
            scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offsets_q,
            offsets_kv,
            SEQ_LEN,
        )
    
    if STAGE == 3:
        o_i, l_i, m_i = _attn_fwd_inner(
            o_i, l_i, m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_idx_q,
            scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2, 
            offsets_q,
            offsets_kv,
            SEQ_LEN,
        )
    
    o_i = o_i / l_i[:, None]
    tl.store(O_block_ptr, o_i.to(Q.dtype.element_ty))
    tl.store(M_ptr, m_i + tl.math.log(l_i))
    

@triton.jit
def _attn_bwd_preprocess(
    O,
    dO,
    D,
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    block_index_q = tl.program_id(0)
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    index_batch_head = tl.program_id(1)
    offs_dim = tl.arange(0, HEAD_DIM)

    O_block = tl.load(
        O
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM
        + offs_dim[None, :]
    ) # (BLOCK_SIZE_Q, HEAD_DIM)

    dO_block = tl.load(
        dO
        + index_batch_head * SEQ_LEN * HEAD_DIM
        + offs_q[:, None] * HEAD_DIM
        + offs_dim[None, :]
    ) # (BLOCK_SIZE_Q, HEAD_DIM)

    D_block = tl.sum(O_block * dO_block, axis=1) # (BLOCK_SIZE_Q)

    D_block_ptr = D + index_batch_head * SEQ_LEN + offs_q
    tl.store(D_block_ptr, D_block)

@triton.jit
def _attn_bwd_dk_dv(
    Q,
    K,
    V,
    scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offsets_batch_head = (stride_batch * index_batch + stride_head * index_head).to(tl.int64)

    offsets_batch_head_seq = ((index_batch * NUM_HEADS + index_head) * SEQ_LEN).to(tl.int64)

    Q += offsets_batch_head
    K += offsets_batch_head
    V += offsets_batch_head
    dO += offsets_batch_head
    dQ += offsets_batch_head
    dK += offsets_batch_head
    dV += offsets_batch_head

    M += offsets_batch_head_seq
    D += offsets_batch_head_seq

    offs_dim = tl.arange(0, HEAD_DIM)

    index_block_kv = tl.program_id(0)
    start_kv = index_block_kv * BLOCK_KV

    offs_kv = start_kv + tl.arange(0, BLOCK_KV)

    dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    K_block = tl.load(
        K + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    ) # (BLOCK_KV, HEAD_DIM)

    V_block = tl.load(
        V + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    ) # (BLOCK_KV, HEAD_DIM)

    offs_q = tl.arange(0, BLOCK_Q)

    qT_ptrs = Q + offs_q[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    dO_ptrs = dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim

    curr_q = 0
    num_steps = SEQ_LEN // BLOCK_Q

    for _ in range(num_steps):
        qT_block = tl.load(qT_ptrs)
        offs_q = curr_q + tl.arange(0, BLOCK_Q)
        m = tl.load(M + offs_q)

        QK_T_block = scale * tl.dot(K_block, qT_block) # this is S^T ((QK^T)^T) not S (QK^T)
        P_T_block = tl.math.exp(QK_T_block - m[None, :])

        if STAGE == 3: #causal
            mask_block = (
                offs_q[None, :] >= offs_kv[:, None]
            )

            P_T_block = tl.where(mask_block, P_T_block, 0.0)

        dO_block = tl.load(dO_ptrs)
        dV_block += tl.dot(P_T_block.to(tl.float16), dO_block)

        Di = tl.load(D + offs_q)

        # dP = dO x V^T, so dP^T = V x dO^T
        dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)

        # dS = P * (dP - D), so dS^T = P^T * (dP^T - D^T)
        dS_T_block = P_T_block * (dpT_block - Di[None, :])
        dS_T_block = dS_T_block.to(tl.float16)

        dK_block += scale * tl.dot(dS_T_block, tl.trans(qT_block))

        curr_q += BLOCK_Q
        qT_ptrs += BLOCK_Q * stride_seq
        dO_ptrs += BLOCK_Q * stride_seq

    dV_block_ptrs = dV + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dV_block_ptrs, dV_block)

    dK_block_ptrs = dK + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dK_block_ptrs, dK_block)
    
@triton.jit
def _attn_bwd_dq(
    Q,
    K,
    V,
    scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offsets_batch_head = (stride_batch * index_batch + stride_head * index_head).to(tl.int64)

    offsets_batch_head_seq = ((index_batch * NUM_HEADS + index_head) * SEQ_LEN).to(tl.int64)

    Q += offsets_batch_head
    K += offsets_batch_head
    V += offsets_batch_head
    dO += offsets_batch_head
    dQ += offsets_batch_head
    dK += offsets_batch_head
    dV += offsets_batch_head

    M += offsets_batch_head_seq
    D += offsets_batch_head_seq

    offs_dim = tl.arange(0, HEAD_DIM)
    index_block_q = tl.program_id(0)

    start_q = index_block_q * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)

    Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
    dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    dO_block = tl.load(dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)

    M_block = tl.load(M + offs_q)
    M_block = M_block
    
    offs_kv = tl.arange(0, BLOCK_KV)

    kT_ptrs = K + offs_dim[:, None] * stride_dim + offs_kv[None, :] * stride_seq
    v_ptrs = V + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim

    Di = tl.load(D + offs_q)

    curr_kv = 0
    num_steps = SEQ_LEN // BLOCK_KV

    for _ in range(num_steps):
        K_T_block = tl.load(kT_ptrs)
        V_block = tl.load(v_ptrs)

        QK_block = scale * tl.dot(Q_block, K_T_block)
        P_block = tl.math.exp(QK_block - M_block[:, None])

        if STAGE == 3:
            offs_kv = curr_kv + tl.arange(0, BLOCK_KV)
            mask = offs_q[:, None] >= offs_kv[None, :]
            P_block = tl.where(mask, P_block, 0.0)
        
        dP_block = tl.dot(dO_block, tl.trans(V_block)).to(tl.float32)
        dS_block = P_block * (dP_block - Di[:, None])
        dS_block = dS_block.to(tl.float16)

        dQ_block += scale * tl.dot(dS_block, tl.trans(K_T_block))

        curr_kv += BLOCK_KV
        kT_ptrs += BLOCK_KV * stride_seq
        v_ptrs += BLOCK_KV * stride_seq

    dQ_ptr = dQ + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dQ_ptr, dQ_block)


class FlashAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool, scale):
        HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V = V.shape[-1]

        BATCH_SIZE, NUMS_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_Q == HEAD_DIM_V

        O = torch.empty_like(Q)
        stage = 3 if causal else 1

        BLOCK_SIZE_Q = 16
        BLOCK_SIZE_KV = 16

        grid = lambda meta: (triton.cdiv(SEQ_LEN, meta['BLOCK_SIZE_Q']), BATCH_SIZE * NUMS_HEADS, 1, )

        M = torch.empty((BATCH_SIZE, NUMS_HEADS, SEQ_LEN), device='cuda', dtype=torch.float32)

        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            scale=scale,
            M=M,
            O=O,
            stride_Q_batch = Q.stride(0),
            stride_Q_head = Q.stride(1),
            stride_Q_seq = Q.stride(2),
            stride_Q_dim = Q.stride(3),
            stride_K_batch = K.stride(0),
            stride_K_head = K.stride(1),
            stride_K_seq = K.stride(2),
            stride_K_dim = K.stride(3),
            stride_V_batch = V.stride(0),
            stride_V_head = V.stride(1),
            stride_V_seq = V.stride(2),
            stride_V_dim = V.stride(3),
            stride_O_batch = O.stride(0),
            stride_O_head = O.stride(1),
            stride_O_seq = O.stride(2),
            stride_O_dim = O.stride(3),
            stride_M_batch = M.stride(0),
            stride_M_head = M.stride(1),
            stride_M_seq = M.stride(2),
            BATCH_SIZE=BATCH_SIZE,
            NUMS_HEADS=NUMS_HEADS,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM_K,
            BLOCK_SIZE_Q=BLOCK_SIZE_Q,
            BLOCK_SIZE_KV=BLOCK_SIZE_KV,
            STAGE=stage,
        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.scale = scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal

        return O
    
    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        Q, K, V, O, M = ctx.saved_tensors


        assert dO.is_contiguous()
        assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()

        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3]
        NUM_WARPS, NUM_STAGES = 4, 1
        BLOCK_SIZE_Q_BWD = 32
        BLOCK_SIZE_KV_BWD = 32

        preprocess_grid = (SEQ_LEN // BLOCK_SIZE_Q_BWD, BATCH_SIZE * NUM_HEADS)
        D = torch.empty_like(M) # (BATCH_SIZE, NUM_HEADS, SEQ_LEN)

        _attn_bwd_preprocess[preprocess_grid](
            O=O,
            dO=dO,
            D=D,
            SEQ_LEN=SEQ_LEN,
            BLOCK_SIZE_Q=BLOCK_SIZE_Q_BWD,
            HEAD_DIM=ctx.HEAD_DIM,
        )

        grid_dk_dv = (SEQ_LEN // BLOCK_SIZE_KV_BWD, 1, BATCH_SIZE * NUM_HEADS)
        stage = 3 if ctx.causal else 1

        _attn_bwd_dk_dv[grid_dk_dv](
            Q=Q,
            K=K,
            V=V,
            scale=ctx.scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_Q_BWD,
            BLOCK_KV=BLOCK_SIZE_KV_BWD,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

        grid_dq = (SEQ_LEN // BLOCK_SIZE_Q_BWD, 1, BATCH_SIZE * NUM_HEADS)
        _attn_bwd_dq[grid_dq](
            Q=Q,
            K=K,
            V=V,
            scale=ctx.scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_Q_BWD,
            BLOCK_KV=BLOCK_SIZE_KV_BWD,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

        return dQ, dK, dV, None, None


def test(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device='cuda'
        ).normal_(mean=0.0, std=0.5).requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device='cuda'
        ).normal_(mean=0.0, std=0.5).requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device='cuda'
        ).normal_(mean=0.0, std=0.5).requires_grad_()
    )

    scale = 1 / (HEAD_DIM)**0.5

    d0 = torch.randn_like(Q)

    mask = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device='cuda'))
    P = torch.matmul(Q, K.transpose(2,3)) * scale
    if causal:
        P[:, :, mask == 0] = float('-inf')
    P = torch.softmax(P.float(), dim=-1).half()
    ref_O = torch.matmul(P, V)
    ref_O.backward(d0)
    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None

    tri_out = FlashAttention.apply(Q, K, V, causal, scale).half()
    tri_out.backward(d0)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None

    rtol = 1e-2
    atol = 1e-2
    
    assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)

    print("All checks passed")

if __name__ == "__main__":
    print("\nTesting non-causal attention")
    test(2, 4, 1024, 256, False)
    print("\nTesting causal attention")
    test(2, 4, 1024, 256, True)