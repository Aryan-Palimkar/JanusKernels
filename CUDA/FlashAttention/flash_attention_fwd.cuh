#pragma once

template <const int BLOCK_SIZE_Q, const int BLOCK_SIZE_KV, const int EMB_DIM>
__global__ void flash_attn_fwd(
    const float* Q_ptr, 
    const float* K_ptr, 
    const float* V_ptr, 
    float* M_ptr, 
    float* O_ptr,
    const int stride_batch, const int stride_head, const int stride_seq,
    const int batch_size, const int seq_len, const int num_heads,
    const float scale  
){
    const int block_idx_unroll = blockIdx.x;
    const int num_q_blocks = (seq_len + BLOCK_SIZE_Q - 1) / BLOCK_SIZE_Q;

    const int head_idx = block_idx_unroll / (num_q_blocks * batch_size);
    const int batch_idx = (block_idx_unroll / num_q_blocks) % batch_size;
    const int q_block_idx = block_idx_unroll % num_q_blocks;
    const int tid = threadIdx.x;
    
    const int base_offset = batch_idx * stride_batch + head_idx * stride_head;
    const float* Q_base = Q_ptr + base_offset;
    const float* K_base = K_ptr + base_offset;
    const float* V_base = V_ptr + base_offset;

    extern __shared__ float sram[];
    float* shared_Q = sram;
    float* shared_K = shared_Q + BLOCK_SIZE_Q * EMB_DIM;
    float* shared_V = shared_K + BLOCK_SIZE_KV * EMB_DIM;

    const int q_start = q_block_idx * BLOCK_SIZE_Q;
    const int valid_q = min(BLOCK_SIZE_Q, seq_len - q_start);
    
    for(int i = tid; i < valid_q * EMB_DIM; i += blockDim.x){
        int row = i / EMB_DIM;
        int col = i % EMB_DIM;
        shared_Q[row * EMB_DIM + col] = Q_base[(q_start + row) * stride_seq + col];
    }
    __syncthreads();

    float acc[EMB_DIM];
    float m_i = -INFINITY;
    float l_i = 0.0f;
    
    #pragma unroll
    for(int i = 0; i < EMB_DIM; i++) acc[i] = 0.0f;

    const int num_kv_blocks = (seq_len + BLOCK_SIZE_KV - 1) / BLOCK_SIZE_KV;
    
    for(int kv_block = 0; kv_block < num_kv_blocks; kv_block++){
        const int k_start = kv_block * BLOCK_SIZE_KV;
        const int valid_k = min(BLOCK_SIZE_KV, seq_len - k_start);
        __syncthreads(); 

        for(int i = tid; i < valid_k * EMB_DIM; i += blockDim.x){
            int row = i / EMB_DIM;
            int col = i % EMB_DIM;
            shared_K[row * EMB_DIM + col] = K_base[(k_start + row) * stride_seq + col];
            shared_V[row * EMB_DIM + col] = V_base[(k_start + row) * stride_seq + col];
        }
        __syncthreads();

        if(tid < valid_q){
            float* q_row = shared_Q + tid * EMB_DIM;
            float scores[BLOCK_SIZE_KV];
            float block_max = -INFINITY;

            // Q @ K^T
            for(int k = 0; k < valid_k; k++){
                float* k_row = shared_K + k * EMB_DIM;
                float dot = 0.0f;
                
                #pragma unroll
                for(int d = 0; d < EMB_DIM; d++){
                    dot += q_row[d] * k_row[d];
                }
                
                scores[k] = dot * scale;
                block_max = fmaxf(block_max, scores[k]);
            }

            float m_prev = m_i;
            m_i = fmaxf(m_i, block_max);
            float correction = expf(m_prev - m_i);
            float sum = 0.0f;

            for(int k = 0; k < valid_k; k++){
                scores[k] = expf(scores[k] - m_i);
                sum += scores[k];
            }

            l_i = l_i * correction + sum;

            #pragma unroll
            for(int d = 0; d < EMB_DIM; d++){
                acc[d] *= correction;
            }

            // output += scores @ V
            for(int k = 0; k < valid_k; k++){
                float* v_row = shared_V + k * EMB_DIM;
                float weight = scores[k];
                
                #pragma unroll
                for(int d = 0; d < EMB_DIM; d++){
                    acc[d] += weight * v_row[d];
                }
            }
        }
    } 

    if(tid < valid_q){
        const int global_q_idx = q_start + tid;
        float* out = O_ptr + base_offset + global_q_idx * stride_seq;
        float inv_l = 1.0f / l_i;
        
        #pragma unroll
        for(int d = 0; d < EMB_DIM; d++){
            out[d] = acc[d] * inv_l;
        }
    } 
}