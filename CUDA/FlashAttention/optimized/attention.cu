#include "utils.cuh"
#include <cuda_fp16.h>
#include <float.h>
#include <iostream>

template<int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS>
__launch_bounds__(NUM_WARPS * WARP_SIZE)
__global__ void flash_attn_forward(
    const half* Q, // [bs, num_heads, len_q, DIM]
    const half* K, // [bs, num_heads, len_kv, DIM]
    const half* V, // [bs, num_heads, len_kv, DIM]
    half* O,       // [bs, num_heads, len_q, DIM]
    const unsigned int bs,
    const unsigned int num_heads,
    const unsigned int len_q,
    const unsigned int len_kv
){
    constexpr unsigned int NUM_THREADS = NUM_WARPS * WARP_SIZE;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;

    const int q_block_id = blockIdx.x;
    const int head_id = blockIdx.y;
    const int bs_id = blockIdx.z;

   const size_t qo_offset =  (((size_t)bs_id * num_heads + head_id) * len_q + q_block_id * BLOCK_Q) * DIM;
   const size_t kv_offset =  ((size_t)bs_id * num_heads + head_id) * len_kv * DIM;

   Q += qo_offset;
   K += kv_offset;
   V += kv_offset;
   O += qo_offset;

   extern __shared__ half smem[];
   const uint32_t Q_smem = __cvta_generic_to_shared(smem);
    const uint32_t K_smem = Q_smem + BLOCK_Q * DIM * sizeof(half);
   const uint32_t V_smem = K_smem + BLOCK_KV * DIM * sizeof(half);

   constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;
   constexpr int MMA_M = 16;
   constexpr int MMA_N = 8;
   constexpr int MMA_K = 16;

   uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4];
   uint32_t K_rmem[BLOCK_KV / MMA_N][DIM / MMA_K][2];
   uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];
   uint32_t V_rmem[BLOCK_KV / MMA_K][DIM / MMA_N][2];

   float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4] = {};

   const float softmax_scale = rsqrtf(static_cast<float>(DIM));

   float rowmax[WARP_Q / MMA_M][2];
   float rowsumexp[WARP_Q / MMA_M][2] = {};
   #pragma unroll
  for (int i = 0; i < WARP_Q / MMA_M; i++) {
    rowmax[i][0] = -FLT_MAX;
    rowmax[i][1] = -FLT_MAX;
  }

    tileMemcpy<BLOCK_Q, DIM, NUM_THREADS>(Q_smem, Q, DIM, tid);
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_all;");
  __syncthreads();

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
            // For x4 loads, each lane supplies a row pointer within the 16-row MMA tile.
            const int row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id % 8) + ((lane_id / 8) % 2) * 8;
            const int col = mma_id_d * MMA_K + (lane_id / 16) * 8;
            const uint32_t off = (row * DIM + col) * sizeof(half);
            const uint32_t addr = Q_smem + swizzle<DIM * sizeof(half)>(off);
            ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], addr);
        }

  __syncthreads();

  
  for(int off_kv = 0; off_kv < len_kv; off_kv += BLOCK_KV){
    float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};

        tileMemcpy<BLOCK_KV, DIM, NUM_THREADS>(K_smem, K, DIM, tid);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    // K: smem -> rf
    for(int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++){
        for(int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++){
            const unsigned int row = mma_id_kv * MMA_N + (lane_id % 8);
            const unsigned int col = mma_id_d * MMA_K + (lane_id / 16 * 8);
            const uint32_t off = (row * DIM + col) * sizeof(half);
            const uint32_t addr = K_smem + swizzle<DIM * sizeof(half)>(off);
            ldmatrix_x2(K_rmem[mma_id_kv][mma_id_d], addr);
        }
    }

    // MMA: S = QK^T
    for(int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++){
        for(int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++){
            for(int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++){
                mma_m16n8k16(
                    Q_rmem[mma_id_q][mma_id_d],
                    K_rmem[mma_id_kv][mma_id_d],
                    S_rmem[mma_id_q][mma_id_kv]
                );
            }
        }
    }

    // softmax
    for(int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++){
        float this_rowmax[2] = {-FLT_MAX, -FLT_MAX};

        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        float *regs = S_rmem[mma_id_q][mma_id_kv];

        regs[0] *= softmax_scale;
        regs[1] *= softmax_scale;
        regs[2] *= softmax_scale;
        regs[3] *= softmax_scale;

        this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1]));
        this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3]));
      }

      // warp reduction
      this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[0], 1));
      this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[0], 2));
      this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[1], 1));
      this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[1], 2));

      this_rowmax[0] = max(this_rowmax[0], rowmax[mma_id_q][0]);
      this_rowmax[1] = max(this_rowmax[1], rowmax[mma_id_q][1]);

      float rescale[2];
      rescale[0] = __expf(rowmax[mma_id_q][0] - this_rowmax[0]);
      rescale[1] = __expf(rowmax[mma_id_q][1] - this_rowmax[1]);

      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
        float *o = O_rmem[mma_id_q][mma_id_d];
        o[0] *= rescale[0]; o[1] *= rescale[0];
        o[2] *= rescale[1]; o[3] *= rescale[1];
      }

      rowmax[mma_id_q][0] = this_rowmax[0];
      rowmax[mma_id_q][1] = this_rowmax[1];

      float this_rowsumexp[2] = {};

      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        float *regs = S_rmem[mma_id_q][mma_id_kv];

        regs[0] = __expf(regs[0] - rowmax[mma_id_q][0]);
        regs[1] = __expf(regs[1] - rowmax[mma_id_q][0]);
        regs[2] = __expf(regs[2] - rowmax[mma_id_q][1]);
        regs[3] = __expf(regs[3] - rowmax[mma_id_q][1]);

        this_rowsumexp[0] += regs[0] + regs[1];
        this_rowsumexp[1] += regs[2] + regs[3];

                half2 *p = reinterpret_cast<half2 *>(P_rmem[mma_id_q][mma_id_kv / 2]);

                p[(mma_id_kv % 2) * 2]     = __floats2half2_rn(regs[0], regs[1]);
                p[(mma_id_kv % 2) * 2 + 1] = __floats2half2_rn(regs[2], regs[3]);
      }

      this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 1);
      this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 2);
      this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 1);
      this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 2);

      rowsumexp[mma_id_q][0] =
          rowsumexp[mma_id_q][0] * rescale[0] + this_rowsumexp[0];
      rowsumexp[mma_id_q][1] =
          rowsumexp[mma_id_q][1] * rescale[1] + this_rowsumexp[1];
    }

    // load V
    tileMemcpy<BLOCK_KV, DIM, NUM_THREADS>(V_smem, V, DIM, tid);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    for(int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++){
        for(int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++){
            const uint32_t row = mma_id_kv * MMA_K + (lane_id % 16);
            const uint32_t col = mma_id_d * MMA_N;
            const uint32_t off = (row * DIM + col) * sizeof(half);
            const uint32_t addr = V_smem + swizzle<DIM * sizeof(half)>(off);
            ldmatrix_x2_trans(V_rmem[mma_id_kv][mma_id_d], addr);
        }
    }

    // MMA: O += PV
    for(int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++){
        for(int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++){
            for(int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++){
                mma_m16n8k16(
                    P_rmem[mma_id_q][mma_id_kv],
                    V_rmem[mma_id_kv][mma_id_d],
                    O_rmem[mma_id_q][mma_id_d]
                );
            }
        }
    }

    K += BLOCK_KV * DIM;
    V += BLOCK_KV * DIM;
  }

  for(int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++){
    for(int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++){
        const int row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
        const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;

        float *regs = O_rmem[mma_id_q][mma_id_d];

        regs[0] /= rowsumexp[mma_id_q][0];
        regs[1] /= rowsumexp[mma_id_q][0];
        regs[2] /= rowsumexp[mma_id_q][1];
        regs[3] /= rowsumexp[mma_id_q][1];

        reinterpret_cast<half2 *>(O + (row + 0) * DIM + col)[0] =
            __floats2half2_rn(regs[0], regs[1]);

        reinterpret_cast<half2 *>(O + (row + 8) * DIM + col)[0] =
            __floats2half2_rn(regs[2], regs[3]);
    }
  }
}