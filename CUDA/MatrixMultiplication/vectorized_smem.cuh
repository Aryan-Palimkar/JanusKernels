#pragma once

template <int TILE_SIZE, int TM, int TN>
__global__ void tiled_2d_register_blocking_vectorized(float *A, float *B, float *C, int M, int K, int N){
    __shared__ float4 As[TILE_SIZE][TILE_SIZE / 4];
    __shared__ float4 Bs[TILE_SIZE][TILE_SIZE / 4];

    int tx = threadIdx.x, bx = blockIdx.x;
    int ty = threadIdx.y, by = blockIdx.y;

    int row = by * TILE_SIZE + ty * TM;
    int col = bx * TILE_SIZE + tx * TN;

    float sum[TM][TN] = {0.0f};

    #pragma unroll 1
    for(int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++){
        for(int j = 0; j < TN / 4; j++){
            for(int i = 0; i < TM; i++){
                int a_row = ty * TM + i;
                int a_col = (tx * (TN / 4) + j) * 4;
                int g_row = by * TILE_SIZE + a_row;
                int g_col = tile * TILE_SIZE + a_col;
                float4 tmp = {0.0f, 0.0f, 0.0f, 0.0f};
                if(g_row < M && g_col + 3 < K){
                    tmp = *reinterpret_cast<float4*>(&A[g_row * K + g_col]);
                }else {
                    if(g_row < M){
                        if (g_col + 0 < K) tmp.x = A[g_row * K + g_col + 0];
                        if (g_col + 1 < K) tmp.y = A[g_row * K + g_col + 1];
                        if (g_col + 2 < K) tmp.z = A[g_row * K + g_col + 2];
                        if (g_col + 3 < K) tmp.w = A[g_row * K + g_col + 3];
                    }
                }
                As[a_row][tx * (TN / 4) + j] = tmp;
            }
        }

        for(int j = 0; j < TN / 4; j++){
            for(int i = 0; i < TM; i++){
                int b_row = ty * TM + i;
                int b_col = (tx * (TN / 4) + j) * 4;
                int g_row = tile * TILE_SIZE + b_row;
                int g_col = bx * TILE_SIZE + b_col;
                float4 tmp = {0.0f, 0.0f, 0.0f, 0.0f};
                if(g_row < K && g_col + 3 < N){
                    tmp = *reinterpret_cast<float4*>(&B[g_row * N + g_col]);
                }else {
                    if(g_row < K){
                        if (g_col + 0 < N) tmp.x = B[g_row * N + g_col + 0];
                        if (g_col + 1 < N) tmp.y = B[g_row * N + g_col + 1];
                        if (g_col + 2 < N) tmp.z = B[g_row * N + g_col + 2];
                        if (g_col + 3 < N) tmp.w = B[g_row * N + g_col + 3];
                    }
                }
                Bs[b_row][tx * (TN / 4) + j] = tmp;
            }
        }

        __syncthreads();

        #pragma unroll 2
        for(int k4 = 0; k4 < TILE_SIZE / 4; ++k4){
            float4 a4[TM];
            #pragma unroll
            for(int i = 0; i < TM; i++){
                a4[i] = As[ty * TM + i][k4];
            }

            #pragma unroll
            for(int sub = 0; sub < 4; ++sub){
                float av[TM];
                #pragma unroll
                for(int i = 0; i < TM; i++){
                    float4 t = a4[i];
                    av[i] = (sub == 0 ? t.x : (sub == 1 ? t.y : (sub == 2 ? t.z : t.w)));
                }

                float bv[TN];
                #pragma unroll
                for(int j = 0; j < TN / 4; j++){
                    float4 tmp = Bs[k4 * 4 + sub][tx * (TN / 4) + j];
                    bv[j * 4 + 0] = tmp.x;
                    bv[j * 4 + 1] = tmp.y;
                    bv[j * 4 + 2] = tmp.z;
                    bv[j * 4 + 3] = tmp.w;
                }

                #pragma unroll
                for(int i = 0; i < TM; i++){
                    for(int j = 0; j < TN; j++){
                        sum[i][j] += av[i] * bv[j];
                    }
                }
            }
        }

        __syncthreads();
    }

    for(int i = 0; i < TM; i++){
        for(int j = 0; j < TN / 4; j++){
            int r = row + i;
            int c = col + j * 4;
            if(r < M && c + 3 < N){
                float4 tmp = {sum[i][j * 4 + 0], sum[i][j * 4 + 1], sum[i][j * 4 + 2], sum[i][j * 4 + 3]};
                *reinterpret_cast<float4*>(&C[r * N + c]) = tmp;
            }else {
                if(r < M){
                    if (c + 0 < N) C[r * N + c + 0] = sum[i][j * 4 + 0];
                    if (c + 1 < N) C[r * N + c + 1] = sum[i][j * 4 + 1];
                    if (c + 2 < N) C[r * N + c + 2] = sum[i][j * 4 + 2];
                    if (c + 3 < N) C[r * N + c + 3] = sum[i][j * 4 + 3];
                }
            }
        }
    }
}