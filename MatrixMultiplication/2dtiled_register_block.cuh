#pragma once

template <int TILE_SIZE, int TM, int TN>
__global__ void tiled_2d_register_blocking(float *A, float *B, float *C, int M, int K, int N){
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, bx = blockIdx.x;
    int ty = threadIdx.y, by = blockIdx.y;

    int row = by * TILE_SIZE + ty * TM;
    int col = bx * TILE_SIZE + tx * TN;
    
    float sum[TM][TN] = {0.0f};

    #pragma unroll 1
    for(int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++){
        for(int j = 0; j < TN; j++){
            for(int i = 0; i < TM; i++){
                if(row + i < M && tile * TILE_SIZE + tx * TN + j < K){
                    As[ty * TM + i][tx * TN + j] = A[(row + i) * K + tile * TILE_SIZE + tx * TN + j];
                } else{
                    As[ty * TM + i][tx * TN + j] = 0.0f;
                }
            }
        }

        for(int j = 0; j < TN; j++){
            for(int i = 0; i < TM; i++){
                if(tile * TILE_SIZE + ty * TM + i < K && col + j < N){
                    Bs[ty * TM + i][tx * TN + j] = B[(tile * TILE_SIZE + ty * TM + i) * N + col + j];
                } else{
                    Bs[ty * TM + i][tx * TN + j] = 0.0f;
                }
            }
        }

        __syncthreads();

        #pragma unroll 4
        for(int k = 0; k < TILE_SIZE; k++){
            float av[TM];
            #pragma unroll
            for(int i = 0; i < TM; i++){
                av[i] = As[ty * TM + i][k];
            }

            float bv[TN];
            for(int j = 0; j < TN; j++){
                bv[j] = Bs[k][tx * TN + j];
            }

            #pragma unroll
            for(int i = 0; i < TM; i++){
                for(int j = 0; j < TN; j++){
                    sum[i][j] += av[i] * bv[j];
                }
            }
        }

        __syncthreads();
    }

    for(int i = 0; i < TM; i++){
        for(int j = 0; j < TN; j++){
            int r = row + i;
            int c = col + j;
            if(r < M && c < N){
                C[r * N + c] = sum[i][j];
            }
        }
    }
}