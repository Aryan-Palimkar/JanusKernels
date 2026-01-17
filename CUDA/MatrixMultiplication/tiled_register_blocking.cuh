#pragma once

template <int TILE_SIZE, int TM>
__global__ void tiled_register_blocking(float *A, float *B, float *C, int M, int K, int N){
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, bx = blockIdx.x;
    int ty = threadIdx.y, by = blockIdx.y;

    int row = by * TILE_SIZE + ty * TM;
    int col = bx * TILE_SIZE + tx;
    
    float sum[TM] = {0.0f};

    #pragma unroll 1
    for(int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++){
        #pragma unroll
        for(int i = 0; i < TM; i++){
            if(row + i < M && tile * TILE_SIZE + tx < K){
                As[ty * TM + i][tx] = A[(row+i)*K + tile*TILE_SIZE + tx];
            } else{
                As[ty * TM + i][tx] = 0.0f;
            }
        }

        #pragma unroll
        for(int i = 0; i < TM; i++){
            if(col < N && tile * TILE_SIZE + ty * TM + i < K){
                Bs[ty * TM + i][tx] = B[(tile * TILE_SIZE + ty * TM + i)*N + col];
            } else{
                Bs[ty * TM + i][tx] = 0.0f;
            }
        }

        __syncthreads();
        #pragma unroll 4
        for(int k = 0; k < TILE_SIZE; k++){
            float b = Bs[k][tx];
            #pragma unroll
            for(int i = 0; i < TM; i++){
                sum[i] += As[ty * TM + i][k] * b;
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for(int i = 0; i < TM; i++){
        if(row + i < M && col < N){
            C[(row+i) * N + col] = sum[i];
        }
    }
}
