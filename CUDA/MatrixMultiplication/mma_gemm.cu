#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

constexpr __host__ __device__ unsigned int int_log2(unsigned int x) {
	return (x <= 1u) ? 0u : (1u + int_log2(x >> 1));
}

__device__ __forceinline__ uint32_t cvta_to_shared_u32(const void *pointer) {
	uint32_t address;
	asm("{\n\t"
		"  .reg .u64 u64addr;\n\t"
		"  cvta.to.shared.u64 u64addr, %1;\n\t"
		"  cvt.u32.u64 %0, u64addr;\n\t"
		"}"
		: "=r"(address)
		: "l"(pointer));
	return address;
}

__device__ __forceinline__ void cp_async_ca_16B(void* dst_smem, const void* src_gmem) {
	uint32_t dst_addr = cvta_to_shared_u32(dst_smem);
	asm volatile(
		"cp.async.cg.shared.global [%0], [%1], 16;\n"
		:: "r"(dst_addr), "l"(src_gmem)
	);
}

__device__ __forceinline__ void cp_async_commit_group() {
	asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
	asm volatile("cp.async.wait_group 0;\n");
}

__device__ __forceinline__ void stmatrix_m16n8(
	half* dst,
	half (&reg)[4],
	unsigned int dst_stride_bytes
)
{
	const unsigned int laneIdx = threadIdx.x % 32;
	uint32_t (&reg_) [2] = reinterpret_cast<uint32_t(&)[2]>(reg);
	uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(dst);
	dst_stride_bytes /= sizeof(uint32_t);
	unsigned int fragment_row = laneIdx / 4;
	const unsigned int fragment_col = laneIdx % 4;

	dst_ptr[fragment_row * dst_stride_bytes + fragment_col] = reg_[0];
	fragment_row += 8;
	dst_ptr[fragment_row * dst_stride_bytes + fragment_col] = reg_[1];
}

__device__ __forceinline__ void ldmatrix_m16n8_gmem(
	half* src,
	half (&reg)[4],
	unsigned int src_stride_bytes
)
{
	const unsigned int laneIdx = threadIdx.x % 32;
	uint32_t (&reg_) [2] = reinterpret_cast<uint32_t(&)[2]>(reg);
	uint32_t* src_ptr = reinterpret_cast<uint32_t*>(src);
	src_stride_bytes /= sizeof(uint32_t);
	unsigned int fragment_row = laneIdx / 4;
	const unsigned int fragment_col = laneIdx % 4;

	reg_[0] = src_ptr[fragment_row * src_stride_bytes + fragment_col];
	fragment_row += 8;
	reg_[1] = src_ptr[fragment_row * src_stride_bytes + fragment_col];
}

template <unsigned int TM, unsigned int TK, unsigned int STRIDE, unsigned int SWIZZLE_MASK>
__device__ __forceinline__ void ldmatrix(
	half* src,
	half (&dst_reg)[TM][TK][4]
)
{
	const unsigned int lane = threadIdx.x & 31u;
	for (unsigned int mma_m = 0; mma_m < TM; mma_m++) {
		for (unsigned int mma_k = 0; mma_k < TK; mma_k++) {
			const unsigned int ldm_row = lane & 7u;
			const unsigned int ldm_mat = (lane >> 3u) & 1u;
			const unsigned int a_row = mma_m * 16u + ldm_mat * 8u + ldm_row;
			const unsigned int a_col = mma_k * 8u;
			const unsigned int a_col_vec = a_col >> 3u;
			const unsigned int a_col_vec_swizzled = a_col_vec ^ (a_row & SWIZZLE_MASK);
			const unsigned int a_col_swizzled = a_col_vec_swizzled << 3u;
			uint32_t thread_addr = cvta_to_shared_u32(src + a_row * STRIDE + a_col_swizzled);

			uint32_t (&dst_u32)[2] = reinterpret_cast<uint32_t(&)[2]>(dst_reg[mma_m][mma_k]);
			asm volatile(
				"ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
				"{%0, %1}, [%2];"
				: "=r"(dst_u32[0]), "=r"(dst_u32[1])
				: "r"(thread_addr)
			);
		}
	}
}

template <unsigned int TK, unsigned int TN, unsigned int STRIDE, unsigned int SWIZZLE_MASK>
__device__ __forceinline__ void ldmatrix(
	half* src,
	half (&dst_reg)[TK][TN][2]
)
{
	const unsigned int lane = threadIdx.x & 31u;
	for (unsigned int mma_k = 0; mma_k < TK; mma_k++) {
		for (unsigned int mma_n = 0; mma_n < TN; mma_n++) {
			const unsigned int b_row = lane & 7u;
			const unsigned int b_col = mma_n * 8u;
			const unsigned int b_k = mma_k * 8u + b_row;
			const unsigned int b_col_vec = b_col >> 3u;
			const unsigned int b_col_vec_swizzled = b_col_vec ^ (b_k & SWIZZLE_MASK);
			const unsigned int b_col_swizzled = b_col_vec_swizzled << 3u;
			uint32_t thread_addr = cvta_to_shared_u32(src + b_k * STRIDE + b_col_swizzled);
			uint32_t& dst_u32 = reinterpret_cast<uint32_t&>(dst_reg[mma_k][mma_n]);
			asm volatile(
				"ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 "
				"{%0}, [%1];"
				: "=r"(dst_u32)
				: "r"(thread_addr)
			);
		}
	}
}

template<
unsigned int TILE_ROWS,
unsigned int TILE_COLS,
unsigned int NUM_THREADS,
unsigned int SWIZZLE_BITS
>
__device__ __forceinline__ void tileMemcpySwizzle(
	half* src,
	half* dst,
	const unsigned int src_stride
){
	constexpr unsigned int SWIZZLE_MASK =
		(SWIZZLE_BITS == 0u) ? 0u : ((1u << SWIZZLE_BITS) - 1u);

	float4* src_float4 = reinterpret_cast<float4*>(src);
	float4* dst_float4 = reinterpret_cast<float4*>(dst);
	const unsigned int src_stride_vectorized = src_stride / 8;

	constexpr unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
	static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0, "NUM_THREADS must divide vectorized tile columns");

	const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

	constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
	constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;
	unsigned int thread_row = thread_idx / TILE_COLS_VECTORIZED;
	const unsigned int thread_col = thread_idx % TILE_COLS_VECTORIZED;

	#pragma unroll
	for(unsigned int i = 0; i < NUM_ITERS; i++){
		const unsigned int src_index = thread_row * src_stride_vectorized + thread_col;
		const unsigned int dst_col = thread_col ^ (thread_row & SWIZZLE_MASK);
		const unsigned int dst_index = thread_row * TILE_COLS_VECTORIZED + dst_col;
		cp_async_ca_16B(&dst_float4[dst_index], &src_float4[src_index]);
		thread_row += ROW_STEP;
	}
	cp_async_commit_group();
}

// need seperate swizzle logic for A since TILE_COL = 32 due to which we end up with less than 8 chunks, hence swizzle bits will not be 0b111
template<
unsigned int TILE_ROWS,
unsigned int NUM_THREADS
>
__device__ __forceinline__ void tileMemCpySwizzleA(
	half* src,
	half* dst,
	const unsigned int src_stride
){
	constexpr unsigned int SWIZZLE_MASK_1 = 0b10000;
	constexpr unsigned int SWIZZLE_BITS_1 = 4;
	constexpr unsigned int SWIZZLE_MASK_2 = 0b1100;
	constexpr unsigned int SWIZZLE_BITS_2 = 2;
	constexpr unsigned int TILE_COLS = 32;

	float4* src_float4 = reinterpret_cast<float4*>(src);
	float4* dst_float4 = reinterpret_cast<float4*>(dst);

	const unsigned int src_stride_vectorized = src_stride / 8;

	constexpr unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
	static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0, "NUM_THREADS must divide vectorized tile columns");

	const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

	constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
	constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;
	unsigned int thread_row = thread_idx / TILE_COLS_VECTORIZED;
	const unsigned int thread_col = thread_idx % TILE_COLS_VECTORIZED;

	#pragma unroll
	for(unsigned int i = 0; i < NUM_ITERS; i++){
		const unsigned int src_index = thread_row * src_stride_vectorized + thread_col;
		unsigned int dst_index = thread_row * TILE_COLS_VECTORIZED + thread_col;
		dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_1) >> SWIZZLE_BITS_1);
		dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_2) >> SWIZZLE_BITS_2);
		dst_float4[dst_index] =  src_float4[src_index];
		thread_row += ROW_STEP;
	}
}


template<
unsigned int TILE_ROWS,
unsigned int TILE_COLS,
unsigned int NUM_THREADS,
unsigned int ELEMENTS_PER_THREAD>
__device__ __forceinline__ void tileMemcpyLoad(
	half* src,
	float4 (&dst_reg)[ELEMENTS_PER_THREAD],
	const unsigned int src_stride
){
	float4* src_float4 = reinterpret_cast<float4*>(src);
	const unsigned int src_stride_vectorized = src_stride / 8;

	constexpr unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
	static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0, "NUM_THREADS must divide vectorized tile columns");

	const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

	constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
	constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;
	unsigned int thread_row = thread_idx / TILE_COLS_VECTORIZED;
	const unsigned int thread_col = thread_idx % TILE_COLS_VECTORIZED;

	static_assert(ELEMENTS_PER_THREAD == NUM_ITERS, "ELEMENTS_PER_THREAD must match NUM_ITERS");

	#pragma unroll
	for(unsigned int i = 0; i < NUM_ITERS; i++){
		const unsigned int src_index = thread_row * src_stride_vectorized + thread_col;
		dst_reg[i] = src_float4[src_index];
		thread_row += ROW_STEP;
	}
}


template<
unsigned int TILE_ROWS,
unsigned int TILE_COLS,
unsigned int NUM_THREADS,
unsigned int SWIZZLE_BITS,
unsigned int ELEMENTS_PER_THREAD
>
__device__ __forceinline__ void tileMemcpySwizzleStore(
	float4 src_reg[ELEMENTS_PER_THREAD],
	half* dst
){
	constexpr unsigned int SWIZZLE_MASK =
		(SWIZZLE_BITS == 0u) ? 0u : ((1u << SWIZZLE_BITS) - 1u);
	float4* dst_float4 = reinterpret_cast<float4*>(dst);
	constexpr unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
	static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0, "NUM_THREADS must divide vectorized tile columns");

	const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

	constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
	constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;
	unsigned int thread_row = thread_idx / TILE_COLS_VECTORIZED;
	const unsigned int thread_col = thread_idx % TILE_COLS_VECTORIZED;

	static_assert(ELEMENTS_PER_THREAD == NUM_ITERS, "ELEMENTS_PER_THREAD must match NUM_ITERS");

	#pragma unroll
	for(unsigned int i = 0; i < NUM_ITERS; i++){
		const unsigned int dst_col = thread_col ^ (thread_row & SWIZZLE_MASK);
		const unsigned int dst_index = thread_row * TILE_COLS_VECTORIZED + dst_col;
		dst_float4[dst_index] = src_reg[i];
		thread_row += ROW_STEP;
	}
}

template<
unsigned int TILE_ROWS,
unsigned int NUM_THREADS,
unsigned int ELEMENTS_PER_THREAD
>
__device__ __forceinline__ void tileMemcpySwizzleStoreA(
	const float4 (&src_reg)[ELEMENTS_PER_THREAD],
	half* dst
){
	constexpr unsigned int SWIZZLE_MASK_1 = 0b10000;
	constexpr unsigned int SWIZZLE_BITS_1 = 4;
	constexpr unsigned int SWIZZLE_MASK_2 = 0b1100;
	constexpr unsigned int SWIZZLE_BITS_2 = 2;
	constexpr unsigned int TILE_COLS = 32;

	float4* dst_float4 = reinterpret_cast<float4*>(dst);

	constexpr unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;
	static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0, "NUM_THREADS must divide vectorized tile columns");
    
	const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

	constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS_VECTORIZED;
	constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;
	unsigned int thread_row = thread_idx / TILE_COLS_VECTORIZED;
	const unsigned int thread_col = thread_idx % TILE_COLS_VECTORIZED;

	static_assert(ELEMENTS_PER_THREAD == NUM_ITERS, "ELEMENTS_PER_THREAD must match NUM_ITERS");
    
	#pragma unroll
	for(unsigned int i = 0; i < NUM_ITERS; i++){
		unsigned int dst_index = thread_row * TILE_COLS_VECTORIZED + thread_col;
		dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_1) >> SWIZZLE_BITS_1);
		dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK_2) >> SWIZZLE_BITS_2);
		dst_float4[dst_index] =  src_reg[i];
		thread_row += ROW_STEP;
	}
}

template<
unsigned int BM_dim,
unsigned int BN_dim,
unsigned int BK_dim,
unsigned int WM_dim,
unsigned int WN_dim,
unsigned int WK_dim,
unsigned int NUM_THREADS
>
__global__ void mma_fp16_gemm(
	half* A, half* B, half* C, half* D,
	const float alpha, const float beta,
	const unsigned int M, const unsigned int N, unsigned int K
){
	constexpr unsigned int MMA_M_dim = 16;
	constexpr unsigned int MMA_N_dim = 8;
    constexpr unsigned int MMA_K_dim = 8;
    
	constexpr unsigned int SWIZZLE_BITS_A_RAW = int_log2(BK_dim / 8);
	constexpr unsigned int SWIZZLE_BITS_B_RAW = int_log2(BN_dim / 8);
	constexpr unsigned int SWIZZLE_BITS_A = (SWIZZLE_BITS_A_RAW > 3u) ? 3u : SWIZZLE_BITS_A_RAW;
	constexpr unsigned int SWIZZLE_BITS_B = (SWIZZLE_BITS_B_RAW > 3u) ? 3u : SWIZZLE_BITS_B_RAW;
	constexpr unsigned int SWIZZLE_MASK_A = (SWIZZLE_BITS_A == 0u) ? 0u : ((1u << SWIZZLE_BITS_A) - 1u);
	constexpr unsigned int SWIZZLE_MASK_B = (SWIZZLE_BITS_B == 0u) ? 0u : ((1u << SWIZZLE_BITS_B) - 1u);

	constexpr unsigned int mma_tiles_per_warp_k = 4;
	constexpr unsigned int mma_tiles_per_warp_m = WM_dim / MMA_M_dim;
	constexpr unsigned int mma_tiles_per_warp_n = WN_dim / MMA_N_dim;
	constexpr unsigned int WARPS_M = BM_dim / WM_dim;
	constexpr unsigned int WARPS_N = BN_dim / WN_dim;
	const unsigned int num_block_tiles_k = K / BK_dim;

	const unsigned int block_m = blockIdx.y;
	const unsigned int block_n = blockIdx.x;
	const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
	const unsigned int warp_id = tid >> 5;
	const unsigned int warp_m = warp_id / WARPS_N;
	const unsigned int warp_n = warp_id % WARPS_N;
    
	extern __shared__ half shmem[];
	half* A_block_smem = shmem;
	half* B_block_smem = &shmem[BM_dim * BK_dim];
	const unsigned int CD_stride = N;
	constexpr unsigned int A_STAGE_ELEMS = BM_dim * BK_dim;
	constexpr unsigned int B_STAGE_ELEMS = BK_dim * BN_dim;
	constexpr unsigned int STAGE_ELEMS = A_STAGE_ELEMS + B_STAGE_ELEMS;

	uint32_t acc_register[mma_tiles_per_warp_m][mma_tiles_per_warp_n][2];
	uint32_t A_register[mma_tiles_per_warp_m][mma_tiles_per_warp_k][2];
	uint32_t B_register[mma_tiles_per_warp_k][mma_tiles_per_warp_n];

	half (&acc_register_) [mma_tiles_per_warp_m][mma_tiles_per_warp_n][4] = reinterpret_cast<half(&)[mma_tiles_per_warp_m][mma_tiles_per_warp_n][4]>(acc_register);
	half (&A_register_) [mma_tiles_per_warp_m][mma_tiles_per_warp_k][4] = reinterpret_cast<half(&)[mma_tiles_per_warp_m][mma_tiles_per_warp_k][4]>(A_register);
	half (&B_register_) [mma_tiles_per_warp_k][mma_tiles_per_warp_n][2] = reinterpret_cast<half(&)[mma_tiles_per_warp_k][mma_tiles_per_warp_n][2]>(B_register);
    
	for (unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++){
		for (unsigned int mma_n = 0; mma_n < mma_tiles_per_warp_n; mma_n++){
			acc_register_[mma_m][mma_n][0] = 0;
			acc_register_[mma_m][mma_n][1] = 0;
			acc_register_[mma_m][mma_n][2] = 0;
			acc_register_[mma_m][mma_n][3] = 0;
		}
	}

	static_assert(BK_dim == 32, "Kernel currently supports BK=32");
	static_assert(NUM_THREADS == 256, "Kernel currently supports 256 threads per block");
	static_assert(BM_dim % WM_dim == 0, "BM must be divisible by WM");
	static_assert(BN_dim % WN_dim == 0, "BN must be divisible by WN");
	static_assert(WM_dim % MMA_M_dim == 0, "WM must be divisible by MMA_M");
	static_assert(WN_dim % MMA_N_dim == 0, "WN must be divisible by MMA_N");
	half* A_block_gmem = A + (block_m * BM_dim * K);
	half* B_block_gmem = B + (block_n * BN_dim);
	tileMemcpySwizzle<BM_dim, BK_dim, NUM_THREADS, SWIZZLE_BITS_A>(A_block_gmem, A_block_smem, K);
	tileMemcpySwizzle<BK_dim, BN_dim, NUM_THREADS, SWIZZLE_BITS_B>(B_block_gmem, B_block_smem, N);

	for(unsigned int block_k = 0; block_k < num_block_tiles_k; block_k++){
		cp_async_wait_all();
		__syncthreads();

		half* A_stage = A_block_smem + ((block_k & 1u) * STAGE_ELEMS);
		half* B_stage = A_stage + A_STAGE_ELEMS;

		if((block_k + 1u) < num_block_tiles_k){
			half* A_next_stage = A_block_smem + (((block_k + 1u) & 1u) * STAGE_ELEMS);
			half* B_next_stage = A_next_stage + A_STAGE_ELEMS;
			half* A_next_gmem = A + (block_m * BM_dim * K) + ((block_k + 1u) * BK_dim);
			half* B_next_gmem = B + ((block_k + 1u) * BK_dim * N) + (block_n * BN_dim);
			tileMemcpySwizzle<BM_dim, BK_dim, NUM_THREADS, SWIZZLE_BITS_A>(A_next_gmem, A_next_stage, K);
			tileMemcpySwizzle<BK_dim, BN_dim, NUM_THREADS, SWIZZLE_BITS_B>(B_next_gmem, B_next_stage, N);
		}

		half* A_warp_tile = A_stage + (warp_m * WM_dim * BK_dim);
		half* B_warp_tile = B_stage + (warp_n * WN_dim);

		// TODO: fix later
		ldmatrix<mma_tiles_per_warp_m, mma_tiles_per_warp_k, BK_dim, SWIZZLE_MASK_A>(A_warp_tile, A_register_);
		ldmatrix<mma_tiles_per_warp_k, mma_tiles_per_warp_n, BN_dim, SWIZZLE_MASK_B>(B_warp_tile, B_register_);

		#pragma unroll
		for(unsigned int mma_k = 0; mma_k < mma_tiles_per_warp_k; mma_k++){
			#pragma unroll
			for(unsigned int mma_n = 0; mma_n < mma_tiles_per_warp_n; mma_n++){
				#pragma unroll
				for(unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++){
					asm volatile(
						"mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
						"{%0, %1}, "
						"{%2, %3}, "
						"{%4}, "
						"{%5, %6};"
						: "+r"(acc_register[mma_m][mma_n][0]), "+r"(acc_register[mma_m][mma_n][1])
						: "r"(A_register[mma_m][mma_k][0]), "r"(A_register[mma_m][mma_k][1]),
						  "r"(B_register[mma_k][mma_n]),
						  "r"(acc_register[mma_m][mma_n][0]), "r"(acc_register[mma_m][mma_n][1])
					);
				}
			}
		}

	}


	//////////////
	// epilogue //
	//////////////
	half alpha_ = (half)alpha;
	half beta_ = (half)beta;
	half C_register[mma_tiles_per_warp_m][mma_tiles_per_warp_n][4];
    
	half* C_block_gmem = C + (block_m * BM_dim * CD_stride) + (block_n * BN_dim);
	half* C_warp_gmem = C_block_gmem + (warp_m * WM_dim * CD_stride) + (warp_n * WN_dim);
	half* D_block_gmem = D + (block_m * BM_dim * CD_stride) + (block_n * BN_dim);
	half* D_warp_gmem = D_block_gmem + (warp_m * WM_dim * CD_stride) + (warp_n * WN_dim);

	for (unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++) {
		for (unsigned int mma_n = 0; mma_n < mma_tiles_per_warp_n; mma_n++) {
			if (beta != 0.0f) {
				half* C_mma_tile = C_warp_gmem + (mma_m * MMA_M_dim * CD_stride) + (mma_n * MMA_N_dim);
				ldmatrix_m16n8_gmem(C_mma_tile, C_register[mma_m][mma_n], N * sizeof(half));

				acc_register_[mma_m][mma_n][0] = acc_register_[mma_m][mma_n][0] * alpha_ + C_register[mma_m][mma_n][0] * beta_;
				acc_register_[mma_m][mma_n][1] = acc_register_[mma_m][mma_n][1] * alpha_ + C_register[mma_m][mma_n][1] * beta_;
				acc_register_[mma_m][mma_n][2] = acc_register_[mma_m][mma_n][2] * alpha_ + C_register[mma_m][mma_n][2] * beta_;
				acc_register_[mma_m][mma_n][3] = acc_register_[mma_m][mma_n][3] * alpha_ + C_register[mma_m][mma_n][3] * beta_;
			} else {
				acc_register_[mma_m][mma_n][0] = acc_register_[mma_m][mma_n][0] * alpha_;
				acc_register_[mma_m][mma_n][1] = acc_register_[mma_m][mma_n][1] * alpha_;
				acc_register_[mma_m][mma_n][2] = acc_register_[mma_m][mma_n][2] * alpha_;
				acc_register_[mma_m][mma_n][3] = acc_register_[mma_m][mma_n][3] * alpha_;
			}
		}
	}

	for (unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++)
	{
		for (unsigned int mma_n = 0; mma_n < mma_tiles_per_warp_n; mma_n++)
		{
			half* D_mma_tile = D_warp_gmem + (mma_m * MMA_M_dim * CD_stride) + (mma_n * MMA_N_dim);
			stmatrix_m16n8(D_mma_tile, acc_register_[mma_m][mma_n], N * sizeof(half));
		}
	}

}

#define CUDA_CHECK(call)                                                                  \
	do {                                                                                  \
		cudaError_t err__ = (call);                                                      \
		if (err__ != cudaSuccess) {                                                      \
			std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
						 cudaGetErrorString(err__));                                     \
			std::exit(EXIT_FAILURE);                                                      \
		}                                                                                 \
	} while (0)

#define CUBLAS_CHECK(call)                                                                \
	do {                                                                                  \
		cublasStatus_t stat__ = (call);                                                  \
		if (stat__ != CUBLAS_STATUS_SUCCESS) {                                           \
			std::fprintf(stderr, "cuBLAS error %s:%d: status=%d\n", __FILE__, __LINE__, \
						 static_cast<int>(stat__));                                  \
			std::exit(EXIT_FAILURE);                                                      \
		}                                                                                 \
	} while (0)

struct Config {
	unsigned int M = 8192;
	unsigned int N = 8192;
	unsigned int K = 8192;
	int warmup = 3;
	int iters = 5;
	float alpha = 1.0f;
	float beta = 0.0f;
};

static bool parse_args(int argc, char** argv, Config& cfg) {
	if (argc == 1) {
		return true;
	}

	if (argc != 8) {
		std::fprintf(
			stderr,
			"Usage: %s [M N K warmup iters alpha beta]\n"
			"Example: %s 8192 8192 8192 5 20 1.0 0.0\n",
			argv[0], argv[0]
		);
		return false;
	}

	cfg.M = static_cast<unsigned int>(std::strtoul(argv[1], nullptr, 10));
	cfg.N = static_cast<unsigned int>(std::strtoul(argv[2], nullptr, 10));
	cfg.K = static_cast<unsigned int>(std::strtoul(argv[3], nullptr, 10));
	cfg.warmup = std::atoi(argv[4]);
	cfg.iters = std::atoi(argv[5]);
	cfg.alpha = static_cast<float>(std::atof(argv[6]));
	cfg.beta = static_cast<float>(std::atof(argv[7]));
	return true;
}

int main(int argc, char** argv) {
	constexpr unsigned int BM = 256;
	constexpr unsigned int BN = 256;
	constexpr unsigned int BK = 32;
	constexpr unsigned int WM = 128;
	constexpr unsigned int WN = 64;
	constexpr unsigned int WK = 32;
	constexpr unsigned int NUM_THREADS = 256;

	Config cfg;
	if (!parse_args(argc, argv, cfg)) {
		return 1;
	}

	if ((cfg.M % BM) || (cfg.N % BN) || (cfg.K % BK)) {
		std::fprintf(stderr,
					 "Error: M,N,K must be multiples of BM,BN,BK.\n"
					 "Got M=%u N=%u K=%u with BM=%u BN=%u BK=%u\n",
					 cfg.M, cfg.N, cfg.K, BM, BN, BK);
		return 1;
	}

	int dev = 0;
	CUDA_CHECK(cudaSetDevice(dev));
	cudaDeviceProp prop{};
	CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

	const size_t bytesA = static_cast<size_t>(cfg.M) * cfg.K * sizeof(half);
	const size_t bytesB = static_cast<size_t>(cfg.K) * cfg.N * sizeof(half);
	const size_t bytesC = static_cast<size_t>(cfg.M) * cfg.N * sizeof(half);

	std::vector<half> hA(static_cast<size_t>(cfg.M) * cfg.K);
	std::vector<half> hB(static_cast<size_t>(cfg.K) * cfg.N);
	std::vector<half> hC(static_cast<size_t>(cfg.M) * cfg.N);

	std::mt19937 rng(1234);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	for (size_t i = 0; i < hA.size(); ++i) hA[i] = __float2half(dist(rng));
	for (size_t i = 0; i < hB.size(); ++i) hB[i] = __float2half(dist(rng));
	for (size_t i = 0; i < hC.size(); ++i) hC[i] = __float2half(dist(rng));

	half *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr;
	CUDA_CHECK(cudaMalloc(&dA, bytesA));
	CUDA_CHECK(cudaMalloc(&dB, bytesB));
	CUDA_CHECK(cudaMalloc(&dC, bytesC));
	CUDA_CHECK(cudaMalloc(&dD, bytesC));

	CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dC, hC.data(), bytesC, cudaMemcpyHostToDevice));

	dim3 block(32, NUM_THREADS / 32, 1);
	dim3 grid(cfg.N / BN, cfg.M / BM, 1);
	const size_t single_buffer_smem_bytes =
		static_cast<size_t>(BM) * BK * sizeof(half) +
		static_cast<size_t>(BK) * BN * sizeof(half);
	const size_t shmem_bytes = 2 * single_buffer_smem_bytes;

	using KernelPtr = void (*)(half*, half*, half*, half*, float, float,
		unsigned int, unsigned int, unsigned int);
	KernelPtr kernel = mma_fp16_gemm<BM, BN, BK, WM, WN, WK, NUM_THREADS>;
	CUDA_CHECK(cudaFuncSetAttribute(
		kernel,
		cudaFuncAttributeMaxDynamicSharedMemorySize,
		static_cast<int>(shmem_bytes)
	));

	for (int i = 0; i < cfg.warmup; ++i) {
		mma_fp16_gemm<BM, BN, BK, WM, WN, WK, NUM_THREADS><<<grid, block, shmem_bytes>>>(
			dA, dB, dC, dD, cfg.alpha, cfg.beta, cfg.M, cfg.N, cfg.K
		);
	}
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	CUDA_CHECK(cudaEventRecord(start));
	for (int i = 0; i < cfg.iters; ++i) {
		mma_fp16_gemm<BM, BN, BK, WM, WN, WK, NUM_THREADS><<<grid, block, shmem_bytes>>>(
			dA, dB, dC, dD, cfg.alpha, cfg.beta, cfg.M, cfg.N, cfg.K
		);
	}
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));
	CUDA_CHECK(cudaGetLastError());

	float ms = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
	const float ms_per_iter = ms / static_cast<float>(cfg.iters);
	const double flops = 2.0 * static_cast<double>(cfg.M) * cfg.N * cfg.K;
	const double tflops = flops / (ms_per_iter * 1.0e-3) / 1.0e12;

	std::printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
	std::printf("Shape: M=%u N=%u K=%u | warmup=%d iters=%d | alpha=%.2f beta=%.2f\n",
				cfg.M, cfg.N, cfg.K, cfg.warmup, cfg.iters, cfg.alpha, cfg.beta);
	std::printf("Time: %.3f ms/iter, Throughput: %.3f TFLOP/s\n", ms_per_iter, tflops);

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));
	CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));
	half cublas_alpha_h = __float2half(cfg.alpha);
	half cublas_beta_h = __float2half(cfg.beta);

	for (int i = 0; i < cfg.warmup; ++i) {
		CUBLAS_CHECK(cublasGemmEx(
			cublas_handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			static_cast<int>(cfg.N), static_cast<int>(cfg.M), static_cast<int>(cfg.K),
			&cublas_alpha_h,
			dB, CUDA_R_16F, static_cast<int>(cfg.N),
			dA, CUDA_R_16F, static_cast<int>(cfg.K),
			&cublas_beta_h,
			dD, CUDA_R_16F, static_cast<int>(cfg.N),
			CUDA_R_16F,
			CUBLAS_GEMM_DEFAULT_TENSOR_OP
		));
	}
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaEventRecord(start));
	for (int i = 0; i < cfg.iters; ++i) {
		CUBLAS_CHECK(cublasGemmEx(
			cublas_handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			static_cast<int>(cfg.N), static_cast<int>(cfg.M), static_cast<int>(cfg.K),
			&cublas_alpha_h,
			dB, CUDA_R_16F, static_cast<int>(cfg.N),
			dA, CUDA_R_16F, static_cast<int>(cfg.K),
			&cublas_beta_h,
			dD, CUDA_R_16F, static_cast<int>(cfg.N),
			CUDA_R_16F,
			CUBLAS_GEMM_DEFAULT_TENSOR_OP
		));
	}
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));
	CUDA_CHECK(cudaGetLastError());

	float cublas_ms = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&cublas_ms, start, stop));
	const float cublas_ms_per_iter = cublas_ms / static_cast<float>(cfg.iters);
	const double cublas_tflops = flops / (cublas_ms_per_iter * 1.0e-3) / 1.0e12;
	std::printf("cuBLAS (FP16 accumulate): %.3f ms/iter, Throughput: %.3f TFLOP/s\n", cublas_ms_per_iter, cublas_tflops);

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));
	CUBLAS_CHECK(cublasDestroy(cublas_handle));
	CUDA_CHECK(cudaFree(dA));
	CUDA_CHECK(cudaFree(dB));
	CUDA_CHECK(cudaFree(dC));
	CUDA_CHECK(cudaFree(dD));
	return 0;
}
