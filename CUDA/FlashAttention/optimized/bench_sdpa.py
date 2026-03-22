import torch
import torch.nn.functional as F
import argparse

def benchmark_sdpa(bs=1, num_heads=16, seq_len=1024, dim=128, iters=100, warmup=20):
    device = 'cuda'
    dtype = torch.float16

    print(f"Allocating tensors: bs={bs}, heads={num_heads}, seq_len={seq_len}, dim={dim} on {device}")
    q = torch.randn(bs, num_heads, seq_len, dim, device=device, dtype=dtype)
    k = torch.randn(bs, num_heads, seq_len, dim, device=device, dtype=dtype)
    v = torch.randn(bs, num_heads, seq_len, dim, device=device, dtype=dtype)

    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        try:
            # Warmup
            for _ in range(warmup):
                _ = F.scaled_dot_product_attention(q, k, v)
        except RuntimeError as e:
            print(f"Flash Attention not supported on this GPU/env: {e}")
            print("Falling back to default SDPA behavior...")
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                for _ in range(warmup):
                    _ = F.scaled_dot_product_attention(q, k, v)
        
        torch.cuda.synchronize()
        
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(iters):
            _ = F.scaled_dot_product_attention(q, k, v)
        end_event.record()
        
        torch.cuda.synchronize()
        
        total_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = total_time_ms / iters

    flops = 4.0 * bs * num_heads * seq_len * seq_len * dim
    tflops = (flops / (avg_time_ms * 1e-3)) / 1e12

    print(f"\n--- PyTorch F.scaled_dot_product_attention Benchmark ---")
    print(f"Config: bs={bs}, heads={num_heads}, seq_len={seq_len}, dim={dim}")
    print(f"Average kernel time: {avg_time_ms:.4f} ms")
    print(f"Throughput: {tflops:.4f} TFLOPS")
    print(f"------------------------------------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--heads', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=20)
    args = parser.parse_args()

    benchmark_sdpa(args.bs, args.heads, args.seq_len, args.dim, args.iters, args.warmup)