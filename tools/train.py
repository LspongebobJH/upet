import torch
import time
import os

# Fix memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

print(f"rank: {rank} | local_rank: {local_rank} | world_size: {world_size}")
def h200_full_vram_400w_no_oom():
    print(f"set device to {local_rank}")
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()

    # Get H200 total VRAM
    total_vram = torch.cuda.get_device_properties(local_rank).total_memory
    total_gb = total_vram / 1024**3
    safety = 10 * 1024**3  # 1.2GB buffer for CUDA context
    print(f"total gb: {total_gb} | safety: {safety}")

    # Create ONLY ONE tensor to fill 99% VRAM
    target_bytes = total_vram - safety
    fp16_bytes = 2
    size = int((target_bytes / fp16_bytes) ** 0.5)
    size = (size // 1024) * 1024

    # ALLOCATE SINGLE FULL TENSOR (fills H200 VRAM)
    a = torch.randn(size, size, dtype=torch.float16, device="cuda")
    torch.cuda.synchronize()

    used_gb = torch.cuda.memory_allocated() / 1024**3
    # print(f"✅ H200 VRAM Used: {used_gb:.2f}/{total_gb:.2f} GB (FULL)")
    # print(f"✅ NO extra tensors | NO OOM risk")
    # print(f"🚀 400W power loop started...\n")

    # INFINITE IN-PLACE COMPUTE (0 new VRAM allocated)
    # Uses Tensor Cores, max power, NO new memory
    while True:
        # In-place matmul (uses existing memory only)
        torch.matmul(a, a, out=a)
        
        # In-place operations (add + relu) — 0 VRAM cost
        a.add_(0.005)
        a.relu_()
        
        # Sync to keep power steady at 400W
        torch.cuda.synchronize()

if __name__ == "__main__":
    try:
        h200_full_vram_400w_no_oom()
    except KeyboardInterrupt:
        print("\n🛑 Stopped | VRAM released")
        torch.cuda.empty_cache()