#!/usr/bin/env python3
"""Check GPU usage and identify which GPUs are free."""
import torch
import subprocess
import sys

def check_gpu_usage():
    """Check GPU memory usage and identify free GPUs."""
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return
    
    print("=" * 60)
    print("GPU Status Check")
    print("=" * 60)
    
    num_gpus = torch.cuda.device_count()
    print(f"\nFound {num_gpus} GPU(s):\n")
    
    free_gpus = []
    used_gpus = []
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = props.total_memory / 1024**3
        
        usage_pct = (reserved / total) * 100
        
        print(f"GPU {i}: {props.name}")
        print(f"  Total Memory: {total:.2f} GB")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Usage: {usage_pct:.1f}%")
        
        # Consider GPU free if less than 100MB is reserved
        if reserved < 0.1:
            free_gpus.append(i)
            print(f"  Status: ✅ FREE")
        else:
            used_gpus.append(i)
            print(f"  Status: ⚠️  IN USE")
        print()
    
    print("=" * 60)
    if free_gpus:
        print(f"Free GPUs: {', '.join(map(str, free_gpus))}")
    else:
        print("No free GPUs found")
    
    if used_gpus:
        print(f"GPUs in use: {', '.join(map(str, used_gpus))}")
    print("=" * 60)
    
    # Try to check nvidia-smi for process info
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', 
                                '--format=csv,noheader'], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("\nDetailed GPU Info (from nvidia-smi):")
            print(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        print("\n(Note: nvidia-smi not available or timed out)")

if __name__ == "__main__":
    check_gpu_usage()

