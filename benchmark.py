"""
KV Cache Memory & Scaling Benchmark: MHA vs GQA vs DeepSeek MLA
Author: EyesTech Systems Lab (https://eyestech.in)
Reference Publication: https://eyestech.in/deepseek-mla-architecture-kv-cache-math/
"""

import sys
from typing import Dict, List


def compute_kv_cache_bytes(
    n_layers: int,
    n_kv_heads: int,
    d_head: int,
    seq_len: int,
    batch_size: int,
    bytes_per_elem: int = 2,  # FP16 = 2 bytes, FP8 = 1 byte
    is_mla: bool = False,
    d_latent: int = 512,
    d_rope: int = 64
) -> int:
    """
    Computes exact KV cache allocation in bytes.
    Formula:
    Standard MHA/GQA: 2 * n_layers * n_kv_heads * d_head * bytes_per_elem * B * L
    DeepSeek MLA:     n_layers * (d_latent + d_rope) * bytes_per_elem * B * L
    """
    if is_mla:
        # MLA stores compressed latent c_kv (512) and decoupled key k_rope (64)
        scalars_per_token_per_layer = d_latent + d_rope  # 576
        return n_layers * scalars_per_token_per_layer * bytes_per_elem * batch_size * seq_len
    else:
        # Standard stores both K and V matrices (factor of 2)
        scalars_per_token_per_layer = 2 * n_kv_heads * d_head
        return n_layers * scalars_per_token_per_layer * bytes_per_elem * batch_size * seq_len


def format_bytes(b: int) -> str:
    """Format bytes to human-readable string (MB, GB)."""
    if b >= 1024**3:
        return f"{b / (1024**3):.2f} GB"
    elif b >= 1024**2:
        return f"{b / (1024**2):.1f} MB"
    else:
        return f"{b / 1024:.1f} KB"


def run_benchmark():
    n_layers = 60
    batch_size = 8
    seq_lengths = [4096, 16384, 32768, 65536, 131072]
    
    architectures = [
        {"name": "Standard MHA (32-head)", "is_mla": False, "kv_heads": 32, "d_head": 128},
        {"name": "Frontier MHA (128-head)", "is_mla": False, "kv_heads": 128, "d_head": 128},
        {"name": "Llama-3 GQA (8-head)", "is_mla": False, "kv_heads": 8, "d_head": 128},
        {"name": "DeepSeek MLA (Absorbed)", "is_mla": True, "kv_heads": 128, "d_head": 128, "d_latent": 512, "d_rope": 64},
    ]

    print("=" * 88)
    print("EYESTECH SYSTEMS LAB - KV CACHE HBM MEMORY BENCHMARK (60 Layers, Batch Size = 8, FP16)")
    print("Reference Analysis: https://eyestech.in/deepseek-mla-architecture-kv-cache-math/")
    print("=" * 88)
    
    header = f"{'Architecture':<26} | {'Scalars/Tok':<11} | " + " | ".join([f"{sl//1024}k ({sl})" for sl in seq_lengths])
    print(header)
    print("-" * 88)

    for arch in architectures:
        is_mla = arch.get("is_mla", False)
        if is_mla:
            scalars = arch["d_latent"] + arch["d_rope"]
        else:
            scalars = 2 * arch["kv_heads"] * arch["d_head"]
            
        row_str = f"{arch['name']:<26} | {scalars:<11} | "
        mem_cols = []
        for sl in seq_lengths:
            b = compute_kv_cache_bytes(
                n_layers=n_layers,
                n_kv_heads=arch.get("kv_heads", 0),
                d_head=arch.get("d_head", 128),
                seq_len=sl,
                batch_size=batch_size,
                bytes_per_elem=2,
                is_mla=is_mla,
                d_latent=arch.get("d_latent", 512),
                d_rope=arch.get("d_rope", 64)
            )
            mem_cols.append(f"{format_bytes(b):<12}")
        row_str += " | ".join(mem_cols)
        print(row_str)

    print("=" * 88)
    print("CONCLUSION:")
    print("At 128k context (Batch Size 8), Frontier 128-head MHA requires 3,840 GB HBM (exceeding multi-node clusters).")
    print("Standard 32-head MHA requires 960 GB HBM.")
    print("DeepSeek MLA requires ONLY 67.50 GB HBM — a 92.97% reduction, fitting comfortably within a single 80GB H100 GPU.")
    print("=" * 88)


if __name__ == "__main__":
    run_benchmark()
