# DeepSeek Multi-Head Latent Attention (MLA) & KV Cache Benchmark

[![EyesTech Systems Research](https://img.shields.io/badge/EyesTech-Systems_Research-002050?style=flat-square&logo=gitbook)](https://eyestech.in/deepseek-mla-architecture-kv-cache-math/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg?style=flat-square)](LICENSE)

A clean, standalone, production-grade PyTorch reference implementation of **DeepSeek Multi-Head Latent Attention (MLA)** featuring **Query Absorption**, **Decoupled RoPE**, and bare-metal **KV Cache Memory Benchmarking** up to 128k context windows.

> 📖 **Canonical Systems Audit**:  
> For the complete mathematical proof, memory bandwidth derivations, and production serving economics (vLLM / SGLang), read the flagship investigation:  
> 👉 **[DeepSeek MLA Architecture: How It Cuts KV Cache by 93%](https://eyestech.in/deepseek-mla-architecture-kv-cache-math/)** at **[EyesTech Systems Lab](https://eyestech.in)**.

---

## ⚡ Key Architectural Takeaways

In autoregressive large language model serving, the primary hardware bottleneck at 32k–128k tokens is **High Bandwidth Memory (HBM) exhaustion**, not raw FLOPs.

| Attention Mechanism | Stored Scalars / Token / Layer | KV Cache @ 128k Context (BS=8, FP16) | HBM Memory Wall Feasibility |
| :--- | :---: | :---: | :--- |
| **Frontier 128-Head MHA** | 32,768 | **3,840 GB** | ❌ Exceeds multi-node clusters |
| **Standard 32-Head MHA** | 8,192 | **960 GB** | ❌ Requires 12x 80GB H100 GPUs |
| **Llama-3 8-Head GQA** | 2,048 | **240 GB** | ⚠️ Requires 3x 80GB H100 GPUs |
| **DeepSeek MLA (Absorbed)** | **576** | **67.5 GB** |  **Fits on a single 80GB H100 GPU (-93%)** |

### How MLA Cuts KV Cache by 92.97%:
1. **Low-Rank Latent Compression ($c_t^{KV}$)**: Compresses Key and Value tensors into a compact 512-dimensional shared latent vector.
2. **Decoupled Rotary Position Embedding ($k_t^R$)**: Keeps a 64-dimensional uncompressed RoPE key to maintain exact positional semantics without inflating the latent vector ($512 + 64 = 576$ scalars per token).
3. **Inference Query Absorption ($q_{\text{absorbed}} = q \cdot W_{UK}$)**: Exploits matrix associativity during autoregressive decoding. The up-projection matrices are folded directly into the active Query tensors, allowing attention to be computed directly against compressed latents without runtime decompression.

---

## 🚀 Quickstart & Reproduction

### 1. Installation
Clone the repository and install PyTorch:
```bash
git clone https://github.com/eyestech-labs/deepseek-mla-kvcache.git
cd deepseek-mla-kvcache
pip install -r requirements.txt
```

### 2. Run the KV Cache Scaling Benchmark
Measure exact HBM memory footprint across sequence lengths from 4k to 128k:
```bash
python benchmark.py
```

### 3. Run the Reference MLA PyTorch Kernel
Execute a standalone autoregressive decode step demonstrating query absorption:
```python
import torch
from mla_decode import MultiHeadLatentAttentionDecode

device = "cuda" if torch.cuda.is_available() else "cpu"
mla = MultiHeadLatentAttentionDecode(
    d_model=5120,
    n_heads=128,
    d_head=128,
    d_latent=512,
    d_rope=64
).to(device)

# Simulate 128k context stream at position 131,071
seq_len = 1024
h_t = torch.randn(2, 1, 5120, device=device)
cache_latent = torch.randn(2, seq_len, 512, device=device)
cache_rope = torch.randn(2, seq_len, 64, device=device)

output, updated_latent, updated_rope = mla.forward_decode(
    h_t, current_pos=seq_len, kv_cache_latent=cache_latent, kv_cache_rope=cache_rope
)

print(f"Token Output Shape: {output.shape}")
print(f"Stored KV Cache per Token: {updated_latent.shape[-1] + updated_rope.shape[-1]} scalars")
```

---

## 📊 Benchmark Telemetry (60 Layers, Batch Size = 8, FP16)

```text
========================================================================================
Architecture               | Scalars/Tok | 4k (4096)  | 16k (16384) | 32k (32768) | 64k (65536) | 128k (131072)
----------------------------------------------------------------------------------------
Standard MHA (32-head)     | 8192        | 30.00 GB   | 120.00 GB   | 240.00 GB   | 480.00 GB   | 960.00 GB   
Frontier MHA (128-head)    | 32768       | 120.00 GB  | 480.00 GB   | 960.00 GB   | 1920.00 GB  | 3840.00 GB  
Llama-3 GQA (8-head)       | 2048        | 7.50 GB    | 30.00 GB    | 60.00 GB    | 120.00 GB   | 240.00 GB   
DeepSeek MLA (Absorbed)    | 576         | 2.11 GB    | 8.44 GB     | 16.88 GB    | 33.75 GB    | 67.50 GB    
========================================================================================
```

---

## 📚 Citation & Attribution

If you use this benchmark harness, reference implementation, or mathematical formalization in academic papers or technical audits, please cite:

### BibTeX
```bibtex
@misc{fischer2026deepseekmla,
  author = {Fischer, Klaus and Ranganathan, Devika},
  title = {DeepSeek MLA Architecture: How It Cuts KV Cache by 93%},
  howpublished = {\url{https://eyestech.in/deepseek-mla-architecture-kv-cache-math/}},
  journal = {EyesTech Systems Research},
  year = {2026},
  note = {EyesTech Systems Lab Hardware Audit Series}
}
```

Or reference `CITATION.cff` via GitHub's native citation tool.

---

## ⚖️ License
Released under the [MIT License](LICENSE). Maintained by [EyesTech Systems Lab](https://eyestech.in).
