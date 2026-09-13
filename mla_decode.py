"""
DeepSeek Multi-Head Latent Attention (MLA) Reference Decoding Kernel
Author: EyesTech Systems Lab (https://eyestech.in)
License: MIT
"""

import math
import torch
import torch.nn as nn
from typing import Tuple


class MultiHeadLatentAttentionDecode(nn.Module):
    """
    Production-grade Multi-Head Latent Attention (MLA) Autoregressive Decoding Kernel
    Demonstrating Query Absorption, Decoupled RoPE, and Zero-Decompression KV Cache Streaming.
    
    Reference Publication:
    EyesTech Systems Research: https://eyestech.in/deepseek-mla-architecture-kv-cache-math/
    """
    def __init__(
        self,
        d_model: int = 5120,
        n_heads: int = 128,
        d_head: int = 128,
        d_latent: int = 512,
        d_rope: int = 64
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_latent = d_latent  # d_c (512 scalars)
        self.d_rope = d_rope      # d_h^R (64 scalars)
        self.scale = 1.0 / math.sqrt(d_head + d_rope)

        # 1. KV Down-Projection: Projects hidden state to shared latent space
        self.W_DKV = nn.Linear(d_model, d_latent, bias=False)
        
        # 2. KV Up-Projection Matrices (absorbed during inference)
        self.W_UK = nn.Parameter(torch.empty(n_heads, d_head, d_latent))
        self.W_UV = nn.Parameter(torch.empty(n_heads, d_head, d_latent))
        
        # 3. Decoupled RoPE Key Projection (shared across all heads)
        self.W_KR = nn.Linear(d_model, d_rope, bias=False)

        # 4. Query Compression & Projections
        self.W_DQ = nn.Linear(d_model, 1536, bias=False)
        self.W_UQ = nn.Linear(1536, n_heads * d_head, bias=False)
        self.W_QR = nn.Linear(1536, n_heads * d_rope, bias=False)

        # 5. Output Projection
        self.W_O = nn.Linear(n_heads * d_head, d_model, bias=False)
        
        # Initialize parameters
        nn.init.normal_(self.W_UK, std=0.02)
        nn.init.normal_(self.W_UV, std=0.02)

    def apply_rope(self, x: torch.Tensor, pos: int) -> torch.Tensor:
        """Applies 1D Rotary Position Embedding to 2D coordinates."""
        half_dim = x.shape[-1] // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half_dim, device=x.device) / half_dim)
        angles = pos * freqs
        cos = torch.cos(angles).repeat(2)
        sin = torch.sin(angles).repeat(2)
        x_rot = torch.cat([-x[..., half_dim:], x[..., :half_dim]], dim=-1)
        return (x * cos) + (x_rot * sin)

    def forward_decode(
        self,
        h_t: torch.Tensor,
        current_pos: int,
        kv_cache_latent: torch.Tensor,
        kv_cache_rope: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes a single-token decode step with matrix absorption.
        
        Args:
            h_t: [Batch, 1, d_model] - Input activation of current decoding token.
            current_pos: Integer position of token in sequence.
            kv_cache_latent: [Batch, SeqLen, d_latent] - Compressed KV cache latents.
            kv_cache_rope: [Batch, SeqLen, d_rope] - Decoupled RoPE keys.
            
        Returns:
            output: [Batch, 1, d_model]
            new_kv_cache_latent: [Batch, SeqLen + 1, d_latent]
            new_kv_cache_rope: [Batch, SeqLen + 1, d_rope]
        """
        B = h_t.shape[0]

        # --- STEP 1: Compute Current Token Cache Entries ---
        c_t_kv = self.W_DKV(h_t)                                # [B, 1, 512]
        k_t_rope = self.apply_rope(self.W_KR(h_t), current_pos)   # [B, 1, 64]

        # Append to persistent KV cache (Total scalars added: 512 + 64 = 576 per token)
        kv_cache_latent = torch.cat([kv_cache_latent, c_t_kv], dim=1)
        kv_cache_rope = torch.cat([kv_cache_rope, k_t_rope], dim=1)

        # --- STEP 2: Ephemeral Query Processing ---
        c_t_q = self.W_DQ(h_t)                                   # [B, 1, 1536]
        q_content = self.W_UQ(c_t_q).view(B, self.n_heads, self.d_head) # [B, 128, 128]
        q_rope = self.W_QR(c_t_q).view(B, self.n_heads, self.d_rope)     # [B, 128, 64]
        q_rope = self.apply_rope(q_rope, current_pos)

        # --- STEP 3: MATRIX ABSORPTION (The Memory Optimization) ---
        # Instead of up-projecting kv_cache_latent (SeqLen x 512 -> SeqLen x 16384),
        # project active Query into latent space: q_absorbed = q_content @ W_UK
        # W_UK: [128, 128, 512] -> absorbed_q: [B, 128, 512]
        q_absorbed = torch.einsum('bhd,hdm->bhm', q_content, self.W_UK)

        # --- STEP 4: Direct Latent Attention Dot Product ---
        # Content score computed in 512-dim latent space
        score_content = torch.einsum('bhm,bsm->bhs', q_absorbed, kv_cache_latent)
        # Positional score computed in 64-dim RoPE space
        score_rope = torch.einsum('bhr,bsr->bhs', q_rope, kv_cache_rope)

        attention_scores = (score_content + score_rope) * self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1) # [B, 128, SeqLen]

        # --- STEP 5: Value Aggregation in Latent Space ---
        # Sum attention weights directly against 512-dim cached latents
        u_latent = torch.einsum('bhs,bsm->bhm', attention_weights, kv_cache_latent) # [B, 128, 512]

        # Final projection via fused Value-Output matrix
        v_projected = torch.einsum('bhm,hdm->bhd', u_latent, self.W_UV)
        output = self.W_O(v_projected.reshape(B, 1, self.n_heads * self.d_head))
        
        return output, kv_cache_latent, kv_cache_rope
