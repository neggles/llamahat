from typing import Optional

import torch
from torch import Tensor

from llamahat.cext import (
    apply_rep_penalty,
    cleanup,
    column_remap,
    half_matmul,
    half_matmul_cublas,
    make_q4,
    q4_matmul,
    q4_matmul_lora,
    rep_penalty,
    rms_norm,
    rope_,
)

# Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension
none_tensor = torch.empty((1, 1), device="meta")


def ext_make_q4(qweight: Tensor, qzeros: Tensor, scales: Tensor, g_idx: Tensor, device: int):
    """Construct Q4Matrix, return handle"""
    return make_q4(qweight, qzeros, scales, g_idx if g_idx is not None else none_tensor, device)


def ext_q4_matmul(
    x: Tensor,
    q4: int,
    q4_width: int,
    lora_A: Optional[Tensor] = None,
    lora_B: Optional[Tensor] = None,
):
    """Matrix multiplication, returns x @ q4"""
    outshape = x.shape[:-1] + (q4_width,)
    x = x.view(-1, x.shape[-1])
    output = torch.empty((x.shape[0], q4_width), dtype=torch.float16, device=x.device)

    if lora_A is None:
        q4_matmul(x, q4, output)
    else:
        lora_temp = torch.empty((x.shape[0], lora_A.shape[1]), dtype=torch.float16, device=x.device)
        q4_matmul_lora(x, q4, output, lora_A, lora_B, lora_temp)

    return output.view(outshape)


def ext_half_matmul(x: Tensor, w: Tensor, cublas: bool = False) -> Tensor:
    """Matrix multiplication, returns x @ w, both half-precision tensors"""
    outshape = x.shape[:-1] + (w.shape[1],)
    x = x.view(-1, x.shape[-1])

    if cublas:
        output = torch.empty((x.shape[0], w.shape[1]), dtype=torch.float16, device=x.device)
        half_matmul_cublas(x, w, output)
    else:
        output = torch.zeros((x.shape[0], w.shape[1]), dtype=torch.float16, device=x.device)
        half_matmul(x, w, output)

    return output.view(outshape)


def ext_rope_(x: Tensor, sin: Tensor, cos: Tensor, past_len: int, num_heads: int, head_dim: int):
    """RoPE embeddings, in_place"""
    rope_(x, sin, cos, past_len, num_heads, head_dim)


def ext_rms_norm(x: Tensor, w: Tensor, epsilon: float):
    """RMS norm: x = x * w / sqrt(row_mean(x * x) + epsilon)"""
    outshape = x.shape
    x = x.view(-1, x.shape[-1])
    output = torch.empty_like(x)
    rms_norm(x, w, output, epsilon)
    return output.view(outshape)


def ext_rms_norm_(x: Tensor, w: Tensor, epsilon: float):
    outshape = x.shape
    x = x.view(-1, x.shape[-1])
    rms_norm(x, w, x, epsilon)


def ext_rep_penalty_mask_cpu(
    vocab_size: Tensor, sequence: Tensor, penalty_max: float, sustain: int, decay: int
):
    """Repetition penalty"""
    rep_mask = torch.empty(vocab_size, dtype=torch.float32)
    rep_penalty(sequence, rep_mask, penalty_max, sustain, decay)
    return rep_mask


__all__ = [
    "apply_rep_penalty",
    "cleanup",
    "column_remap",
    "half_matmul",
    "half_matmul_cublas",
    "make_q4",
    "q4_matmul",
    "q4_matmul_lora",
    "rep_penalty",
    "rms_norm",
    "rope_",
    "none_tensor",
    "ext_make_q4",
    "ext_q4_matmul",
    "ext_half_matmul",
    "ext_rope_",
    "ext_rms_norm",
    "ext_rms_norm_",
    "ext_rep_penalty_mask_cpu",
]
