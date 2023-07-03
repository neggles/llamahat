import torch

def apply_rep_penalty(
    sequence: torch.Tensor,
    penalty_max: float,
    sustain: int,
    decay: int,
    logits: torch.Tensor,
) -> None: ...
def cleanup() -> None: ...
def column_remap(
    x: torch.Tensor,
    x_new: torch.Tensor,
    x_map: torch.Tensor,
) -> None: ...
def half_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
) -> None: ...
def half_matmul_cublas(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
) -> None: ...
def make_q4(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    device: int,
) -> int: ...
def prepare_buffers(
    device: torch.device,
    temp_state: torch.Tensor,
    temp_mlp: torch.Tensor,
    temp_zeros_float: torch.Tensor,
    temp_dq: torch.Tensor,
) -> None: ...
def q4_attn(
    x: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    epsilon: float,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    q_proj: int,
    k_proj: int,
    v_proj: int,
    sin: torch.Tensor,
    cos: torch.Tensor,
    q_len: int,
    past_len: int,
    num_heads: int,
    head_dim: int,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    max_seq_len: int,
    q_a: torch.Tensor,
    q_b: torch.Tensor,
    k_a: torch.Tensor,
    k_b: torch.Tensor,
    v_a: torch.Tensor,
    v_b: torch.Tensor,
    lora_temp: torch.Tensor,
) -> None: ...
def q4_attn_2(
    x: torch.Tensor,
    attn_output: torch.Tensor,
    o_proj: int,
    o_a: torch.Tensor,
    o_b: torch.Tensor,
    lora_temp: torch.Tensor,
) -> None: ...
def q4_matmul(
    x: torch.Tensor,
    w: int,
    out: torch.Tensor,
) -> None: ...
def q4_matmul_lora(
    x: torch.Tensor,
    w: int,
    out: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    lora_temp: torch.Tensor,
) -> None: ...
def q4_mlp(
    x: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    epsilon: float,
    gate: int,
    up: int,
    down: int,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    up_a: torch.Tensor,
    up_b: torch.Tensor,
    down_a: torch.Tensor,
    down_b: torch.Tensor,
    lora_temp: torch.Tensor,
) -> None: ...
def rep_penalty(
    sequence: torch.Tensor,
    rep_mask: torch.Tensor,
    penalty_max: float,
    sustain: int,
    decay: int,
) -> None: ...
def rms_norm(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    epsilon: float,
) -> None: ...
def rope_(
    x: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    past_len: int,
    num_heads: int,
    head_dim: int,
) -> None: ...
def set_tuning_params(
    matmul_recons_thd: int,
    fused_mlp_thd: int,
    sdp_thd: int,
    matmul_fused_remap: bool,
    rmsnorm_no_half2: bool,
    rope_no_half2: bool,
    matmul_no_half2: bool,
    silu_no_half2: bool,
    concurrent_streams: bool,
) -> None: ...
