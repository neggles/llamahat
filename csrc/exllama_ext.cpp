#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstdio>

#include "cuda_buffers.cuh"
#include "func/column_remap.cuh"
#include "func/half_matmul.cuh"
#include "func/q4_attn.cuh"
#include "func/q4_matmul.cuh"
#include "func/q4_matrix.cuh"
#include "func/q4_mlp.cuh"
#include "func/rep_penalty.h"
#include "func/rms_norm.cuh"
#include "func/rope.cuh"
#include "tuning.h"
#include "util.cuh"

// Check CUDA return code. We don't want to include Torch headers in the .cu files because parsing
// them adds almost a minute to the compile time on a 12900K. Also passing exceptions back to Python
// is super tricky, so in place of exceptions, CUDA functions return with a cudaError_t which we can
// parse and dump to the console.

void check_cuda(cudaError_t ret) {
    switch (ret) {
        case cudaSuccess:
            break;

        case cudaUnspecified:
            printf(" **** Unspecified error\n");
            TORCH_CHECK(false, "CUDA error");
            break;

        default:
            printf(" **** CUDA error\n");
            printf(" **** %s\n", cudaGetErrorString(ret));
            TORCH_CHECK(false, "CUDA error");
            break;
    }
}

// Some decluttering macros

#define STRINGIFY_(__x) #__x
#define STRINGIFY(__x)  STRINGIFY_(__x)
#define TORCH_CHECK_DTYPE(__x, __dtype)                                                   \
    TORCH_CHECK_TYPE(                                                                     \
        (__x).dtype() == torch::__dtype, #__x " is incorrect datatype, must be " #__dtype \
    )
#define TORCH_CHECK_DTYPE_OPT(__x, __dtype)                          \
    TORCH_CHECK_TYPE(                                                \
        (__x).device().is_meta() || (__x).dtype() == torch::__dtype, \
        #__x " is incorrect datatype, must be " #__dtype             \
    )
#define TORCH_CHECK_SHAPES(__x, __dim_x, __y, __dim_y, __scale_y) \
    TORCH_CHECK_VALUE(                                            \
        (__x).size(__dim_x) == (__y).size(__dim_y) * __scale_y,   \
        #__x " and " #__y " have incompatible shapes"             \
    )
#define TORCH_CHECK_SHAPES_OPT(__x, __dim_x, __y, __dim_y, __scale_y)                       \
    TORCH_CHECK_VALUE(                                                                      \
        (__x).device().is_meta() || (__x).size(__dim_x) == (__y).size(__dim_y) * __scale_y, \
        #__x " and " #__y " have incompatible shapes"                                       \
    )
#define TORCH_CHECK_SHAPE_MOD(__x, __dim_x, __mod)                                    \
    TORCH_CHECK_VALUE(                                                                \
        (__x).size(__dim_x) % __mod == 0,                                             \
        #__x ".shape[" STRINGIFY(__dim_x) "] must be a multiple of " STRINGIFY(__mod) \
    )

#define TORCH_CHECK_DEVICE_INDEX(__index)                                      \
    do {                                                                       \
        TORCH_CHECK_INDEX(__index >= 0, "no device index");                    \
        TORCH_CHECK_INDEX(__index < CUDA_MAX_DEVICES, "invalid device index"); \
    } while (0)

#define TORCH_CHECK_QUANT(__w, __w_scales, __w_zeros, __seq_g_idx, __x_map) \
    do {                                                                    \
        TORCH_CHECK_DTYPE(__w, kInt);                                       \
        TORCH_CHECK_DTYPE(__w_scales, kHalf);                               \
        TORCH_CHECK_DTYPE(__w_zeros, kInt);                                 \
        TORCH_CHECK_DTYPE_OPT(__seq_g_idx, kShort);                         \
        TORCH_CHECK_DTYPE_OPT(__x_map, kInt);                               \
        TORCH_CHECK_SHAPES_OPT(__seq_g_idx, 0, __w, 0, 2 * 8);              \
        TORCH_CHECK_SHAPES_OPT(__x_map, 0, __w, 0, 8);                      \
    } while (0)

int get_groupsize(torch::Tensor w, torch::Tensor w_zeros) {
    int groupsize = w.size(0) * 8 / w_zeros.size(0);
    TORCH_CHECK(
        groupsize * w_zeros.size(0) == w.size(0) * 8,
        "w.shape[-2] must be a multiple of zeros.shape[-2]"
    )
    return groupsize;
}

// Tuning parameters

ExLlamaTuning tuningParams;

void set_tuning_params(
    int  matmul_recons_thd,
    int  fused_mlp_thd,
    int  sdp_thd,
    bool matmul_fused_remap,
    bool rmsnorm_no_half2,
    bool rope_no_half2,
    bool matmul_no_half2,
    bool silu_no_half2,
    bool concurrent_streams
) {
    tuningParams.matmul_recons_thd  = matmul_recons_thd;
    tuningParams.fused_mlp_thd      = fused_mlp_thd;
    tuningParams.sdp_thd            = sdp_thd;
    tuningParams.matmul_fused_remap = matmul_fused_remap;

    tuningParams.rmsnorm_no_half2   = rmsnorm_no_half2;
    tuningParams.rope_no_half2      = rope_no_half2;
    tuningParams.matmul_no_half2    = matmul_no_half2;
    tuningParams.silu_no_half2      = silu_no_half2;
    tuningParams.concurrent_streams = concurrent_streams;
}

// Release all unmanaged objects allocated by the extension

void cleanup() {
    cleanup_buffers_cuda();
    g_q4_free_matrices();
}

// Prepare buffers for forward pass

void prepare_buffers(
    torch::Device device,
    torch::Tensor temp_state,
    torch::Tensor temp_mlp,
    torch::Tensor temp_zeros_float,
    torch::Tensor temp_dq
) {
    int device_index = device.index();
    TORCH_CHECK_DEVICE_INDEX(device_index);
    const at::cuda::OptionalCUDAGuard device_guard(device);

    int max_zeros_float = temp_zeros_float.size(-1);

    prepare_buffers_cuda(
        device_index,
        (half*)temp_state.data_ptr(),
        (half*)temp_mlp.data_ptr(),
        (float*)temp_zeros_float.data_ptr(),
        (half*)temp_dq.data_ptr(),
        max_zeros_float
    );
}

// Create Q4Matrix, return handle

uintptr_t make_q4(
    torch::Tensor qweight,
    torch::Tensor qzeros,
    torch::Tensor scales,
    torch::Tensor g_idx,
    int           device
) {
    TORCH_CHECK_DTYPE(qweight, kInt);
    TORCH_CHECK_DTYPE(qzeros, kInt);
    TORCH_CHECK_DTYPE(scales, kHalf);
    TORCH_CHECK_DTYPE_OPT(g_idx, kInt);
    TORCH_CHECK_SHAPES(qweight, 1, qzeros, 1, 8);
    TORCH_CHECK_SHAPES(scales, 1, qweight, 1, 1);
    TORCH_CHECK_SHAPES(qzeros, 0, scales, 0, 1);

    int width  = qweight.size(1);
    int height = qweight.size(0) * 8;
    int groups = qzeros.size(0);

    Q4Matrix* m = new Q4Matrix(
        height,
        width,
        groups,

        (uint32_t*)qweight.data_ptr(),
        (uint32_t*)qzeros.data_ptr(),
        (half*)scales.data_ptr(),
        g_idx.device().is_meta() ? NULL : (uint32_t*)g_idx.data_ptr(),

        device
    );

    g_q4_keep_matrix(m);
    return reinterpret_cast<uintptr_t>(m);
}

// Matmul half @ quant -> half

void q4_matmul(torch::Tensor x, uintptr_t w, torch::Tensor out) {
    Q4Matrix* wm = reinterpret_cast<Q4Matrix*>(w);

    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(out, kHalf);
    TORCH_CHECK_SHAPES(x, 0, out, 0, 1);
    TORCH_CHECK(wm->height == x.size(-1), "x and w have incompatible shapes")

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    int x_height = x.size(0);

    if (tuningParams.matmul_recons_thd == 0 || x_height < tuningParams.matmul_recons_thd) {
        q4_matmul_cuda(&tuningParams, (half*)x.data_ptr(), x_height, wm, (half*)out.data_ptr());
    } else {
        q4_matmul_recons_cuda(
            &tuningParams,
            (half*)x.data_ptr(),
            x_height,
            wm,
            (half*)out.data_ptr(),
            at::cuda::getCurrentCUDABlasHandle()
        );
    }
}

// Matmul half @ quant + half @ half @ half -> half
// Same as q4_matmul, but adds (x @ lora_A) @ lora_B to the result

void q4_matmul_lora(
    torch::Tensor x,
    uintptr_t     w,
    torch::Tensor out,
    torch::Tensor lora_A,
    torch::Tensor lora_B,
    torch::Tensor lora_temp // empty tensor, shape of (x @ lora_A)
) {
    Q4Matrix* wm = reinterpret_cast<Q4Matrix*>(w);
    TORCH_CHECK(wm->height == x.size(-1), "x and w have incompatible shapes")

    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(out, kHalf);
    TORCH_CHECK_SHAPES(x, 0, out, 0, 1);
    TORCH_CHECK_SHAPES(x, 0, lora_temp, 0, 1);
    TORCH_CHECK_SHAPES(x, 1, lora_A, 0, 1);
    TORCH_CHECK_SHAPES(lora_A, 1, lora_B, 0, 1);
    TORCH_CHECK_SHAPES(lora_B, 1, out, 1, 1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    cublasHandle_t                    handle = at::cuda::getCurrentCUDABlasHandle();

    // lora_temp = x @ lora_A

    half_matmul_cublas_cuda(
        &tuningParams,
        (half*)x.data_ptr(),
        (half*)lora_A.data_ptr(),
        (half*)lora_temp.data_ptr(),
        x.size(0),
        x.size(1),
        lora_A.size(1),
        handle
    );

    // out = lora_temp @ lora_B

    half_matmul_cublas_cuda(
        &tuningParams,
        (half*)lora_temp.data_ptr(),
        (half*)lora_B.data_ptr(),
        (half*)out.data_ptr(),
        lora_temp.size(0),
        lora_temp.size(1),
        lora_B.size(1),
        handle
    );

    int x_height = x.size(0);

    if (tuningParams.matmul_recons_thd == 0 || x_height < tuningParams.matmul_recons_thd) {
        q4_matmul_cuda(
            &tuningParams, (half*)x.data_ptr(), x_height, wm, (half*)out.data_ptr(), true
        );
    } else {
        q4_matmul_recons_cuda(
            &tuningParams, (half*)x.data_ptr(), x_height, wm, (half*)out.data_ptr(), handle, true
        );
    }
}

// Remap columns in half tensor

void column_remap(torch::Tensor x, torch::Tensor x_new, torch::Tensor x_map) {
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(x_new, kHalf);
    TORCH_CHECK_DTYPE(x_map, kInt);
    TORCH_CHECK_SHAPES(x_map, 0, x, 1, 1);

    int height = x.size(0);
    int width  = x.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    column_remap_cuda(
        (half*)x.data_ptr(), (half*)x_new.data_ptr(), height, width, (uint32_t*)x_map.data_ptr()
    );
}

// Matmul half @ half -> half, custom kernel

void half_matmul(torch::Tensor x, torch::Tensor w, torch::Tensor out) {
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_DTYPE(out, kHalf);
    TORCH_CHECK_SHAPES(x, 1, w, 0, 1);

    int height = x.size(0);
    int dim    = x.size(1);
    int width  = w.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    half_matmul_cuda(
        (half*)x.data_ptr(), (half*)w.data_ptr(), (half*)out.data_ptr(), height, dim, width
    );
}

// Matmul half @ half -> half using cuBLAS

void half_matmul_cublas(torch::Tensor x, torch::Tensor w, torch::Tensor out) {
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_DTYPE(out, kHalf);
    TORCH_CHECK_SHAPES(x, 1, w, 0, 1);

    int height = x.size(0);
    int dim    = x.size(1);
    int width  = w.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    cublasHandle_t                    handle = at::cuda::getCurrentCUDABlasHandle();

    half_matmul_cublas_cuda(
        &tuningParams,
        (half*)x.data_ptr(),
        (half*)w.data_ptr(),
        (half*)out.data_ptr(),
        height,
        dim,
        width,
        handle
    );
}

// Llama self attention (WIP)

void q4_attn(
    torch::Tensor x,               // shape == (bsz, q_len, dim)
    torch::Tensor rms_norm_weight, // shape == (x.shape[1],) == (dim,)
    float         epsilon,
    torch::Tensor query_states, // shape == (bsz, q_len, dim)
    torch::Tensor key_states,   // shape == (bsz, q_len, dim)
    torch::Tensor value_states, // shape == (bsz, q_len, dim)
    uintptr_t     q_proj,
    uintptr_t     k_proj,
    uintptr_t     v_proj,
    torch::Tensor sin,
    torch::Tensor cos,
    int           q_len,
    int           past_len,
    int           num_heads,
    int           head_dim,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    int           max_seq_len,
    torch::Tensor q_a,
    torch::Tensor q_b,
    torch::Tensor k_a,
    torch::Tensor k_b,
    torch::Tensor v_a,
    torch::Tensor v_b,
    torch::Tensor lora_temp
) {
    TORCH_CHECK_DTYPE(query_states, kHalf);
    TORCH_CHECK_DTYPE(key_states, kHalf);

    int bsz = query_states.size(0);
    int dim = query_states.size(2);

    torch::Device device       = x.device();
    int           device_index = device.index();
    TORCH_CHECK_DEVICE_INDEX(device_index);
    const at::cuda::OptionalCUDAGuard device_guard(device);

    cudaStream_t current_stream = at::cuda::getCurrentCUDAStream().stream();

    int q_rank = q_a.device().is_meta() ? 0 : q_a.size(1);
    int k_rank = k_a.device().is_meta() ? 0 : k_a.size(1);
    int v_rank = v_a.device().is_meta() ? 0 : v_a.size(1);

    q4_attn_cuda(
        &tuningParams,
        current_stream,
        at::cuda::getCurrentCUDABlasHandle(),
        (half*)x.data_ptr(),
        (half*)rms_norm_weight.data_ptr(),
        epsilon,
        (half*)query_states.data_ptr(),
        (half*)key_states.data_ptr(),
        (half*)value_states.data_ptr(),
        reinterpret_cast<Q4Matrix*>(q_proj),
        reinterpret_cast<Q4Matrix*>(k_proj),
        reinterpret_cast<Q4Matrix*>(v_proj),
        (half*)sin.data_ptr(),
        (half*)cos.data_ptr(),
        bsz,
        q_len,
        dim,
        head_dim,
        num_heads,
        past_len,
        (half*)key_cache.data_ptr(),
        (half*)value_cache.data_ptr(),
        q_rank ? (half*)q_a.data_ptr() : NULL,
        q_rank ? (half*)q_b.data_ptr() : NULL,
        q_rank,
        k_rank ? (half*)k_a.data_ptr() : NULL,
        k_rank ? (half*)k_b.data_ptr() : NULL,
        k_rank,
        v_rank ? (half*)v_a.data_ptr() : NULL,
        v_rank ? (half*)v_b.data_ptr() : NULL,
        v_rank,
        lora_temp.device().is_meta() ? NULL : (half*)lora_temp.data_ptr(),
        max_seq_len,
        device_index
    );
}

void q4_attn_2(
    torch::Tensor x,
    torch::Tensor attn_output,
    uintptr_t     o_proj,
    torch::Tensor o_a,
    torch::Tensor o_b,
    torch::Tensor lora_temp
) {
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(attn_output, kHalf);

    int height = x.size(0);

    int o_rank = o_a.device().is_meta() ? 0 : o_a.size(1);

    q4_attn_2_cuda(
        &tuningParams,
        at::cuda::getCurrentCUDABlasHandle(),
        (half*)x.data_ptr(),
        (half*)attn_output.data_ptr(),
        reinterpret_cast<Q4Matrix*>(o_proj),
        height,
        o_rank ? (half*)o_a.data_ptr() : NULL,
        o_rank ? (half*)o_b.data_ptr() : NULL,
        o_rank,
        lora_temp.device().is_meta() ? NULL : (half*)lora_temp.data_ptr()
    );
}

// Llama MLP

void q4_mlp(
    torch::Tensor x,               // shape == (height, dim)
    torch::Tensor rms_norm_weight, // shape == (x.shape[1],) == (dim,)
    float         epsilon,
    uintptr_t     gate,
    uintptr_t     up,
    uintptr_t     down,
    torch::Tensor gate_a,
    torch::Tensor gate_b,
    torch::Tensor up_a,
    torch::Tensor up_b,
    torch::Tensor down_a,
    torch::Tensor down_b,
    torch::Tensor lora_temp
) {
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(rms_norm_weight, kHalf);

    int height = x.size(0);
    int dim    = x.size(1);

    torch::Device device       = x.device();
    int           device_index = device.index();
    TORCH_CHECK_DEVICE_INDEX(device_index);
    const at::cuda::OptionalCUDAGuard device_guard(device);

    int gate_rank = gate_a.device().is_meta() ? 0 : gate_a.size(1);
    int up_rank   = gate_a.device().is_meta() ? 0 : up_a.size(1);
    int down_rank = gate_a.device().is_meta() ? 0 : down_a.size(1);

    q4_mlp_cuda(
        &tuningParams,
        (half*)x.data_ptr(),
        (half*)rms_norm_weight.data_ptr(),
        epsilon,
        reinterpret_cast<Q4Matrix*>(gate),
        reinterpret_cast<Q4Matrix*>(up),
        reinterpret_cast<Q4Matrix*>(down),
        height,
        dim,
        gate_rank ? (half*)gate_a.data_ptr() : NULL,
        gate_rank ? (half*)gate_b.data_ptr() : NULL,
        gate_rank,
        up_rank ? (half*)up_a.data_ptr() : NULL,
        up_rank ? (half*)up_b.data_ptr() : NULL,
        up_rank,
        down_rank ? (half*)down_a.data_ptr() : NULL,
        down_rank ? (half*)down_b.data_ptr() : NULL,
        down_rank,
        lora_temp.device().is_meta() ? NULL : (half*)lora_temp.data_ptr(),
        at::cuda::getCurrentCUDABlasHandle(),
        device_index
    );
}

// RMS layernorm

void rms_norm(torch::Tensor x, torch::Tensor w, torch::Tensor out, float epsilon) {
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_DTYPE(out, kHalf);
    TORCH_CHECK_SHAPES(x, 1, w, 0, 1);
    TORCH_CHECK_SHAPES(x, 1, w, 0, 1);
    TORCH_CHECK_SHAPES(x, 0, out, 0, 1);
    TORCH_CHECK_SHAPES(x, 1, out, 1, 1);

    int rows = x.size(0);
    int dim  = x.size(1);

    torch::Device device       = x.device();
    int           device_index = device.index();
    TORCH_CHECK_DEVICE_INDEX(device_index);
    const at::cuda::OptionalCUDAGuard device_guard(device);

    rms_norm_cuda(
        &tuningParams,
        (half*)x.data_ptr(),
        (half*)w.data_ptr(),
        (half*)out.data_ptr(),
        epsilon,
        rows,
        dim,
        device_index
    );
}

// RoPE rotary positional embeddings

void rope_(
    torch::Tensor x,
    torch::Tensor sin,
    torch::Tensor cos,
    int           past_len,
    int           num_heads,
    int           head_dim
) {
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(sin, kHalf);
    TORCH_CHECK_DTYPE(cos, kHalf);
    TORCH_CHECK(head_dim == cos.size(-1), "cos table does not match head_dim");
    TORCH_CHECK(head_dim == sin.size(-1), "sin table does not match head_dim");

    int bsz            = x.size(0);
    int rows_per_batch = x.numel() / head_dim / bsz;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    rope_cuda(
        &tuningParams,
        (half*)x.data_ptr(),
        (half*)sin.data_ptr(),
        (half*)cos.data_ptr(),
        bsz,
        rows_per_batch,
        head_dim,
        num_heads,
        past_len
    );
}

// Repetition penalty (CPU)

void rep_penalty(
    torch::Tensor sequence,
    torch::Tensor rep_mask,
    float         penalty_max,
    int           sustain,
    int           decay
) {
    TORCH_CHECK_DTYPE(sequence, kLong);
    TORCH_CHECK_DTYPE(rep_mask, kFloat);

    int vocab_size = rep_mask.size(0);
    int seq_len    = sequence.size(-1);

    // TODO: Support batch size

    rep_penalty_cpu(
        vocab_size,
        (uint64_t*)sequence.data_ptr(),
        (float*)rep_mask.data_ptr(),
        penalty_max,
        sustain,
        decay,
        seq_len
    );
}

void apply_rep_penalty(
    torch::Tensor sequence,
    float         penalty_max,
    int           sustain,
    int           decay,
    torch::Tensor logits
) {
    TORCH_CHECK_DTYPE(sequence, kLong);
    TORCH_CHECK_DTYPE(logits, kFloat);
    TORCH_CHECK_SHAPES(sequence, 0, logits, 0, 1);

    int vocab_size = logits.size(-1);
    int bsz        = sequence.size(0);
    int seq_len    = sequence.size(-1);

    for (int i = 0; i < bsz; i++) {
        apply_rep_penalty_cpu(
            vocab_size,
            ((uint64_t*)sequence.data_ptr()) + i * seq_len,
            penalty_max,
            sustain,
            decay,
            seq_len,
            ((float*)logits.data_ptr()) + i * vocab_size
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using namespace pybind11::literals;
    m.def(
        "set_tuning_params",
        &set_tuning_params,
        "Set tuning parameters in the C++ extension",
        py::arg("matmul_recons_thd"),
        py::arg("fused_mlp_thd"),
        py::arg("sdp_thd"),
        py::arg("matmul_fused_remap"),
        py::arg("rmsnorm_no_half2"),
        py::arg("rope_no_half2"),
        py::arg("matmul_no_half2"),
        py::arg("silu_no_half2"),
        py::arg("concurrent_streams")
    );
    m.def(
        "prepare_buffers",
        &prepare_buffers,
        "Prepare buffers for forward pass",
        py::arg("device"),
        py::arg("temp_state"),
        py::arg("temp_mlp"),
        py::arg("temp_zeros_float"),
        py::arg("temp_dq")
    );
    m.def("cleanup", &cleanup, "Release all unmanaged objects allocated by the extension");
    m.def(
        "make_q4",
        &make_q4,
        "Create Q4Matrix, return handle",
        py::arg("qweight"),
        py::arg("qzeros"),
        py::arg("scales"),
        py::arg("g_idx"),
        py::arg("device")
    );
    m.def(
        "q4_matmul",
        &q4_matmul,
        "Matmul half @ quant -> half",
        py::arg("x"),
        py::arg("w"),
        py::arg("out")
    );
    m.def(
        "q4_matmul_lora",
        &q4_matmul_lora,
        "Matmul half @ quant + half @ half @ half -> half. Same as q4_matmul, "
        "but adds (x @ lora_A) @ lora_B to the result",
        py::arg("x"),
        py::arg("w"),
        py::arg("out"),
        py::arg("lora_A"),
        py::arg("lora_B"),
        py::arg("lora_temp")
    );
    m.def(
        "q4_attn",
        &q4_attn,
        "LLaMA self attention (WIP)",
        py::arg("x"),
        py::arg("rms_norm_weight"),
        py::arg("epsilon"),
        py::arg("query_states"),
        py::arg("key_states"),
        py::arg("value_states"),
        py::arg("q_proj"),
        py::arg("k_proj"),
        py::arg("v_proj"),
        py::arg("sin"),
        py::arg("cos"),
        py::arg("q_len"),
        py::arg("past_len"),
        py::arg("num_heads"),
        py::arg("head_dim"),
        py::arg("key_cache"),
        py::arg("value_cache"),
        py::arg("max_seq_len"),
        py::arg("q_a"),
        py::arg("q_b"),
        py::arg("k_a"),
        py::arg("k_b"),
        py::arg("v_a"),
        py::arg("v_b"),
        py::arg("lora_temp")
    );
    m.def(
        "q4_attn_2",
        &q4_attn_2,
        "int4 quantized LLaMA self attention (WIP, 2nd version)",
        py::arg("x"),
        py::arg("attn_output"),
        py::arg("o_proj"),
        py::arg("o_a"),
        py::arg("o_b"),
        py::arg("lora_temp")
    );
    m.def(
        "q4_mlp",
        &q4_mlp,
        "int4 quantized LLaMA mlp",
        py::arg("x"),
        py::arg("rms_norm_weight"),
        py::arg("epsilon"),
        py::arg("gate"),
        py::arg("up"),
        py::arg("down"),
        py::arg("gate_a"),
        py::arg("gate_b"),
        py::arg("up_a"),
        py::arg("up_b"),
        py::arg("down_a"),
        py::arg("down_b"),
        py::arg("lora_temp")
    );
    m.def(
        "column_remap",
        &column_remap,
        "Column remap for half-precision tensor",
        py::arg("x"),
        py::arg("x_new"),
        py::arg("x_map")
    );
    m.def(
        "rms_norm",
        &rms_norm,
        "RMS layernorm",
        py::arg("x"),
        py::arg("w"),
        py::arg("out"),
        py::arg("epsilon")
    );
    m.def(
        "rope_",
        &rope_,
        "RoPE rotary positional embeddings",
        py::arg("x"),
        py::arg("sin"),
        py::arg("cos"),
        py::arg("past_len"),
        py::arg("num_heads"),
        py::arg("head_dim")
    );
    m.def(
        "half_matmul",
        &half_matmul,
        "Matmul half @ half -> half, custom kernel",
        py::arg("x"),
        py::arg("w"),
        py::arg("out")
    );
    m.def(
        "half_matmul_cublas",
        &half_matmul_cublas,
        "Matmul half @ half -> half using cuBLAS",
        py::arg("x"),
        py::arg("w"),
        py::arg("out")
    );

    m.def(
        "rep_penalty",
        &rep_penalty,
        "Repetition penalty",
        py::arg("sequence"),
        py::arg("rep_mask"),
        py::arg("penalty_max"),
        py::arg("sustain"),
        py::arg("decay")
    );
    m.def(
        "apply_rep_penalty",
        &apply_rep_penalty,
        "Apply repetition penalty",
        py::arg("sequence"),
        py::arg("penalty_max"),
        py::arg("sustain"),
        py::arg("decay"),
        py::arg("logits")
    );
}
