#ifndef _Q4_MLP_CUH
#define _Q4_MLP_CUH

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "func/q4_matrix.cuh"
#include "tuning.h"

void q4_mlp_cuda(
    ExLlamaTuning* tuningParams,
    half*          x,               // shape == (height, dim)
    const half*    rms_norm_weight, // shape == (x.shape[1],) == (dim,)
    float          epsilon,
    Q4Matrix*      gate,
    Q4Matrix*      up,
    Q4Matrix*      down,
    const int      height,
    const int      dim,
    const half*    gate_a,
    const half*    gate_b,
    const int      gate_rank,
    const half*    up_a,
    const half*    up_b,
    const int      up_rank,
    const half*    down_a,
    const half*    down_b,
    const int      down_rank,
    half*          lora_temp,
    cublasHandle_t handle,
    const int      device_index
);

#endif // _Q4_MLP_CUH
