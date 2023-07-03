#ifndef _Q4_MATMUL_CUH
#define _Q4_MATMUL_CUH

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

#include "q4_matrix.cuh"
#include "tuning.h"

// Workaround for hipify_python using rocblas instead of hipblas.
#if defined(USE_ROCM)
#    include <hipblas/hipblas.h>
#    define rocblas_handle hipblasHandle_t
#endif

void q4_matmul_cuda(
    ExLlamaTuning*  tuningParams,
    const half*     x,
    const int       x_height,
    const Q4Matrix* w,
    half*           out,
    bool            no_zero    = false,
    cudaStream_t    alt_stream = NULL
);

void q4_matmul_recons_cuda(
    ExLlamaTuning*       tuningParams,
    const half*          x,
    const int            x_height,
    Q4Matrix*            w,
    half*                out,
    const cublasHandle_t handle,
    bool                 no_zero = false
);

#endif // _Q4_MATMUL_CUH
