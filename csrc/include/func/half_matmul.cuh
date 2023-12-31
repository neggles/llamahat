#ifndef _HALF_MATMUL_CUH
#define _HALF_MATMUL_CUH

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "tuning.h"

// Workaround for hipify_python using rocblas instead of hipblas.
#if defined(USE_ROCM)
#    include <hipblas/hipblas.h>
#    define rocblas_handle hipblasHandle_t
#endif

void half_matmul_cuda(
    const half*  x,
    const half*  w,
    half*        out,
    const int    height,
    const int    dim,
    const int    width,
    cudaStream_t alt_stream = NULL
);

void half_matmul_cublas_cuda(
    ExLlamaTuning* tuningParams,
    const half*    x,
    const half*    w,
    half*          out,
    const int      height,
    const int      dim,
    const int      width,
    cublasHandle_t handle,
    bool           no_zero    = false,
    cudaStream_t   alt_stream = NULL
);

void half_matmul_small_cuda(
    ExLlamaTuning* tuningParams,
    const half*    x,
    const half*    w,
    half*          out,
    const int      height,
    const int      dim,
    const int      width,
    bool           no_zero    = false,
    cudaStream_t   alt_stream = NULL
);

#endif // _HALF_MATMUL_CUH
