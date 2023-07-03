#ifndef _ROPE_CUH
#define _ROPE_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "tuning.h"

void rope_cuda(
    ExLlamaTuning* tuningParams,
    half*          x,
    const half*    sin,
    const half*    cos,
    const int      bsz,
    const int      rows,
    const int      head_dim,
    const int      num_heads,
    const int      past_len,
    cudaStream_t   alt_stream = NULL
);

#endif // _ROPE_CUH
