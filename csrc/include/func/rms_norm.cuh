#ifndef _RMS_NORM_CUH
#define _RMS_NORM_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "tuning.h"

void rms_norm_cuda(
    ExLlamaTuning* tuningParams,
    half*          x,
    const half*    w,
    half*          out,
    const float    epsilon,
    const int      rows,
    const int      dim,
    const int      device_index
);

#endif // _RMS_NORM_CUH
