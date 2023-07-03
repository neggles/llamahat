#ifndef _COLUMN_REMAP_CUH
#define _COLUMN_REMAP_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

void column_remap_cuda(
    const half*     x,
    half*           x_new,
    const int       x_height,
    const int       x_width,
    const uint32_t* x_map
);

#endif // _COLUMN_REMAP_CUH
