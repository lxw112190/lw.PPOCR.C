#ifndef LW_SIMD_KERNELS_H
#define LW_SIMD_KERNELS_H

#include <stdint.h>

void lw_sse2_matmul_shared_f32(
    const float* input,
    const float* weights,
    float* output,
    uint32_t batch_count,
    uint32_t rows,
    uint32_t inner_dimension,
    uint32_t columns);

#endif
