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
void lw_avx2_matmul_shared_f32(
    const float* input,
    const float* weights,
    float* output,
    uint32_t batch_count,
    uint32_t rows,
    uint32_t inner_dimension,
    uint32_t columns);
void lw_sse2_conv1x1_unit_f32(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    const int32_t input_dimensions[4],
    const int32_t output_dimensions[4],
    uint32_t groups,
    uint32_t input_channels_per_group,
    uint32_t output_channels_per_group);
void lw_avx2_conv1x1_unit_f32(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    const int32_t input_dimensions[4],
    const int32_t output_dimensions[4],
    uint32_t groups,
    uint32_t input_channels_per_group,
    uint32_t output_channels_per_group);

#endif
