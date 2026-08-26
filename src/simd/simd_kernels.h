#ifndef LW_SIMD_KERNELS_H
#define LW_SIMD_KERNELS_H

/* Architecture-specific kernels; use only after checking lw_cpu_simd_level(). */

#include "scalar_kernels.h"

#include <stdint.h>

void lw_sse2_binary_contiguous_f32(lw_scalar_binary_op operation, const float* left,
                                   const float* right, float* output, uint64_t element_count);
void lw_sse2_binary_right_scalar_f32(lw_scalar_binary_op operation, const float* left, float right,
                                     float* output, uint64_t element_count);
void lw_avx2_binary_contiguous_f32(lw_scalar_binary_op operation, const float* left,
                                   const float* right, float* output, uint64_t element_count);
void lw_avx2_binary_right_scalar_f32(lw_scalar_binary_op operation, const float* left, float right,
                                     float* output, uint64_t element_count);

void lw_sse2_matmul_shared_f32(const float* input, const float* weights, float* output,
                               uint32_t batch_count, uint32_t rows, uint32_t inner_dimension,
                               uint32_t columns);
void lw_avx2_matmul_shared_f32(const float* input, const float* weights, float* output,
                               uint32_t batch_count, uint32_t rows, uint32_t inner_dimension,
                               uint32_t columns);
void lw_sse2_conv1x1_unit_f32(const float* input, const float* weights, const float* bias,
                              float* output, const int32_t input_dimensions[4],
                              const int32_t output_dimensions[4], uint32_t groups,
                              uint32_t input_channels_per_group,
                              uint32_t output_channels_per_group);
void lw_avx2_conv1x1_unit_f32(const float* input, const float* weights, const float* bias,
                              float* output, const int32_t input_dimensions[4],
                              const int32_t output_dimensions[4], uint32_t groups,
                              uint32_t input_channels_per_group,
                              uint32_t output_channels_per_group);
void lw_sse2_depthwise_conv3x3_unit_pad1_f32(const float* input, const float* weights,
                                             const float* bias, float* output,
                                             const int32_t dimensions[4]);
void lw_avx2_depthwise_conv3x3_unit_pad1_f32(const float* input, const float* weights,
                                             const float* bias, float* output,
                                             const int32_t dimensions[4]);
void lw_sse2_depthwise_conv5x5_unit_pad2_f32(const float* input, const float* weights,
                                             const float* bias, float* output,
                                             const int32_t dimensions[4]);
void lw_avx2_depthwise_conv5x5_unit_pad2_f32(const float* input, const float* weights,
                                             const float* bias, float* output,
                                             const int32_t dimensions[4]);
void lw_sse2_conv3x3_unit_pad1_f32(const float* input, const float* weights, const float* bias,
                                   float* output, const int32_t input_dimensions[4],
                                   const int32_t output_dimensions[4]);
void lw_avx2_conv3x3_unit_pad1_f32(const float* input, const float* weights, const float* bias,
                                   float* output, const int32_t input_dimensions[4],
                                   const int32_t output_dimensions[4]);
void lw_sse2_conv2x2_unit_pad_end1_f32(const float* input, const float* weights, const float* bias,
                                       float* output, const int32_t input_dimensions[4],
                                       const int32_t output_dimensions[4]);
void lw_avx2_conv2x2_unit_pad_end1_f32(const float* input, const float* weights, const float* bias,
                                       float* output, const int32_t input_dimensions[4],
                                       const int32_t output_dimensions[4]);
void lw_sse2_conv3x3_stride2_pad1_f32(const float* input, const float* weights, const float* bias,
                                      float* output, const int32_t input_dimensions[4],
                                      const int32_t output_dimensions[4]);
void lw_avx2_conv3x3_stride2_pad1_f32(const float* input, const float* weights, const float* bias,
                                      float* output, const int32_t input_dimensions[4],
                                      const int32_t output_dimensions[4]);

#endif
