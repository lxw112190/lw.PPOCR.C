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
void lw_avx2_relu_contiguous_f32(const float* input, float* output, uint64_t element_count);
void lw_avx2_binary_channel_f32(lw_scalar_binary_op operation, const float* full, const float* channel, float* output, uint64_t pixels, uint32_t channels, int broadcast_is_left);
void lw_avx2_binary_channel_nchw_f32(lw_scalar_binary_op operation, const float* full, const float* channel, float* output, uint64_t spatial, uint32_t channels, int broadcast_is_left);
void lw_avx2_hard_sigmoid_contiguous_f32(const float* input, float* output, uint64_t element_count, float alpha, float beta);
void lw_avx2_hard_sigmoid_exact_f32(const float* input, float* output, uint64_t element_count, float alpha, float beta);
void lw_avx2_relu_f32(const float* input, float* output, uint64_t element_count);
void lw_avx2_erf_f32(const float* input, float* output, uint64_t element_count);
void lw_avx2_gelu_f32(const float* input, float* output, uint64_t element_count);
/* Transpose a rows x cols row-major float matrix into cols x rows row-major.
 * Pure data movement, bit-identical to the scalar transpose. */
void lw_avx2_transpose_2d_f32(const float* input, float* output,
                              uint32_t rows, uint32_t cols);
void lw_avx2_softmax_contiguous_f32(const float* input, float* output, uint64_t row_count,
                                    uint64_t axis_count);
void lw_avx2_ctc_emitted_softmax_contiguous_f32(const float* input,
                                                const uint32_t* best_indices,
                                                float* emitted_probabilities,
                                                uint64_t row_count, uint64_t axis_count);
void lw_wasm128_erf_f32(const float* input, float* output, uint64_t element_count);
void lw_wasm128_gelu_f32(const float* input, float* output, uint64_t element_count);
void lw_wasm128_binary_contiguous_f32(lw_scalar_binary_op operation,
                                      const float* left, const float* right,
                                      float* output, uint64_t count);
void lw_wasm128_binary_scalar_f32(lw_scalar_binary_op operation,
                                  const float* tensor, float scalar, float* output,
                                  uint64_t count, int scalar_is_left);
void lw_wasm128_binary_channel_nhwc_f32(lw_scalar_binary_op operation,
    const float* tensor, const float* channel, float* output,
    uint64_t pixels, uint32_t channels, int channel_is_left);
void lw_wasm128_affine_nhwc_f32(const float* input, const float* mul,
    const float* add, float* output, uint32_t pixels, uint32_t channels);
void lw_wasm128_relu_f32(const float* input, float* output, uint64_t count);
int lw_wasm128_threshold_bitmap_f32(const float* prediction, uint8_t* bitmap,
                                    uint64_t pixel_count, float threshold);
void lw_wasm128_reduce_mean_hw_f32(const float* input, float* output,
    uint32_t batch, uint32_t height, uint32_t width, uint32_t channels);
void lw_wasm128_pool_nhwc_f32(const float* input, float* output,
    uint32_t batch, uint32_t input_height, uint32_t input_width,
    uint32_t output_height, uint32_t output_width, uint32_t channels,
    uint32_t kernel_h, uint32_t kernel_w, uint32_t stride_h, uint32_t stride_w,
    uint32_t pad_top, uint32_t pad_left, uint8_t count_include_pad, uint8_t is_max);
void lw_wasm128_packed_matmul_shared_f32(const float* input,
    const float* packed_weights, float* output, uint32_t batch_count,
    uint32_t rows, uint32_t inner_dimension, uint32_t columns);

void lw_sse2_matmul_shared_f32(const float* input, const float* weights, float* output,
                               uint32_t batch_count, uint32_t rows, uint32_t inner_dimension,
                               uint32_t columns);
void lw_neon_packed_matmul_shared_f32(const float* input, const float* packed_weights,
                                      const float* bias, float* output, uint32_t batch_count,
                                      uint32_t rows, uint32_t inner_dimension, uint32_t columns);
void lw_avx2_matmul_shared_f32(const float* input, const float* weights, float* output,
                               uint32_t batch_count, uint32_t rows, uint32_t inner_dimension,
                               uint32_t columns);
void lw_avx2_packed_matmul_shared_f32(const float* input, const float* packed_weights,
                                      float* output, uint32_t batch_count, uint32_t rows,
                                      uint32_t inner_dimension, uint32_t columns);
void lw_avx2_packed_matmul_bias_argmax_f32(
    const float* input, const float* packed_weights, const float* bias, float* output,
    uint32_t* best_indices, uint32_t batch_count, uint32_t rows,
    uint32_t inner_dimension, uint32_t columns);
void lw_avx2_fma_packed_matmul_bias_argmax_f32(
    const float* input, const float* packed_weights, const float* bias, float* output,
    uint32_t* best_indices, uint32_t batch_count, uint32_t rows,
    uint32_t inner_dimension, uint32_t columns);
void lw_avx2_fma_packed_matmul_argmax_scores_f32(
    const float* input, const float* packed_weights, const float* bias,
    uint32_t* best_indices, float* scores, uint32_t batch_count, uint32_t rows,
    uint32_t inner_dimension, uint32_t columns);
void lw_avx2_ctc_row_probabilities_f32(
    const float* input, const float* packed_weights, const float* bias,
    const uint32_t* best_indices, const float* scores, float* probabilities,
    uint32_t rows, uint32_t inner_dimension, uint32_t columns);
int lw_avx2_threshold_bitmap_f32(const float* prediction, uint8_t* bitmap,
                                 uint64_t pixel_count, float threshold);
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
void lw_sse2_packed_conv1x1_f32(const float* input, const float* packed_weights, const float* bias,
                                float* output, const int32_t input_dimensions[4],
                                const int32_t output_dimensions[4]);
void lw_avx2_fma_packed_conv1x1_f32(const float* input, const float* packed_weights, const float* bias,
                                      float* output, const int32_t input_dimensions[4],
                                      const int32_t output_dimensions[4]);
void lw_avx2_fma_packed_conv1x1_8x8_f32(
    const float* input, const float* packed_weights, const float* bias, float* output,
    const int32_t input_dimensions[4], const int32_t output_dimensions[4]);
void lw_avx2_packed_conv1x1_f32(const float* input, const float* packed_weights, const float* bias,
                                float* output, const int32_t input_dimensions[4],
                                const int32_t output_dimensions[4]);
void lw_neon_packed_conv1x1_f32(const float* input, const float* packed_weights, const float* bias,
                                float* output, const int32_t input_dimensions[4],
                                const int32_t output_dimensions[4]);
void lw_lsx_packed_conv1x1_f32(const float* input, const float* packed_weights, const float* bias,
                               float* output, const int32_t input_dimensions[4],
                               const int32_t output_dimensions[4]);
void lw_sse2_depthwise_conv3x3_unit_pad1_f32(const float* input, const float* weights,
                                             const float* bias, float* output,
                                             const int32_t dimensions[4]);
void lw_avx2_depthwise_conv3x3_unit_pad1_f32(const float* input, const float* weights,
                                             const float* bias, float* output,
                                             const int32_t dimensions[4]);
void lw_sse2_depthwise_conv3x3_stride2x1_pad1_f32(
    const float* input, const float* weights, const float* bias, float* output,
    const int32_t input_dimensions[4], const int32_t output_dimensions[4]);
void lw_avx2_depthwise_conv3x3_stride2x1_pad1_f32(
    const float* input, const float* weights, const float* bias, float* output,
    const int32_t input_dimensions[4], const int32_t output_dimensions[4]);
void lw_sse2_depthwise_conv5x5_unit_pad2_f32(const float* input, const float* weights,
                                             const float* bias, float* output,
                                             const int32_t dimensions[4]);
void lw_avx2_depthwise_conv5x5_unit_pad2_f32(const float* input, const float* weights,
                                             const float* bias, float* output,
                                           const int32_t dimensions[4]);
void lw_neon_depthwise_conv5x5_unit_pad2_f32(const float* input, const float* weights,
                                            const float* bias, float* output,
                                            const int32_t dimensions[4]);
void lw_avx2_depthwise_conv9x9_unit_pad4_f32(const float* input, const float* weights,
                                             const float* bias, float* output,
                                             const int32_t dimensions[4]);
void lw_sse2_conv3x3_unit_pad1_f32(const float* input, const float* weights, const float* bias,
                                   float* output, const int32_t input_dimensions[4],
                                   const int32_t output_dimensions[4]);
void lw_avx2_conv3x3_unit_pad1_f32(const float* input, const float* weights, const float* bias,
                                   float* output, const int32_t input_dimensions[4],
                                   const int32_t output_dimensions[4]);
void lw_avx2_conv7x7_unit_pad3_f32(const float* input, const float* weights, const float* bias,
                                   float* output, const int32_t input_dimensions[4],
                                   const int32_t output_dimensions[4]);
void lw_avx2_conv7x7_four_outputs_unit_pad3_f32(
    const float* input, const float* weights, const float* bias, float* output,
    const int32_t input_dimensions[4], const int32_t output_dimensions[4]);
void lw_avx2_conv5x5_unit_pad2_f32(const float* input, const float* weights, const float* bias,
                                   float* output, const int32_t input_dimensions[4],
                                   const int32_t output_dimensions[4]);
void lw_avx2_conv5x5_four_outputs_unit_pad2_f32(
    const float* input, const float* weights, const float* bias, float* output,
    const int32_t input_dimensions[4], const int32_t output_dimensions[4]);
void lw_avx2_conv7x1_unit_pad3_f32(const float* input, const float* weights, const float* bias,
                                   float* output, const int32_t input_dimensions[4],
                                   const int32_t output_dimensions[4]);
void lw_avx2_conv1x7_unit_pad3_f32(const float* input, const float* weights, const float* bias,
                                   float* output, const int32_t input_dimensions[4],
                                   const int32_t output_dimensions[4]);
void lw_avx2_conv5x1_unit_pad2_f32(const float* input, const float* weights, const float* bias,
                                   float* output, const int32_t input_dimensions[4],
                                   const int32_t output_dimensions[4]);
void lw_avx2_conv1x5_unit_pad2_f32(const float* input, const float* weights, const float* bias,
                                   float* output, const int32_t input_dimensions[4],
                                   const int32_t output_dimensions[4]);
void lw_neon_conv3x3_unit_pad1_f32(const float* input, const float* weights, const float* bias,
                                   float* output, const int32_t input_dimensions[4],
                                   const int32_t output_dimensions[4]);
void lw_neon_depthwise_conv3x3_pad1_f32(
    const float* input, const float* weights, const float* bias, float* output,
    const int32_t input_dimensions[4], const int32_t output_dimensions[4], uint32_t stride_y);
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
void lw_neon_conv3x3_stride2_pad1_f32(const float* input, const float* weights, const float* bias,
                                      float* output, const int32_t input_dimensions[4],
                                      const int32_t output_dimensions[4]);
void lw_avx2_packed_conv3x3_stride2_pad1_f32(
    const float* input, const float* packed_weights, const float* bias, float* output,
    const int32_t input_dimensions[4], const int32_t output_dimensions[4]);
void lw_avx2_fma_packed_conv3x3_stride2_pad1_f32(
    const float* input, const float* packed_weights, const float* bias, float* output,
    const int32_t input_dimensions[4], const int32_t output_dimensions[4]);
void lw_avx2_conv_transpose2x2_stride2_f32(const float* input, const float* weights,
                                           const float* bias, float* output,
                                           const int32_t input_dimensions[4],
                                           const int32_t output_dimensions[4]);
void lw_avx2_conv_transpose2x2_stride2_range_f32(
    const float* input, const float* weights, const float* bias, float* output,
    const int32_t input_dimensions[4], const int32_t output_dimensions[4],
    uint32_t output_channel_begin, uint32_t output_channel_end);
void lw_sse2_conv_transpose2x2_stride2_f32(const float* input, const float* weights,
                                           const float* bias, float* output,
                                           const int32_t input_dimensions[4],
                                           const int32_t output_dimensions[4]);
void lw_sse2_conv_transpose2x2_stride2_range_f32(
    const float* input, const float* weights, const float* bias, float* output,
    const int32_t input_dimensions[4], const int32_t output_dimensions[4],
    uint32_t output_channel_begin, uint32_t output_channel_end);
void lw_neon_conv_transpose2x2_stride2_f32(const float* input, const float* weights,
                                           const float* bias, float* output,
                                           const int32_t input_dimensions[4],
                                           const int32_t output_dimensions[4]);
void lw_neon_conv_transpose2x2_stride2_range_f32(
    const float* input, const float* weights, const float* bias, float* output,
    const int32_t input_dimensions[4], const int32_t output_dimensions[4],
    uint32_t output_channel_begin, uint32_t output_channel_end);

#endif
