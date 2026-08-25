#ifndef LW_SCALAR_KERNELS_H
#define LW_SCALAR_KERNELS_H

#include "lw_infer.h"

#include <stdint.h>

typedef enum lw_scalar_binary_op {
    LW_SCALAR_BINARY_ADD = 1,
    LW_SCALAR_BINARY_MUL = 2,
    LW_SCALAR_BINARY_DIV = 3
} lw_scalar_binary_op;

void lw_scalar_binary_contiguous_f32(
    lw_scalar_binary_op operation,
    const float* left,
    const float* right,
    float* output,
    uint64_t element_count);
void lw_scalar_binary_right_scalar_f32(
    lw_scalar_binary_op operation,
    const float* left,
    float right,
    float* output,
    uint64_t element_count);

lw_status lw_scalar_binary_f32(
    lw_scalar_binary_op operation,
    const float* left,
    uint32_t left_rank,
    const int32_t* left_dimensions,
    const float* right,
    uint32_t right_rank,
    const int32_t* right_dimensions,
    float* output,
    uint32_t output_rank,
    const int32_t* output_dimensions);

lw_status lw_scalar_relu_f32(const float* input, float* output, uint64_t element_count);
lw_status lw_scalar_erf_f32(const float* input, float* output, uint64_t element_count);
lw_status lw_scalar_hard_sigmoid_f32(
    const float* input,
    float* output,
    uint64_t element_count,
    float alpha,
    float beta);
lw_status lw_scalar_sigmoid_f32(
    const float* input,
    float* output,
    uint64_t element_count);
lw_status lw_scalar_softmax_f32(
    const float* input,
    float* output,
    uint32_t rank,
    const int32_t* dimensions,
    int32_t axis);
lw_status lw_scalar_transpose_f32(
    const float* input,
    float* output,
    uint32_t rank,
    const int32_t* input_dimensions,
    uint32_t permutation_count,
    const int32_t* permutation,
    const int32_t* output_dimensions);
lw_status lw_scalar_squeeze_f32(
    const float* input,
    float* output,
    uint32_t input_rank,
    const int32_t* input_dimensions,
    uint32_t axes_count,
    const int32_t* axes,
    uint32_t output_rank,
    const int32_t* output_dimensions);
lw_status lw_scalar_unsqueeze_f32(
    const float* input,
    float* output,
    uint32_t input_rank,
    const int32_t* input_dimensions,
    uint32_t axes_count,
    const int32_t* axes,
    uint32_t output_rank,
    const int32_t* output_dimensions);
lw_status lw_scalar_reshape_f32(
    const float* input,
    float* output,
    uint32_t input_rank,
    const int32_t* input_dimensions,
    uint32_t output_rank,
    const int32_t* output_dimensions);
lw_status lw_scalar_concat_f32(
    const float* const* inputs,
    uint32_t input_count,
    const uint32_t* input_ranks,
    const int32_t* const* input_dimensions,
    float* output,
    uint32_t output_rank,
    const int32_t* output_dimensions,
    int32_t axis);
lw_status lw_scalar_resize_nearest_f32(
    const float* input,
    float* output,
    uint32_t rank,
    const int32_t* input_dimensions,
    const int32_t* output_dimensions,
    const float* scales);
lw_status lw_scalar_reduce_mean_f32(
    const float* input,
    float* output,
    uint32_t input_rank,
    const int32_t* input_dimensions,
    uint32_t axes_count,
    const int32_t* axes,
    uint32_t keep_dimensions,
    uint32_t no_op_with_empty_axes,
    uint32_t output_rank,
    const int32_t* output_dimensions);
lw_status lw_scalar_average_pool2d_f32(
    const float* input,
    float* output,
    const int32_t input_dimensions[4],
    const int32_t output_dimensions[4],
    const int32_t kernel[2],
    const int32_t strides[2],
    const int32_t pads[4],
    uint32_t ceil_mode,
    uint32_t count_include_pad);
lw_status lw_scalar_max_pool2d_f32(
    const float* input,
    float* output,
    const int32_t input_dimensions[4],
    const int32_t output_dimensions[4],
    const int32_t kernel[2],
    const int32_t strides[2],
    const int32_t pads[4],
    uint32_t ceil_mode);
lw_status lw_scalar_matmul_shared_f32(
    const float* input,
    const float* weights,
    float* output,
    uint32_t batch_count,
    uint32_t rows,
    uint32_t inner_dimension,
    uint32_t columns);
lw_status lw_matmul_shared_f32(
    const float* input,
    const float* weights,
    float* output,
    uint32_t batch_count,
    uint32_t rows,
    uint32_t inner_dimension,
    uint32_t columns);
lw_status lw_scalar_conv2d_f32(
    const float* input,
    const float* weights,
    const float* bias,
    uint32_t bias_count,
    float* output,
    const int32_t input_dimensions[4],
    const int32_t weight_dimensions[4],
    const int32_t output_dimensions[4],
    const int32_t kernel[2],
    const int32_t strides[2],
    const int32_t dilations[2],
    const int32_t pads[4],
    uint32_t groups);
lw_status lw_scalar_conv_transpose2d_f32(
    const float* input,
    const float* weights,
    const float* bias,
    uint32_t bias_count,
    float* output,
    const int32_t input_dimensions[4],
    const int32_t weight_dimensions[4],
    const int32_t output_dimensions[4],
    const int32_t kernel[2],
    const int32_t strides[2],
    const int32_t dilations[2],
    const int32_t pads[4],
    uint32_t groups);
void lw_scalar_conv1x1_unit_f32(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    const int32_t input_dimensions[4],
    const int32_t output_dimensions[4],
    uint32_t groups,
    uint32_t input_channels_per_group,
    uint32_t output_channels_per_group);
void lw_scalar_depthwise_conv3x3_unit_pad1_f32(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    const int32_t dimensions[4]);
void lw_scalar_conv3x3_stride2_pad1_f32(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    const int32_t input_dimensions[4],
    const int32_t output_dimensions[4]);
lw_status lw_scalar_batch_normalization_f32(
    const float* input,
    const float* scale,
    const float* bias,
    const float* mean,
    const float* variance,
    uint32_t parameter_count,
    float epsilon,
    float* output,
    uint32_t rank,
    const int32_t* dimensions);

#endif
