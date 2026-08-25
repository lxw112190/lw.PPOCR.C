#ifndef LW_SCALAR_KERNELS_H
#define LW_SCALAR_KERNELS_H

#include "lw_infer.h"

#include <stdint.h>

typedef enum lw_scalar_binary_op {
    LW_SCALAR_BINARY_ADD = 1,
    LW_SCALAR_BINARY_MUL = 2,
    LW_SCALAR_BINARY_DIV = 3
} lw_scalar_binary_op;

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
lw_status lw_scalar_matmul_shared_f32(
    const float* input,
    const float* weights,
    float* output,
    uint32_t batch_count,
    uint32_t rows,
    uint32_t inner_dimension,
    uint32_t columns);

#endif
