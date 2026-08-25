#include "scalar_kernels.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>

static int normalize_axis(int32_t axis, uint32_t rank, uint32_t* normalized) {
    int64_t value = axis;
    if (value < 0) {
        value += rank;
    }
    if (value < 0 || value >= rank) {
        return 0;
    }
    *normalized = (uint32_t)value;
    return 1;
}

static lw_status tensor_element_count(
    uint32_t rank,
    const int32_t* dimensions,
    uint64_t* count) {
    uint64_t value = 1u;
    uint32_t index;
    if (rank > LW_MAX_DIMS || (rank != 0u && dimensions == NULL)) {
        return LW_STATUS_INVALID_SHAPE;
    }
    for (index = 0u; index < rank; ++index) {
        if (dimensions[index] <= 0 ||
            value > UINT64_MAX / (uint32_t)dimensions[index]) {
            return LW_STATUS_INVALID_SHAPE;
        }
        value *= (uint32_t)dimensions[index];
    }
    if (value > (uint64_t)(SIZE_MAX / sizeof(float))) {
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    *count = value;
    return LW_STATUS_OK;
}

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
    const int32_t* output_dimensions) {
    int reduced[LW_MAX_DIMS] = {0};
    uint32_t input_to_output[LW_MAX_DIMS];
    uint32_t coordinates[LW_MAX_DIMS] = {0u};
    uint64_t output_strides[LW_MAX_DIMS] = {0u};
    uint64_t input_count;
    uint64_t output_count;
    uint64_t reduction_count = 1u;
    uint64_t input_index;
    uint64_t output_offset = 0u;
    uint64_t stride = 1u;
    uint32_t axis;
    uint32_t expected_output_rank = 0u;
    lw_status status;
    if (input == NULL || output == NULL || input == output) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    if (axes_count > LW_MAX_DIMS || (axes_count != 0u && axes == NULL) ||
        keep_dimensions > 1u || no_op_with_empty_axes > 1u) {
        return LW_STATUS_INVALID_SHAPE;
    }
    status = tensor_element_count(input_rank, input_dimensions, &input_count);
    if (status != LW_STATUS_OK) {
        return status;
    }
    if (axes_count == 0u && no_op_with_empty_axes == 0u) {
        for (axis = 0u; axis < input_rank; ++axis) {
            reduced[axis] = 1;
        }
    } else {
        for (axis = 0u; axis < axes_count; ++axis) {
            uint32_t normalized;
            if (!normalize_axis(axes[axis], input_rank, &normalized) || reduced[normalized]) {
                return LW_STATUS_INVALID_SHAPE;
            }
            reduced[normalized] = 1;
        }
    }
    for (axis = 0u; axis < input_rank; ++axis) {
        input_to_output[axis] = UINT32_MAX;
        if (reduced[axis]) {
            if (reduction_count > UINT64_MAX / (uint32_t)input_dimensions[axis]) {
                return LW_STATUS_OUT_OF_BOUNDS;
            }
            reduction_count *= (uint32_t)input_dimensions[axis];
            if (keep_dimensions != 0u) {
                if (expected_output_rank >= output_rank || output_dimensions == NULL ||
                    output_dimensions[expected_output_rank] != 1) {
                    return LW_STATUS_INVALID_SHAPE;
                }
                ++expected_output_rank;
            }
        } else {
            if (expected_output_rank >= output_rank || output_dimensions == NULL ||
                output_dimensions[expected_output_rank] != input_dimensions[axis]) {
                return LW_STATUS_INVALID_SHAPE;
            }
            input_to_output[axis] = expected_output_rank++;
        }
    }
    if (expected_output_rank != output_rank) {
        return LW_STATUS_INVALID_SHAPE;
    }
    status = tensor_element_count(output_rank, output_dimensions, &output_count);
    if (status != LW_STATUS_OK || output_count > input_count) {
        return status == LW_STATUS_OK ? LW_STATUS_INVALID_SHAPE : status;
    }
    for (axis = output_rank; axis > 0u; --axis) {
        output_strides[axis - 1u] = stride;
        stride *= (uint32_t)output_dimensions[axis - 1u];
    }
    for (input_index = 0u; input_index < output_count; ++input_index) {
        output[(size_t)input_index] = 0.0f;
    }
    for (input_index = 0u; input_index < input_count; ++input_index) {
        output[(size_t)output_offset] += input[(size_t)input_index];
        for (axis = input_rank; axis > 0u; --axis) {
            uint32_t input_axis = axis - 1u;
            uint32_t mapped_axis = input_to_output[input_axis];
            ++coordinates[input_axis];
            if (coordinates[input_axis] < (uint32_t)input_dimensions[input_axis]) {
                if (mapped_axis != UINT32_MAX) {
                    output_offset += output_strides[mapped_axis];
                }
                break;
            }
            coordinates[input_axis] = 0u;
            if (mapped_axis != UINT32_MAX) {
                output_offset -= output_strides[mapped_axis] *
                    ((uint32_t)input_dimensions[input_axis] - 1u);
            }
        }
    }
    for (input_index = 0u; input_index < output_count; ++input_index) {
        output[(size_t)input_index] /= (float)reduction_count;
    }
    return LW_STATUS_OK;
}

static int spatial_output(
    int32_t input,
    int32_t kernel,
    int32_t stride,
    int32_t pad_before,
    int32_t pad_after,
    uint32_t ceil_mode,
    int32_t* output) {
    int64_t numerator;
    int64_t value;
    if (input <= 0 || kernel <= 0 || stride <= 0 ||
        pad_before < 0 || pad_after < 0) {
        return 0;
    }
    numerator = (int64_t)input + pad_before + pad_after - kernel;
    if (numerator < 0) {
        return 0;
    }
    value = (numerator + (ceil_mode != 0u ? stride - 1 : 0)) / stride + 1;
    if (ceil_mode != 0u && (value - 1) * stride >= (int64_t)input + pad_before) {
        --value;
    }
    if (value <= 0 || value > INT32_MAX) {
        return 0;
    }
    *output = (int32_t)value;
    return 1;
}

lw_status lw_scalar_average_pool2d_f32(
    const float* input,
    float* output,
    const int32_t input_dimensions[4],
    const int32_t output_dimensions[4],
    const int32_t kernel[2],
    const int32_t strides[2],
    const int32_t pads[4],
    uint32_t ceil_mode,
    uint32_t count_include_pad) {
    uint64_t input_count;
    uint64_t output_count;
    int32_t expected_height;
    int32_t expected_width;
    uint32_t batch;
    uint32_t channel;
    uint32_t output_y;
    uint32_t output_x;
    lw_status status;
    if (input == NULL || output == NULL || input == output ||
        input_dimensions == NULL || output_dimensions == NULL ||
        kernel == NULL || strides == NULL || pads == NULL) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    if (ceil_mode > 1u || count_include_pad > 1u) {
        return LW_STATUS_INVALID_SHAPE;
    }
    status = tensor_element_count(4u, input_dimensions, &input_count);
    if (status != LW_STATUS_OK) {
        return status;
    }
    status = tensor_element_count(4u, output_dimensions, &output_count);
    if (status != LW_STATUS_OK) {
        return status;
    }
    (void)input_count;
    (void)output_count;
    if (!spatial_output(input_dimensions[2], kernel[0], strides[0],
                        pads[0], pads[2], ceil_mode, &expected_height) ||
        !spatial_output(input_dimensions[3], kernel[1], strides[1],
                        pads[1], pads[3], ceil_mode, &expected_width) ||
        output_dimensions[0] != input_dimensions[0] ||
        output_dimensions[1] != input_dimensions[1] ||
        output_dimensions[2] != expected_height ||
        output_dimensions[3] != expected_width) {
        return LW_STATUS_INVALID_SHAPE;
    }
    for (batch = 0u; batch < (uint32_t)input_dimensions[0]; ++batch) {
        for (channel = 0u; channel < (uint32_t)input_dimensions[1]; ++channel) {
            for (output_y = 0u; output_y < (uint32_t)expected_height; ++output_y) {
                int64_t input_y_start = (int64_t)output_y * strides[0] - pads[0];
                for (output_x = 0u; output_x < (uint32_t)expected_width; ++output_x) {
                    int64_t input_x_start = (int64_t)output_x * strides[1] - pads[1];
                    float sum = 0.0f;
                    uint64_t valid_count = 0u;
                    int32_t kernel_y;
                    int32_t kernel_x;
                    uint64_t output_offset =
                        (((uint64_t)batch * (uint32_t)input_dimensions[1] + channel) *
                         (uint32_t)expected_height + output_y) *
                         (uint32_t)expected_width + output_x;
                    for (kernel_y = 0; kernel_y < kernel[0]; ++kernel_y) {
                        int64_t input_y = input_y_start + kernel_y;
                        for (kernel_x = 0; kernel_x < kernel[1]; ++kernel_x) {
                            int64_t input_x = input_x_start + kernel_x;
                            if (input_y >= 0 && input_y < input_dimensions[2] &&
                                input_x >= 0 && input_x < input_dimensions[3]) {
                                uint64_t input_offset =
                                    (((uint64_t)batch * (uint32_t)input_dimensions[1] + channel) *
                                     (uint32_t)input_dimensions[2] + (uint32_t)input_y) *
                                     (uint32_t)input_dimensions[3] + (uint32_t)input_x;
                                sum += input[(size_t)input_offset];
                                ++valid_count;
                            }
                        }
                    }
                    if (count_include_pad != 0u) {
                        valid_count = (uint64_t)(uint32_t)kernel[0] *
                                      (uint32_t)kernel[1];
                    }
                    if (valid_count == 0u) {
                        return LW_STATUS_INVALID_SHAPE;
                    }
                    output[(size_t)output_offset] = sum / (float)valid_count;
                }
            }
        }
    }
    return LW_STATUS_OK;
}
