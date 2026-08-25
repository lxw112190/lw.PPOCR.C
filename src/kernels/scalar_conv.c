#include "scalar_kernels.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

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

static int spatial_output(
    int32_t input,
    int32_t kernel,
    int32_t stride,
    int32_t dilation,
    int32_t pad_before,
    int32_t pad_after,
    int32_t* output) {
    int64_t effective_kernel;
    int64_t numerator;
    int64_t value;
    if (input <= 0 || kernel <= 0 || stride <= 0 || dilation <= 0 ||
        pad_before < 0 || pad_after < 0) {
        return 0;
    }
    effective_kernel = (int64_t)(kernel - 1) * dilation + 1;
    numerator = (int64_t)input + pad_before + pad_after - effective_kernel;
    if (numerator < 0) {
        return 0;
    }
    value = numerator / stride + 1;
    if (value <= 0 || value > INT32_MAX) {
        return 0;
    }
    *output = (int32_t)value;
    return 1;
}

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
    uint32_t groups) {
    uint64_t input_count;
    uint64_t weight_count;
    uint64_t output_count;
    int32_t expected_height;
    int32_t expected_width;
    uint32_t input_channels_per_group;
    uint32_t output_channels_per_group;
    uint32_t batch;
    uint32_t group;
    lw_status status;
    if (input == NULL || weights == NULL || output == NULL ||
        input_dimensions == NULL || weight_dimensions == NULL ||
        output_dimensions == NULL || kernel == NULL || strides == NULL ||
        dilations == NULL || pads == NULL || output == input ||
        output == weights || (bias != NULL && output == bias)) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    if ((bias == NULL && bias_count != 0u) ||
        (bias != NULL && bias_count == 0u) || groups == 0u) {
        return LW_STATUS_INVALID_SHAPE;
    }
    status = tensor_element_count(4u, input_dimensions, &input_count);
    if (status != LW_STATUS_OK) {
        return status;
    }
    status = tensor_element_count(4u, weight_dimensions, &weight_count);
    if (status != LW_STATUS_OK) {
        return status;
    }
    status = tensor_element_count(4u, output_dimensions, &output_count);
    if (status != LW_STATUS_OK) {
        return status;
    }
    (void)input_count;
    (void)weight_count;
    (void)output_count;
    if (weight_dimensions[0] != output_dimensions[1] ||
        output_dimensions[0] != input_dimensions[0] ||
        (uint64_t)(uint32_t)weight_dimensions[1] * groups !=
            (uint32_t)input_dimensions[1] ||
        (uint32_t)input_dimensions[1] % groups != 0u ||
        (uint32_t)output_dimensions[1] % groups != 0u ||
        weight_dimensions[2] != kernel[0] ||
        weight_dimensions[3] != kernel[1] ||
        (bias != NULL && bias_count != (uint32_t)output_dimensions[1]) ||
        !spatial_output(input_dimensions[2], kernel[0], strides[0], dilations[0],
                        pads[0], pads[2], &expected_height) ||
        !spatial_output(input_dimensions[3], kernel[1], strides[1], dilations[1],
                        pads[1], pads[3], &expected_width) ||
        output_dimensions[2] != expected_height ||
        output_dimensions[3] != expected_width) {
        return LW_STATUS_INVALID_SHAPE;
    }
    input_channels_per_group = (uint32_t)input_dimensions[1] / groups;
    output_channels_per_group = (uint32_t)output_dimensions[1] / groups;
    for (batch = 0u; batch < (uint32_t)input_dimensions[0]; ++batch) {
        for (group = 0u; group < groups; ++group) {
            uint32_t output_channel_in_group;
            uint32_t input_channel_start = group * input_channels_per_group;
            uint32_t output_channel_start = group * output_channels_per_group;
            for (output_channel_in_group = 0u;
                 output_channel_in_group < output_channels_per_group;
                 ++output_channel_in_group) {
                uint32_t output_channel = output_channel_start + output_channel_in_group;
                uint32_t output_y;
                for (output_y = 0u; output_y < (uint32_t)expected_height; ++output_y) {
                    int64_t input_y_start =
                        (int64_t)output_y * strides[0] - pads[0];
                    uint32_t output_x;
                    for (output_x = 0u; output_x < (uint32_t)expected_width; ++output_x) {
                        int64_t input_x_start =
                            (int64_t)output_x * strides[1] - pads[1];
                        float sum = bias == NULL ? 0.0f : bias[output_channel];
                        uint32_t input_channel_in_group;
                        for (input_channel_in_group = 0u;
                             input_channel_in_group < input_channels_per_group;
                             ++input_channel_in_group) {
                            uint32_t input_channel =
                                input_channel_start + input_channel_in_group;
                            uint32_t kernel_y;
                            for (kernel_y = 0u; kernel_y < (uint32_t)kernel[0]; ++kernel_y) {
                                int64_t input_y = input_y_start +
                                    (int64_t)kernel_y * dilations[0];
                                uint32_t kernel_x;
                                if (input_y < 0 || input_y >= input_dimensions[2]) {
                                    continue;
                                }
                                for (kernel_x = 0u; kernel_x < (uint32_t)kernel[1]; ++kernel_x) {
                                    int64_t input_x = input_x_start +
                                        (int64_t)kernel_x * dilations[1];
                                    uint64_t input_offset;
                                    uint64_t weight_offset;
                                    if (input_x < 0 || input_x >= input_dimensions[3]) {
                                        continue;
                                    }
                                    input_offset =
                                        (((uint64_t)batch * (uint32_t)input_dimensions[1] +
                                          input_channel) * (uint32_t)input_dimensions[2] +
                                         (uint32_t)input_y) * (uint32_t)input_dimensions[3] +
                                        (uint32_t)input_x;
                                    weight_offset =
                                        (((uint64_t)output_channel * input_channels_per_group +
                                          input_channel_in_group) * (uint32_t)kernel[0] +
                                         kernel_y) * (uint32_t)kernel[1] + kernel_x;
                                    sum += input[(size_t)input_offset] *
                                           weights[(size_t)weight_offset];
                                }
                            }
                        }
                        output[(size_t)(
                            (((uint64_t)batch * (uint32_t)output_dimensions[1] +
                              output_channel) * (uint32_t)expected_height + output_y) *
                             (uint32_t)expected_width + output_x)] = sum;
                    }
                }
            }
        }
    }
    return LW_STATUS_OK;
}

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
    const int32_t* dimensions) {
    uint64_t element_count;
    uint64_t spatial_count = 1u;
    uint32_t channel_count;
    uint32_t channel;
    lw_status status;
    if (input == NULL || scale == NULL || bias == NULL || mean == NULL ||
        variance == NULL || output == NULL || output == scale || output == bias ||
        output == mean || output == variance) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    if (rank < 2u || rank > LW_MAX_DIMS || dimensions == NULL ||
        !isfinite(epsilon) || epsilon <= 0.0f) {
        return LW_STATUS_INVALID_SHAPE;
    }
    status = tensor_element_count(rank, dimensions, &element_count);
    if (status != LW_STATUS_OK) {
        return status;
    }
    channel_count = (uint32_t)dimensions[1];
    if (parameter_count != channel_count) {
        return LW_STATUS_INVALID_SHAPE;
    }
    for (channel = 2u; channel < rank; ++channel) {
        spatial_count *= (uint32_t)dimensions[channel];
    }
    for (channel = 0u; channel < channel_count; ++channel) {
        float denominator = variance[channel] + epsilon;
        if (!isfinite(scale[channel]) || !isfinite(bias[channel]) ||
            !isfinite(mean[channel]) || !isfinite(variance[channel]) ||
            !isfinite(denominator) || denominator <= 0.0f) {
            return LW_STATUS_INVALID_ARGUMENT;
        }
    }
    for (channel = 0u; channel < channel_count; ++channel) {
        float factor = scale[channel] / sqrtf(variance[channel] + epsilon);
        uint32_t batch;
        for (batch = 0u; batch < (uint32_t)dimensions[0]; ++batch) {
            uint64_t offset =
                ((uint64_t)batch * channel_count + channel) * spatial_count;
            uint64_t spatial;
            for (spatial = 0u; spatial < spatial_count; ++spatial) {
                float value = input[(size_t)(offset + spatial)];
                output[(size_t)(offset + spatial)] =
                    (value - mean[channel]) * factor + bias[channel];
            }
        }
    }
    (void)element_count;
    return LW_STATUS_OK;
}
