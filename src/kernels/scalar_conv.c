#include "scalar_kernels.h"

/*
 * Portable convolution implementations plus guarded SIMD dispatch.
 * The specialized 1x1, depthwise 3x3 and stride-2 3x3 paths preserve the same
 * tensor layout and accumulation order contract as the general fallback.
 */
#include "cpu_features.h"
#include "simd_kernels.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static lw_status tensor_element_count(uint32_t rank, const int32_t* dimensions, uint64_t* count) {
    uint64_t value = 1u;
    uint32_t index;
    if (rank > LW_MAX_DIMS || (rank != 0u && dimensions == NULL)) {
        return LW_STATUS_INVALID_SHAPE;
    }
    for (index = 0u; index < rank; ++index) {
        if (dimensions[index] <= 0 || value > UINT64_MAX / (uint32_t)dimensions[index]) {
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

static int spatial_output(int32_t input, int32_t kernel, int32_t stride, int32_t dilation,
                          int32_t pad_before, int32_t pad_after, int32_t* output) {
    int64_t effective_kernel;
    int64_t numerator;
    int64_t value;
    if (input <= 0 || kernel <= 0 || stride <= 0 || dilation <= 0 || pad_before < 0 ||
        pad_after < 0) {
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

static int transposed_spatial_output(int32_t input, int32_t kernel, int32_t stride,
                                     int32_t dilation, int32_t pad_before, int32_t pad_after,
                                     int32_t* output) {
    int64_t value;
    if (input <= 0 || kernel <= 0 || stride <= 0 || dilation <= 0 || pad_before < 0 ||
        pad_after < 0) {
        return 0;
    }
    value = (int64_t)(input - 1) * stride - pad_before - pad_after +
            (int64_t)(kernel - 1) * dilation + 1;
    if (value <= 0 || value > INT32_MAX) {
        return 0;
    }
    *output = (int32_t)value;
    return 1;
}

void lw_scalar_conv1x1_unit_f32(const float* input, const float* weights, const float* bias,
                                float* output, const int32_t input_dimensions[4],
                                const int32_t output_dimensions[4], uint32_t groups,
                                uint32_t input_channels_per_group,
                                uint32_t output_channels_per_group) {
    uint64_t channel_plane =
        (uint64_t)(uint32_t)input_dimensions[2] * (uint32_t)input_dimensions[3];
    uint32_t batch;
    for (batch = 0u; batch < (uint32_t)input_dimensions[0]; ++batch) {
        uint32_t group;
        for (group = 0u; group < groups; ++group) {
            uint32_t input_channel_start = group * input_channels_per_group;
            uint32_t output_channel_start = group * output_channels_per_group;
            const float* group_input =
                input +
                (size_t)(((uint64_t)batch * (uint32_t)input_dimensions[1] + input_channel_start) *
                         channel_plane);
            float* group_output =
                output +
                (size_t)(((uint64_t)batch * (uint32_t)output_dimensions[1] + output_channel_start) *
                         channel_plane);
            uint32_t output_channel_in_group;
            for (output_channel_in_group = 0u; output_channel_in_group < output_channels_per_group;
                 ++output_channel_in_group) {
                uint32_t output_channel = output_channel_start + output_channel_in_group;
                const float* output_channel_weights =
                    weights + (size_t)((uint64_t)output_channel * input_channels_per_group);
                float* output_channel_data =
                    group_output + (size_t)((uint64_t)output_channel_in_group * channel_plane);
                float initial = bias == NULL ? 0.0f : bias[output_channel];
                uint64_t spatial;
                uint32_t input_channel_in_group;
                for (spatial = 0u; spatial < channel_plane; ++spatial) {
                    output_channel_data[(size_t)spatial] = initial;
                }
                for (input_channel_in_group = 0u; input_channel_in_group < input_channels_per_group;
                     ++input_channel_in_group) {
                    const float* input_channel_data =
                        group_input + (size_t)((uint64_t)input_channel_in_group * channel_plane);
                    float weight = output_channel_weights[input_channel_in_group];
                    for (spatial = 0u; spatial < channel_plane; ++spatial) {
                        output_channel_data[(size_t)spatial] +=
                            input_channel_data[(size_t)spatial] * weight;
                    }
                }
            }
        }
    }
}

void lw_scalar_depthwise_conv3x3_unit_pad1_f32(const float* input, const float* weights,
                                               const float* bias, float* output,
                                               const int32_t dimensions[4]) {
    uint32_t height = (uint32_t)dimensions[2];
    uint32_t width = (uint32_t)dimensions[3];
    uint64_t channel_plane = (uint64_t)height * width;
    uint32_t batch;
    for (batch = 0u; batch < (uint32_t)dimensions[0]; ++batch) {
        uint32_t channel;
        for (channel = 0u; channel < (uint32_t)dimensions[1]; ++channel) {
            uint64_t channel_offset =
                ((uint64_t)batch * (uint32_t)dimensions[1] + channel) * channel_plane;
            const float* input_channel = input + (size_t)channel_offset;
            const float* channel_weights = weights + (size_t)channel * 9u;
            float* output_channel = output + (size_t)channel_offset;
            float initial = bias == NULL ? 0.0f : bias[channel];
            uint64_t spatial;
            uint32_t kernel_y;
            for (spatial = 0u; spatial < channel_plane; ++spatial) {
                output_channel[(size_t)spatial] = initial;
            }
            for (kernel_y = 0u; kernel_y < 3u; ++kernel_y) {
                uint32_t output_y_begin = kernel_y == 0u ? 1u : 0u;
                uint32_t output_y_end = kernel_y == 2u && height != 0u ? height - 1u : height;
                uint32_t kernel_x;
                for (kernel_x = 0u; kernel_x < 3u; ++kernel_x) {
                    uint32_t output_x_begin = kernel_x == 0u ? 1u : 0u;
                    uint32_t output_x_end = kernel_x == 2u && width != 0u ? width - 1u : width;
                    float weight = channel_weights[kernel_y * 3u + kernel_x];
                    uint32_t output_y;
                    for (output_y = output_y_begin; output_y < output_y_end; ++output_y) {
                        uint32_t input_y = output_y + kernel_y - 1u;
                        const float* input_row =
                            input_channel + (size_t)((uint64_t)input_y * width);
                        float* output_row = output_channel + (size_t)((uint64_t)output_y * width);
                        uint32_t output_x;
                        for (output_x = output_x_begin; output_x < output_x_end; ++output_x) {
                            uint32_t input_x = output_x + kernel_x - 1u;
                            output_row[output_x] += input_row[input_x] * weight;
                        }
                    }
                }
            }
        }
    }
}

void lw_scalar_depthwise_conv5x5_unit_pad2_f32(const float* input, const float* weights,
                                               const float* bias, float* output,
                                               const int32_t dimensions[4]) {
    uint32_t height = (uint32_t)dimensions[2];
    uint32_t width = (uint32_t)dimensions[3];
    uint64_t channel_plane = (uint64_t)height * width;
    uint32_t batch;
    for (batch = 0u; batch < (uint32_t)dimensions[0]; ++batch) {
        uint32_t channel;
        for (channel = 0u; channel < (uint32_t)dimensions[1]; ++channel) {
            uint64_t channel_offset =
                ((uint64_t)batch * (uint32_t)dimensions[1] + channel) * channel_plane;
            const float* input_channel = input + (size_t)channel_offset;
            const float* channel_weights = weights + (size_t)channel * 25u;
            float* output_channel = output + (size_t)channel_offset;
            float initial = bias == NULL ? 0.0f : bias[channel];
            uint64_t spatial;
            uint32_t kernel_y;
            for (spatial = 0u; spatial < channel_plane; ++spatial) {
                output_channel[(size_t)spatial] = initial;
            }
            for (kernel_y = 0u; kernel_y < 5u; ++kernel_y) {
                uint32_t output_y_begin = kernel_y < 2u ? 2u - kernel_y : 0u;
                uint32_t output_y_trim = kernel_y > 2u ? kernel_y - 2u : 0u;
                uint32_t output_y_end = output_y_trim < height ? height - output_y_trim : 0u;
                uint32_t kernel_x;
                for (kernel_x = 0u; kernel_x < 5u; ++kernel_x) {
                    uint32_t output_x_begin = kernel_x < 2u ? 2u - kernel_x : 0u;
                    uint32_t output_x_trim = kernel_x > 2u ? kernel_x - 2u : 0u;
                    uint32_t output_x_end = output_x_trim < width ? width - output_x_trim : 0u;
                    float weight = channel_weights[kernel_y * 5u + kernel_x];
                    uint32_t output_y;
                    for (output_y = output_y_begin; output_y < output_y_end; ++output_y) {
                        uint32_t input_y = output_y + kernel_y - 2u;
                        const float* input_row =
                            input_channel + (size_t)((uint64_t)input_y * width);
                        float* output_row = output_channel + (size_t)((uint64_t)output_y * width);
                        uint32_t output_x;
                        for (output_x = output_x_begin; output_x < output_x_end; ++output_x) {
                            uint32_t input_x = output_x + kernel_x - 2u;
                            output_row[output_x] += input_row[input_x] * weight;
                        }
                    }
                }
            }
        }
    }
}

void lw_scalar_conv3x3_unit_pad1_f32(const float* input, const float* weights, const float* bias,
                                     float* output, const int32_t input_dimensions[4],
                                     const int32_t output_dimensions[4]) {
    uint32_t height = (uint32_t)input_dimensions[2];
    uint32_t width = (uint32_t)input_dimensions[3];
    uint64_t channel_plane = (uint64_t)height * width;
    uint32_t batch;
    for (batch = 0u; batch < (uint32_t)input_dimensions[0]; ++batch) {
        const float* batch_input =
            input + (size_t)((uint64_t)batch * (uint32_t)input_dimensions[1] * channel_plane);
        float* batch_output =
            output + (size_t)((uint64_t)batch * (uint32_t)output_dimensions[1] * channel_plane);
        uint32_t output_channel;
        for (output_channel = 0u; output_channel < (uint32_t)output_dimensions[1];
             ++output_channel) {
            const float* output_channel_weights =
                weights + (size_t)((uint64_t)output_channel * (uint32_t)input_dimensions[1] * 9u);
            float* output_channel_data =
                batch_output + (size_t)((uint64_t)output_channel * channel_plane);
            float initial = bias == NULL ? 0.0f : bias[output_channel];
            uint64_t spatial;
            uint32_t input_channel;
            for (spatial = 0u; spatial < channel_plane; ++spatial) {
                output_channel_data[(size_t)spatial] = initial;
            }
            for (input_channel = 0u; input_channel < (uint32_t)input_dimensions[1];
                 ++input_channel) {
                const float* input_channel_data =
                    batch_input + (size_t)((uint64_t)input_channel * channel_plane);
                const float* weight_channel_data =
                    output_channel_weights + (size_t)((uint64_t)input_channel * 9u);
                uint32_t kernel_y;
                for (kernel_y = 0u; kernel_y < 3u; ++kernel_y) {
                    uint32_t output_y_begin = kernel_y == 0u ? 1u : 0u;
                    uint32_t output_y_end = kernel_y == 2u ? height - 1u : height;
                    uint32_t kernel_x;
                    for (kernel_x = 0u; kernel_x < 3u; ++kernel_x) {
                        uint32_t output_x_begin = kernel_x == 0u ? 1u : 0u;
                        uint32_t output_x_end = kernel_x == 2u ? width - 1u : width;
                        float weight = weight_channel_data[kernel_y * 3u + kernel_x];
                        uint32_t output_y;
                        for (output_y = output_y_begin; output_y < output_y_end; ++output_y) {
                            uint32_t input_y = output_y + kernel_y - 1u;
                            const float* input_row =
                                input_channel_data + (size_t)((uint64_t)input_y * width);
                            float* output_row =
                                output_channel_data + (size_t)((uint64_t)output_y * width);
                            uint32_t output_x;
                            for (output_x = output_x_begin; output_x < output_x_end; ++output_x) {
                                uint32_t input_x = output_x + kernel_x - 1u;
                                output_row[output_x] += input_row[input_x] * weight;
                            }
                        }
                    }
                }
            }
        }
    }
}

void lw_scalar_conv2x2_unit_pad_end1_f32(const float* input, const float* weights,
                                         const float* bias, float* output,
                                         const int32_t input_dimensions[4],
                                         const int32_t output_dimensions[4]) {
    uint32_t height = (uint32_t)input_dimensions[2];
    uint32_t width = (uint32_t)input_dimensions[3];
    uint64_t channel_plane = (uint64_t)height * width;
    uint32_t batch;
    for (batch = 0u; batch < (uint32_t)input_dimensions[0]; ++batch) {
        const float* batch_input =
            input + (size_t)((uint64_t)batch * (uint32_t)input_dimensions[1] * channel_plane);
        float* batch_output =
            output + (size_t)((uint64_t)batch * (uint32_t)output_dimensions[1] * channel_plane);
        uint32_t output_channel;
        for (output_channel = 0u; output_channel < (uint32_t)output_dimensions[1];
             ++output_channel) {
            const float* output_channel_weights =
                weights + (size_t)((uint64_t)output_channel * (uint32_t)input_dimensions[1] * 4u);
            float* output_channel_data =
                batch_output + (size_t)((uint64_t)output_channel * channel_plane);
            float initial = bias == NULL ? 0.0f : bias[output_channel];
            uint64_t spatial;
            uint32_t input_channel;
            for (spatial = 0u; spatial < channel_plane; ++spatial) {
                output_channel_data[(size_t)spatial] = initial;
            }
            for (input_channel = 0u; input_channel < (uint32_t)input_dimensions[1];
                 ++input_channel) {
                const float* input_channel_data =
                    batch_input + (size_t)((uint64_t)input_channel * channel_plane);
                const float* weight_channel_data =
                    output_channel_weights + (size_t)((uint64_t)input_channel * 4u);
                uint32_t kernel_y;
                for (kernel_y = 0u; kernel_y < 2u; ++kernel_y) {
                    uint32_t output_y_end = height - kernel_y;
                    uint32_t kernel_x;
                    for (kernel_x = 0u; kernel_x < 2u; ++kernel_x) {
                        uint32_t output_x_end = width - kernel_x;
                        float weight = weight_channel_data[kernel_y * 2u + kernel_x];
                        uint32_t output_y;
                        for (output_y = 0u; output_y < output_y_end; ++output_y) {
                            const float* input_row =
                                input_channel_data +
                                (size_t)((uint64_t)(output_y + kernel_y) * width + kernel_x);
                            float* output_row =
                                output_channel_data + (size_t)((uint64_t)output_y * width);
                            uint32_t output_x;
                            for (output_x = 0u; output_x < output_x_end; ++output_x) {
                                output_row[output_x] += input_row[output_x] * weight;
                            }
                        }
                    }
                }
            }
        }
    }
}

void lw_scalar_conv3x3_stride2_pad1_f32(const float* input, const float* weights, const float* bias,
                                        float* output, const int32_t input_dimensions[4],
                                        const int32_t output_dimensions[4]) {
    uint64_t input_channel_plane =
        (uint64_t)(uint32_t)input_dimensions[2] * (uint32_t)input_dimensions[3];
    uint64_t output_channel_plane =
        (uint64_t)(uint32_t)output_dimensions[2] * (uint32_t)output_dimensions[3];
    uint32_t output_y_begin[3];
    uint32_t output_y_end[3];
    uint32_t output_x_begin[3];
    uint32_t output_x_end[3];
    uint32_t kernel_index;
    uint32_t batch;
    for (kernel_index = 0u; kernel_index < 3u; ++kernel_index) {
        output_y_begin[kernel_index] = 0u;
        output_y_end[kernel_index] = (uint32_t)output_dimensions[2];
        while (output_y_begin[kernel_index] < output_y_end[kernel_index] &&
               (int64_t)output_y_begin[kernel_index] * 2 - 1 + kernel_index < 0) {
            ++output_y_begin[kernel_index];
        }
        while (output_y_end[kernel_index] > output_y_begin[kernel_index] &&
               (int64_t)(output_y_end[kernel_index] - 1u) * 2 - 1 + kernel_index >=
                   input_dimensions[2]) {
            --output_y_end[kernel_index];
        }
        output_x_begin[kernel_index] = 0u;
        output_x_end[kernel_index] = (uint32_t)output_dimensions[3];
        while (output_x_begin[kernel_index] < output_x_end[kernel_index] &&
               (int64_t)output_x_begin[kernel_index] * 2 - 1 + kernel_index < 0) {
            ++output_x_begin[kernel_index];
        }
        while (output_x_end[kernel_index] > output_x_begin[kernel_index] &&
               (int64_t)(output_x_end[kernel_index] - 1u) * 2 - 1 + kernel_index >=
                   input_dimensions[3]) {
            --output_x_end[kernel_index];
        }
    }
    for (batch = 0u; batch < (uint32_t)input_dimensions[0]; ++batch) {
        const float* batch_input =
            input + (size_t)((uint64_t)batch * (uint32_t)input_dimensions[1] * input_channel_plane);
        float* batch_output = output + (size_t)((uint64_t)batch * (uint32_t)output_dimensions[1] *
                                                output_channel_plane);
        uint32_t output_channel;
        for (output_channel = 0u; output_channel < (uint32_t)output_dimensions[1];
             ++output_channel) {
            const float* output_channel_weights =
                weights + (size_t)((uint64_t)output_channel * (uint32_t)input_dimensions[1] * 9u);
            float* output_channel_data =
                batch_output + (size_t)((uint64_t)output_channel * output_channel_plane);
            float initial = bias == NULL ? 0.0f : bias[output_channel];
            uint64_t spatial;
            uint32_t input_channel;
            for (spatial = 0u; spatial < output_channel_plane; ++spatial) {
                output_channel_data[(size_t)spatial] = initial;
            }
            for (input_channel = 0u; input_channel < (uint32_t)input_dimensions[1];
                 ++input_channel) {
                const float* input_channel_data =
                    batch_input + (size_t)((uint64_t)input_channel * input_channel_plane);
                const float* weight_channel_data =
                    output_channel_weights + (size_t)((uint64_t)input_channel * 9u);
                uint32_t kernel_y;
                for (kernel_y = 0u; kernel_y < 3u; ++kernel_y) {
                    uint32_t kernel_x;
                    for (kernel_x = 0u; kernel_x < 3u; ++kernel_x) {
                        uint32_t output_y;
                        float weight = weight_channel_data[(size_t)kernel_y * 3u + kernel_x];
                        for (output_y = output_y_begin[kernel_y]; output_y < output_y_end[kernel_y];
                             ++output_y) {
                            uint32_t input_y = (uint32_t)((int64_t)output_y * 2 - 1 + kernel_y);
                            const float* input_row =
                                input_channel_data +
                                (size_t)((uint64_t)input_y * (uint32_t)input_dimensions[3]);
                            float* output_row =
                                output_channel_data +
                                (size_t)((uint64_t)output_y * (uint32_t)output_dimensions[3]);
                            uint32_t output_x;
                            for (output_x = output_x_begin[kernel_x];
                                 output_x < output_x_end[kernel_x]; ++output_x) {
                                uint32_t input_x = (uint32_t)((int64_t)output_x * 2 - 1 + kernel_x);
                                output_row[output_x] += input_row[input_x] * weight;
                            }
                        }
                    }
                }
            }
        }
    }
}

lw_status lw_scalar_conv2d_f32(const float* input, const float* weights, const float* bias,
                               uint32_t bias_count, float* output,
                               const int32_t input_dimensions[4],
                               const int32_t weight_dimensions[4],
                               const int32_t output_dimensions[4], const int32_t kernel[2],
                               const int32_t strides[2], const int32_t dilations[2],
                               const int32_t pads[4], uint32_t groups) {
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
    if (input == NULL || weights == NULL || output == NULL || input_dimensions == NULL ||
        weight_dimensions == NULL || output_dimensions == NULL || kernel == NULL ||
        strides == NULL || dilations == NULL || pads == NULL || output == input ||
        output == weights || (bias != NULL && output == bias)) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    if ((bias == NULL && bias_count != 0u) || (bias != NULL && bias_count == 0u) || groups == 0u) {
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
        (uint64_t)(uint32_t)weight_dimensions[1] * groups != (uint32_t)input_dimensions[1] ||
        (uint32_t)input_dimensions[1] % groups != 0u ||
        (uint32_t)output_dimensions[1] % groups != 0u || weight_dimensions[2] != kernel[0] ||
        weight_dimensions[3] != kernel[1] ||
        (bias != NULL && bias_count != (uint32_t)output_dimensions[1]) ||
        !spatial_output(input_dimensions[2], kernel[0], strides[0], dilations[0], pads[0], pads[2],
                        &expected_height) ||
        !spatial_output(input_dimensions[3], kernel[1], strides[1], dilations[1], pads[1], pads[3],
                        &expected_width) ||
        output_dimensions[2] != expected_height || output_dimensions[3] != expected_width) {
        return LW_STATUS_INVALID_SHAPE;
    }
    input_channels_per_group = (uint32_t)input_dimensions[1] / groups;
    output_channels_per_group = (uint32_t)output_dimensions[1] / groups;

    /* Match common PP-OCR convolution shapes before entering the general
     * nested loops. Every branch uses the same validated NCHW contract. */
    if (kernel[0] == 1 && kernel[1] == 1 && strides[0] == 1 && strides[1] == 1 &&
        dilations[0] == 1 && dilations[1] == 1 && pads[0] == 0 && pads[1] == 0 && pads[2] == 0 &&
        pads[3] == 0) {
        lw_simd_level simd_level = lw_detect_simd_level();
        if (simd_level >= LW_SIMD_LEVEL_AVX2) {
            lw_avx2_conv1x1_unit_f32(input, weights, bias, output, input_dimensions,
                                     output_dimensions, groups, input_channels_per_group,
                                     output_channels_per_group);
        } else if (simd_level >= LW_SIMD_LEVEL_SSE2) {
            lw_sse2_conv1x1_unit_f32(input, weights, bias, output, input_dimensions,
                                     output_dimensions, groups, input_channels_per_group,
                                     output_channels_per_group);
        } else {
            lw_scalar_conv1x1_unit_f32(input, weights, bias, output, input_dimensions,
                                       output_dimensions, groups, input_channels_per_group,
                                       output_channels_per_group);
        }
        return LW_STATUS_OK;
    }
    if (groups == (uint32_t)input_dimensions[1] && output_dimensions[1] == input_dimensions[1] &&
        weight_dimensions[1] == 1 && kernel[0] == 3 && kernel[1] == 3 && strides[0] == 1 &&
        strides[1] == 1 && dilations[0] == 1 && dilations[1] == 1 && pads[0] == 1 && pads[1] == 1 &&
        pads[2] == 1 && pads[3] == 1 && output_dimensions[2] == input_dimensions[2] &&
        output_dimensions[3] == input_dimensions[3]) {
        lw_simd_level simd_level = lw_detect_simd_level();
        if (simd_level >= LW_SIMD_LEVEL_AVX2) {
            lw_avx2_depthwise_conv3x3_unit_pad1_f32(input, weights, bias, output, input_dimensions);
        } else if (simd_level >= LW_SIMD_LEVEL_SSE2) {
            lw_sse2_depthwise_conv3x3_unit_pad1_f32(input, weights, bias, output, input_dimensions);
        } else {
            lw_scalar_depthwise_conv3x3_unit_pad1_f32(input, weights, bias, output,
                                                      input_dimensions);
        }
        return LW_STATUS_OK;
    }
    if (groups == (uint32_t)input_dimensions[1] && output_dimensions[1] == input_dimensions[1] &&
        weight_dimensions[1] == 1 && kernel[0] == 5 && kernel[1] == 5 && strides[0] == 1 &&
        strides[1] == 1 && dilations[0] == 1 && dilations[1] == 1 && pads[0] == 2 && pads[1] == 2 &&
        pads[2] == 2 && pads[3] == 2 && output_dimensions[2] == input_dimensions[2] &&
        output_dimensions[3] == input_dimensions[3]) {
        lw_simd_level simd_level = lw_detect_simd_level();
        if (simd_level >= LW_SIMD_LEVEL_AVX2) {
            lw_avx2_depthwise_conv5x5_unit_pad2_f32(input, weights, bias, output, input_dimensions);
        } else if (simd_level >= LW_SIMD_LEVEL_SSE2) {
            lw_sse2_depthwise_conv5x5_unit_pad2_f32(input, weights, bias, output, input_dimensions);
        } else {
            lw_scalar_depthwise_conv5x5_unit_pad2_f32(input, weights, bias, output,
                                                      input_dimensions);
        }
        return LW_STATUS_OK;
    }
    if (groups == 1u && kernel[0] == 3 && kernel[1] == 3 && strides[0] == 1 && strides[1] == 1 &&
        dilations[0] == 1 && dilations[1] == 1 && pads[0] == 1 && pads[1] == 1 && pads[2] == 1 &&
        pads[3] == 1 && output_dimensions[2] == input_dimensions[2] &&
        output_dimensions[3] == input_dimensions[3]) {
        lw_simd_level simd_level = lw_detect_simd_level();
        if (simd_level >= LW_SIMD_LEVEL_AVX2) {
            lw_avx2_conv3x3_unit_pad1_f32(input, weights, bias, output, input_dimensions,
                                          output_dimensions);
        } else if (simd_level >= LW_SIMD_LEVEL_SSE2) {
            lw_sse2_conv3x3_unit_pad1_f32(input, weights, bias, output, input_dimensions,
                                          output_dimensions);
        } else {
            lw_scalar_conv3x3_unit_pad1_f32(input, weights, bias, output, input_dimensions,
                                            output_dimensions);
        }
        return LW_STATUS_OK;
    }
    if (groups == 1u && kernel[0] == 2 && kernel[1] == 2 && strides[0] == 1 && strides[1] == 1 &&
        dilations[0] == 1 && dilations[1] == 1 && pads[0] == 0 && pads[1] == 0 && pads[2] == 1 &&
        pads[3] == 1 && output_dimensions[2] == input_dimensions[2] &&
        output_dimensions[3] == input_dimensions[3]) {
        lw_simd_level simd_level = lw_detect_simd_level();
        if (simd_level >= LW_SIMD_LEVEL_AVX2) {
            lw_avx2_conv2x2_unit_pad_end1_f32(input, weights, bias, output, input_dimensions,
                                              output_dimensions);
        } else if (simd_level >= LW_SIMD_LEVEL_SSE2) {
            lw_sse2_conv2x2_unit_pad_end1_f32(input, weights, bias, output, input_dimensions,
                                              output_dimensions);
        } else {
            lw_scalar_conv2x2_unit_pad_end1_f32(input, weights, bias, output, input_dimensions,
                                                output_dimensions);
        }
        return LW_STATUS_OK;
    }
    if (groups == 1u && kernel[0] == 3 && kernel[1] == 3 && strides[0] == 2 && strides[1] == 2 &&
        dilations[0] == 1 && dilations[1] == 1 && pads[0] == 1 && pads[1] == 1 && pads[2] == 1 &&
        pads[3] == 1) {
        lw_simd_level simd_level = lw_detect_simd_level();
        if (simd_level >= LW_SIMD_LEVEL_AVX2) {
            lw_avx2_conv3x3_stride2_pad1_f32(input, weights, bias, output, input_dimensions,
                                             output_dimensions);
        } else if (simd_level >= LW_SIMD_LEVEL_SSE2) {
            lw_sse2_conv3x3_stride2_pad1_f32(input, weights, bias, output, input_dimensions,
                                             output_dimensions);
        } else {
            lw_scalar_conv3x3_stride2_pad1_f32(input, weights, bias, output, input_dimensions,
                                               output_dimensions);
        }
        return LW_STATUS_OK;
    }
    /* General grouped-convolution fallback. Kernel bounds are trimmed once per
     * output coordinate, avoiding an inner-loop branch for padded pixels. */
    for (batch = 0u; batch < (uint32_t)input_dimensions[0]; ++batch) {
        for (group = 0u; group < groups; ++group) {
            uint32_t output_channel_in_group;
            uint32_t input_channel_start = group * input_channels_per_group;
            uint32_t output_channel_start = group * output_channels_per_group;
            uint64_t input_channel_plane =
                (uint64_t)(uint32_t)input_dimensions[2] * (uint32_t)input_dimensions[3];
            uint64_t output_channel_plane =
                (uint64_t)(uint32_t)expected_height * (uint32_t)expected_width;
            uint64_t kernel_channel_plane = (uint64_t)(uint32_t)kernel[0] * (uint32_t)kernel[1];
            const float* group_input =
                input +
                (size_t)(((uint64_t)batch * (uint32_t)input_dimensions[1] + input_channel_start) *
                         input_channel_plane);
            float* group_output =
                output +
                (size_t)(((uint64_t)batch * (uint32_t)output_dimensions[1] + output_channel_start) *
                         output_channel_plane);
            for (output_channel_in_group = 0u; output_channel_in_group < output_channels_per_group;
                 ++output_channel_in_group) {
                uint32_t output_channel = output_channel_start + output_channel_in_group;
                const float* output_channel_weights =
                    weights + (size_t)((uint64_t)output_channel * input_channels_per_group *
                                       kernel_channel_plane);
                float* output_channel_data =
                    group_output +
                    (size_t)((uint64_t)output_channel_in_group * output_channel_plane);
                uint32_t output_y;
                for (output_y = 0u; output_y < (uint32_t)expected_height; ++output_y) {
                    int64_t input_y_start = (int64_t)output_y * strides[0] - pads[0];
                    uint32_t kernel_y_begin = 0u;
                    uint32_t kernel_y_end = (uint32_t)kernel[0];
                    uint32_t output_x;
                    while (kernel_y_begin < kernel_y_end &&
                           input_y_start + (int64_t)kernel_y_begin * dilations[0] < 0) {
                        ++kernel_y_begin;
                    }
                    while (kernel_y_end > kernel_y_begin &&
                           input_y_start + (int64_t)(kernel_y_end - 1u) * dilations[0] >=
                               input_dimensions[2]) {
                        --kernel_y_end;
                    }
                    for (output_x = 0u; output_x < (uint32_t)expected_width; ++output_x) {
                        int64_t input_x_start = (int64_t)output_x * strides[1] - pads[1];
                        uint32_t kernel_x_begin = 0u;
                        uint32_t kernel_x_end = (uint32_t)kernel[1];
                        float sum = bias == NULL ? 0.0f : bias[output_channel];
                        uint32_t input_channel_in_group;
                        while (kernel_x_begin < kernel_x_end &&
                               input_x_start + (int64_t)kernel_x_begin * dilations[1] < 0) {
                            ++kernel_x_begin;
                        }
                        while (kernel_x_end > kernel_x_begin &&
                               input_x_start + (int64_t)(kernel_x_end - 1u) * dilations[1] >=
                                   input_dimensions[3]) {
                            --kernel_x_end;
                        }
                        for (input_channel_in_group = 0u;
                             input_channel_in_group < input_channels_per_group;
                             ++input_channel_in_group) {
                            const float* input_channel_data =
                                group_input +
                                (size_t)((uint64_t)input_channel_in_group * input_channel_plane);
                            const float* weight_channel_data =
                                output_channel_weights +
                                (size_t)((uint64_t)input_channel_in_group * kernel_channel_plane);
                            uint32_t kernel_y;
                            for (kernel_y = kernel_y_begin; kernel_y < kernel_y_end; ++kernel_y) {
                                uint32_t input_y =
                                    (uint32_t)(input_y_start + (int64_t)kernel_y * dilations[0]);
                                const float* input_row =
                                    input_channel_data +
                                    (size_t)((uint64_t)input_y * (uint32_t)input_dimensions[3]);
                                const float* weight_row =
                                    weight_channel_data +
                                    (size_t)((uint64_t)kernel_y * (uint32_t)kernel[1]);
                                uint32_t kernel_x;
                                for (kernel_x = kernel_x_begin; kernel_x < kernel_x_end;
                                     ++kernel_x) {
                                    uint32_t input_x = (uint32_t)(input_x_start +
                                                                  (int64_t)kernel_x * dilations[1]);
                                    sum += input_row[(size_t)input_x] * weight_row[kernel_x];
                                }
                            }
                        }
                        output_channel_data[(size_t)((uint64_t)output_y * (uint32_t)expected_width +
                                                     output_x)] = sum;
                    }
                }
            }
        }
    }
    return LW_STATUS_OK;
}

static void conv_transpose2x2_stride2_f32(const float* input, const float* weights,
                                          const float* bias, float* output,
                                          const int32_t input_dimensions[4],
                                          const int32_t output_dimensions[4]) {
    uint32_t input_channels = (uint32_t)input_dimensions[1];
    uint32_t output_channels = (uint32_t)output_dimensions[1];
    uint32_t input_height = (uint32_t)input_dimensions[2];
    uint32_t input_width = (uint32_t)input_dimensions[3];
    uint32_t output_width = (uint32_t)output_dimensions[3];
    uint64_t input_plane = (uint64_t)input_height * input_width;
    uint64_t output_plane = (uint64_t)(uint32_t)output_dimensions[2] * output_width;
    uint32_t batch;
    for (batch = 0u; batch < (uint32_t)input_dimensions[0]; ++batch) {
        const float* batch_input = input + (size_t)((uint64_t)batch * input_channels * input_plane);
        float* batch_output = output + (size_t)((uint64_t)batch * output_channels * output_plane);
        uint32_t output_channel;
        for (output_channel = 0u; output_channel < output_channels; ++output_channel) {
            float* output_channel_data =
                batch_output + (size_t)((uint64_t)output_channel * output_plane);
            float initial = bias == NULL ? 0.0f : bias[output_channel];
            uint64_t output_index;
            uint32_t input_channel;
            for (output_index = 0u; output_index < output_plane; ++output_index) {
                output_channel_data[(size_t)output_index] = initial;
            }
            for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
                const float* input_channel_data =
                    batch_input + (size_t)((uint64_t)input_channel * input_plane);
                const float* channel_weights =
                    weights +
                    (size_t)(((uint64_t)input_channel * output_channels + output_channel) * 4u);
                uint32_t input_y;
                for (input_y = 0u; input_y < input_height; ++input_y) {
                    const float* input_row =
                        input_channel_data + (size_t)((uint64_t)input_y * input_width);
                    float* output_row0 =
                        output_channel_data + (size_t)((uint64_t)(input_y * 2u) * output_width);
                    float* output_row1 = output_row0 + output_width;
                    uint32_t input_x;
                    for (input_x = 0u; input_x < input_width; ++input_x) {
                        float value = input_row[input_x];
                        uint32_t output_x = input_x * 2u;
                        output_row0[output_x] += value * channel_weights[0];
                        output_row0[output_x + 1u] += value * channel_weights[1];
                        output_row1[output_x] += value * channel_weights[2];
                        output_row1[output_x + 1u] += value * channel_weights[3];
                    }
                }
            }
        }
    }
}

lw_status lw_scalar_conv_transpose2d_f32(
    const float* input, const float* weights, const float* bias, uint32_t bias_count, float* output,
    const int32_t input_dimensions[4], const int32_t weight_dimensions[4],
    const int32_t output_dimensions[4], const int32_t kernel[2], const int32_t strides[2],
    const int32_t dilations[2], const int32_t pads[4], uint32_t groups) {
    uint64_t input_count;
    uint64_t weight_count;
    uint64_t output_count;
    int32_t expected_height;
    int32_t expected_width;
    uint32_t input_channels_per_group;
    uint32_t output_channels_per_group;
    uint32_t batch;
    lw_status status;
    if (input == NULL || weights == NULL || output == NULL || input == output ||
        weights == output || (bias != NULL && bias == output) || input_dimensions == NULL ||
        weight_dimensions == NULL || output_dimensions == NULL || kernel == NULL ||
        strides == NULL || dilations == NULL || pads == NULL || groups == 0u) {
        return LW_STATUS_INVALID_ARGUMENT;
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
    if (groups > (uint32_t)input_dimensions[1] || (uint32_t)input_dimensions[1] % groups != 0u ||
        weight_dimensions[0] != input_dimensions[1] || weight_dimensions[2] != kernel[0] ||
        weight_dimensions[3] != kernel[1] || output_dimensions[0] != input_dimensions[0] ||
        (uint64_t)(uint32_t)weight_dimensions[1] * groups != (uint32_t)output_dimensions[1] ||
        (bias == NULL ? bias_count != 0u : bias_count != (uint32_t)output_dimensions[1]) ||
        !transposed_spatial_output(input_dimensions[2], kernel[0], strides[0], dilations[0],
                                   pads[0], pads[2], &expected_height) ||
        !transposed_spatial_output(input_dimensions[3], kernel[1], strides[1], dilations[1],
                                   pads[1], pads[3], &expected_width) ||
        output_dimensions[2] != expected_height || output_dimensions[3] != expected_width) {
        return LW_STATUS_INVALID_SHAPE;
    }
    input_channels_per_group = (uint32_t)input_dimensions[1] / groups;
    output_channels_per_group = (uint32_t)weight_dimensions[1];
    if (groups == 1u && kernel[0] == 2 && kernel[1] == 2 && strides[0] == 2 && strides[1] == 2 &&
        dilations[0] == 1 && dilations[1] == 1 && pads[0] == 0 && pads[1] == 0 && pads[2] == 0 &&
        pads[3] == 0) {
        lw_simd_level simd_level = lw_detect_simd_level();
        if (simd_level >= LW_SIMD_LEVEL_AVX2) {
            lw_avx2_conv_transpose2x2_stride2_f32(input, weights, bias, output, input_dimensions,
                                                   output_dimensions);
        } else if (simd_level >= LW_SIMD_LEVEL_SSE2) {
            lw_sse2_conv_transpose2x2_stride2_f32(input, weights, bias, output, input_dimensions,
                                                   output_dimensions);
        } else {
            conv_transpose2x2_stride2_f32(input, weights, bias, output, input_dimensions,
                                          output_dimensions);
        }
        return LW_STATUS_OK;
    }
    for (batch = 0u; batch < (uint32_t)output_dimensions[0]; ++batch) {
        uint32_t output_channel;
        for (output_channel = 0u; output_channel < (uint32_t)output_dimensions[1];
             ++output_channel) {
            float initial = bias == NULL ? 0.0f : bias[output_channel];
            uint64_t plane = (uint64_t)(uint32_t)expected_height * (uint32_t)expected_width;
            uint64_t base =
                ((uint64_t)batch * (uint32_t)output_dimensions[1] + output_channel) * plane;
            uint64_t index;
            for (index = 0u; index < plane; ++index) {
                output[(size_t)(base + index)] = initial;
            }
        }
    }
    for (batch = 0u; batch < (uint32_t)input_dimensions[0]; ++batch) {
        uint32_t group;
        for (group = 0u; group < groups; ++group) {
            uint32_t input_channel_in_group;
            for (input_channel_in_group = 0u; input_channel_in_group < input_channels_per_group;
                 ++input_channel_in_group) {
                uint32_t input_channel = group * input_channels_per_group + input_channel_in_group;
                uint32_t input_y;
                for (input_y = 0u; input_y < (uint32_t)input_dimensions[2]; ++input_y) {
                    uint32_t input_x;
                    for (input_x = 0u; input_x < (uint32_t)input_dimensions[3]; ++input_x) {
                        uint64_t input_offset =
                            (((uint64_t)batch * (uint32_t)input_dimensions[1] + input_channel) *
                                 (uint32_t)input_dimensions[2] +
                             input_y) *
                                (uint32_t)input_dimensions[3] +
                            input_x;
                        float input_value = input[(size_t)input_offset];
                        uint32_t output_channel_in_group;
                        for (output_channel_in_group = 0u;
                             output_channel_in_group < output_channels_per_group;
                             ++output_channel_in_group) {
                            uint32_t output_channel =
                                group * output_channels_per_group + output_channel_in_group;
                            uint32_t kernel_y;
                            for (kernel_y = 0u; kernel_y < (uint32_t)kernel[0]; ++kernel_y) {
                                int64_t output_y = (int64_t)input_y * strides[0] - pads[0] +
                                                   (int64_t)kernel_y * dilations[0];
                                uint32_t kernel_x;
                                if (output_y < 0 || output_y >= expected_height) {
                                    continue;
                                }
                                for (kernel_x = 0u; kernel_x < (uint32_t)kernel[1]; ++kernel_x) {
                                    int64_t output_x = (int64_t)input_x * strides[1] - pads[1] +
                                                       (int64_t)kernel_x * dilations[1];
                                    uint64_t weight_offset;
                                    uint64_t output_offset;
                                    if (output_x < 0 || output_x >= expected_width) {
                                        continue;
                                    }
                                    weight_offset =
                                        (((uint64_t)input_channel * output_channels_per_group +
                                          output_channel_in_group) *
                                             (uint32_t)kernel[0] +
                                         kernel_y) *
                                            (uint32_t)kernel[1] +
                                        kernel_x;
                                    output_offset =
                                        (((uint64_t)batch * (uint32_t)output_dimensions[1] +
                                          output_channel) *
                                             (uint32_t)expected_height +
                                         (uint32_t)output_y) *
                                            (uint32_t)expected_width +
                                        (uint32_t)output_x;
                                    output[(size_t)output_offset] +=
                                        input_value * weights[(size_t)weight_offset];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (void)output_count;
    return LW_STATUS_OK;
}

lw_status lw_scalar_batch_normalization_f32(const float* input, const float* scale,
                                            const float* bias, const float* mean,
                                            const float* variance, uint32_t parameter_count,
                                            float epsilon, float* output, uint32_t rank,
                                            const int32_t* dimensions) {
    uint64_t element_count;
    uint64_t spatial_count = 1u;
    uint32_t channel_count;
    uint32_t channel;
    lw_status status;
    if (input == NULL || scale == NULL || bias == NULL || mean == NULL || variance == NULL ||
        output == NULL || output == scale || output == bias || output == mean ||
        output == variance) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    if (rank < 2u || rank > LW_MAX_DIMS || dimensions == NULL || !isfinite(epsilon) ||
        epsilon <= 0.0f) {
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
        if (!isfinite(scale[channel]) || !isfinite(bias[channel]) || !isfinite(mean[channel]) ||
            !isfinite(variance[channel]) || !isfinite(denominator) || denominator <= 0.0f) {
            return LW_STATUS_INVALID_ARGUMENT;
        }
    }
    for (channel = 0u; channel < channel_count; ++channel) {
        float factor = scale[channel] / sqrtf(variance[channel] + epsilon);
        uint32_t batch;
        for (batch = 0u; batch < (uint32_t)dimensions[0]; ++batch) {
            uint64_t offset = ((uint64_t)batch * channel_count + channel) * spatial_count;
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
