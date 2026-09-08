#include "scalar_kernels.h"
#include "cpu_features.h"
#include "packed_conv_internal.h"
#include "simd_kernels.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/*
 * SIMD implementations are allowed to accumulate in a different order than
 * the scalar reference.  On ARM64 this can produce a few ULPs of expected
 * floating-point variation even when the kernel is correct.  Keep the driver
 * strict enough to catch real errors while matching the Python reference
 * test's rtol=2e-5 / atol=3e-6 contract.
 */
static int lw_test_float_buffer_compare(const void* left_bytes, const void* right_bytes,
                                        size_t byte_count) {
    const float* left = (const float*)left_bytes;
    const float* right = (const float*)right_bytes;
    const size_t count = byte_count / sizeof(float);
    size_t index;
    for (index = 0u; index < count; ++index) {
        const float left_value = left[index];
        const float right_value = right[index];
        const float difference = left_value >= right_value ? left_value - right_value
                                                            : right_value - left_value;
        const float left_abs = left_value >= 0.0f ? left_value : -left_value;
        const float right_abs = right_value >= 0.0f ? right_value : -right_value;
        const float scale = left_abs >= right_abs ? left_abs : right_abs;
        if (!(difference <= 3.0e-6f + 2.0e-5f * scale)) {
            return 1;
        }
    }
    return 0;
}

#define memcmp(left, right, byte_count) \
    lw_test_float_buffer_compare((left), (right), (byte_count))

static void print_values(const char* name, const float* values, uint64_t count) {
    uint64_t index;
    printf("%s %" PRIu64, name, count);
    for (index = 0u; index < count; ++index) {
        printf(" %.9g", (double)values[index]);
    }
    putchar('\n');
}

static int expect_status(const char* name, lw_status actual, lw_status expected) {
    if (actual != expected) {
        fprintf(stderr, "%s: expected %s, got %s\n", name, lw_status_string(expected),
                lw_status_string(actual));
        return 0;
    }
    return 1;
}

static int is_32_bit_process(void) {
    volatile size_t pointer_size = sizeof(size_t);
    return pointer_size == 4u;
}

static void fill_values(float* values, uint32_t count, uint32_t multiplier, uint32_t modulus,
                        int32_t offset, float divisor) {
    uint32_t index;
    for (index = 0u; index < count; ++index) {
        values[index] = (float)((int32_t)((index * multiplier) % modulus) - offset) / divisor;
    }
}

static void reference_conv7x7(const float* input, const float* weights, const float* bias,
                              float* output, const int32_t input_dimensions[4],
                              const int32_t output_dimensions[4]) {
    const uint32_t input_channels = (uint32_t)input_dimensions[1];
    const uint32_t input_height = (uint32_t)input_dimensions[2];
    const uint32_t input_width = (uint32_t)input_dimensions[3];
    const uint32_t output_channels = (uint32_t)output_dimensions[1];
    const uint32_t output_height = (uint32_t)output_dimensions[2];
    const uint32_t output_width = (uint32_t)output_dimensions[3];
    const uint64_t input_plane = (uint64_t)input_height * input_width;
    const uint64_t output_plane = (uint64_t)output_height * output_width;
    const uint64_t weights_per_output = (uint64_t)input_channels * 49u;
    uint32_t batch;
    for (batch = 0u; batch < (uint32_t)input_dimensions[0]; ++batch) {
        uint32_t output_channel;
        for (output_channel = 0u; output_channel < output_channels; ++output_channel) {
            const float* output_weights =
                weights + (size_t)((uint64_t)output_channel * weights_per_output);
            float* output_data = output +
                                 (size_t)(((uint64_t)batch * output_channels + output_channel) *
                                          output_plane);
            uint32_t output_y;
            for (output_y = 0u; output_y < output_height; ++output_y) {
                uint32_t output_x;
                for (output_x = 0u; output_x < output_width; ++output_x) {
                    float value = bias == NULL ? 0.0f : bias[output_channel];
                    uint32_t input_channel;
                    for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
                        const float* input_data =
                            input + (size_t)(((uint64_t)batch * input_channels + input_channel) *
                                             input_plane);
                        const float* input_weights =
                            output_weights + (size_t)((uint64_t)input_channel * 49u);
                        uint32_t kernel_y;
                        for (kernel_y = 0u; kernel_y < 7u; ++kernel_y) {
                            const int32_t input_y = (int32_t)output_y + (int32_t)kernel_y - 3;
                            uint32_t kernel_x;
                            if (input_y < 0 || input_y >= (int32_t)input_height) {
                                continue;
                            }
                            for (kernel_x = 0u; kernel_x < 7u; ++kernel_x) {
                                const int32_t input_x =
                                    (int32_t)output_x + (int32_t)kernel_x - 3;
                                if (input_x >= 0 && input_x < (int32_t)input_width) {
                                    value += input_data[(size_t)((uint32_t)input_y * input_width +
                                                                 (uint32_t)input_x)] *
                                             input_weights[kernel_y * 7u + kernel_x];
                                }
                            }
                        }
                    }
                    output_data[(size_t)output_y * output_width + output_x] = value;
                }
            }
        }
    }
}

static void reference_conv5x5(const float* input, const float* weights, const float* bias,
                              float* output, const int32_t input_dimensions[4],
                              const int32_t output_dimensions[4]) {
    const uint32_t input_channels = (uint32_t)input_dimensions[1];
    const uint32_t input_height = (uint32_t)input_dimensions[2];
    const uint32_t input_width = (uint32_t)input_dimensions[3];
    const uint32_t output_channels = (uint32_t)output_dimensions[1];
    const uint32_t output_height = (uint32_t)output_dimensions[2];
    const uint32_t output_width = (uint32_t)output_dimensions[3];
    const uint64_t input_plane = (uint64_t)input_height * input_width;
    const uint64_t output_plane = (uint64_t)output_height * output_width;
    const uint64_t weights_per_output = (uint64_t)input_channels * 25u;
    uint32_t batch;
    for (batch = 0u; batch < (uint32_t)input_dimensions[0]; ++batch) {
        uint32_t output_channel;
        for (output_channel = 0u; output_channel < output_channels; ++output_channel) {
            const float* output_weights =
                weights + (size_t)((uint64_t)output_channel * weights_per_output);
            float* output_data = output +
                                 (size_t)(((uint64_t)batch * output_channels + output_channel) *
                                          output_plane);
            uint32_t output_y;
            for (output_y = 0u; output_y < output_height; ++output_y) {
                uint32_t output_x;
                for (output_x = 0u; output_x < output_width; ++output_x) {
                    float value = bias == NULL ? 0.0f : bias[output_channel];
                    uint32_t input_channel;
                    for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
                        const float* input_data =
                            input + (size_t)(((uint64_t)batch * input_channels + input_channel) *
                                             input_plane);
                        const float* input_weights =
                            output_weights + (size_t)((uint64_t)input_channel * 25u);
                        uint32_t kernel_y;
                        for (kernel_y = 0u; kernel_y < 5u; ++kernel_y) {
                            const int32_t input_y = (int32_t)output_y + (int32_t)kernel_y - 2;
                            uint32_t kernel_x;
                            if (input_y < 0 || input_y >= (int32_t)input_height) continue;
                            for (kernel_x = 0u; kernel_x < 5u; ++kernel_x) {
                                const int32_t input_x =
                                    (int32_t)output_x + (int32_t)kernel_x - 2;
                                if (input_x >= 0 && input_x < (int32_t)input_width) {
                                    value += input_data[(size_t)((uint32_t)input_y * input_width +
                                                                 (uint32_t)input_x)] *
                                             input_weights[kernel_y * 5u + kernel_x];
                                }
                            }
                        }
                    }
                    output_data[(size_t)output_y * output_width + output_x] = value;
                }
            }
        }
    }
}

static void reference_depthwise9x9(const float* input, const float* weights, const float* bias,
                                   float* output, const int32_t dimensions[4]) {
    const uint32_t height = (uint32_t)dimensions[2];
    const uint32_t width = (uint32_t)dimensions[3];
    const uint64_t channel_plane = (uint64_t)height * width;
    uint32_t batch;
    for (batch = 0u; batch < (uint32_t)dimensions[0]; ++batch) {
        uint32_t channel;
        for (channel = 0u; channel < (uint32_t)dimensions[1]; ++channel) {
            const uint64_t channel_offset =
                ((uint64_t)batch * (uint32_t)dimensions[1] + channel) * channel_plane;
            const float* input_channel = input + (size_t)channel_offset;
            const float* channel_weights = weights + (size_t)channel * 81u;
            float* output_channel = output + (size_t)channel_offset;
            const float initial = bias == NULL ? 0.0f : bias[channel];
            uint64_t spatial;
            uint32_t kernel_y;
            for (spatial = 0u; spatial < channel_plane; ++spatial) {
                output_channel[(size_t)spatial] = initial;
            }
            for (kernel_y = 0u; kernel_y < 9u; ++kernel_y) {
                const uint32_t output_y_begin = kernel_y < 4u ? 4u - kernel_y : 0u;
                const uint32_t output_y_trim = kernel_y > 4u ? kernel_y - 4u : 0u;
                const uint32_t output_y_end =
                    output_y_trim < height ? height - output_y_trim : 0u;
                uint32_t kernel_x;
                for (kernel_x = 0u; kernel_x < 9u; ++kernel_x) {
                    const uint32_t output_x_begin = kernel_x < 4u ? 4u - kernel_x : 0u;
                    const uint32_t output_x_trim = kernel_x > 4u ? kernel_x - 4u : 0u;
                    const uint32_t output_x_end =
                        output_x_trim < width ? width - output_x_trim : 0u;
                    const float weight = channel_weights[kernel_y * 9u + kernel_x];
                    uint32_t output_y;
                    for (output_y = output_y_begin; output_y < output_y_end; ++output_y) {
                        const uint32_t input_y = output_y + kernel_y - 4u;
                        const float* input_row =
                            input_channel + (size_t)((uint64_t)input_y * width);
                        float* output_row =
                            output_channel + (size_t)((uint64_t)output_y * width);
                        uint32_t output_x;
                        for (output_x = output_x_begin; output_x < output_x_end; ++output_x) {
                            const uint32_t input_x = output_x + kernel_x - 4u;
                            output_row[output_x] += input_row[input_x] * weight;
                        }
                    }
                }
            }
        }
    }
}

static void reference_conv_axis(const float* input, const float* weights, const float* bias,
                                float* output, const int32_t input_dimensions[4],
                                const int32_t output_dimensions[4], uint32_t kernel_height,
                                uint32_t kernel_width, uint32_t pad_y, uint32_t pad_x) {
    const uint32_t input_channels = (uint32_t)input_dimensions[1];
    const uint32_t input_height = (uint32_t)input_dimensions[2];
    const uint32_t input_width = (uint32_t)input_dimensions[3];
    const uint32_t output_channels = (uint32_t)output_dimensions[1];
    const uint32_t output_height = (uint32_t)output_dimensions[2];
    const uint32_t output_width = (uint32_t)output_dimensions[3];
    const uint64_t input_plane = (uint64_t)input_height * input_width;
    const uint64_t output_plane = (uint64_t)output_height * output_width;
    const uint64_t weights_per_output = (uint64_t)input_channels * kernel_height * kernel_width;
    uint32_t batch;
    for (batch = 0u; batch < (uint32_t)input_dimensions[0]; ++batch) {
        uint32_t output_channel;
        for (output_channel = 0u; output_channel < output_channels; ++output_channel) {
            const float* output_weights =
                weights + (size_t)((uint64_t)output_channel * weights_per_output);
            float* output_data = output +
                                 (size_t)(((uint64_t)batch * output_channels + output_channel) *
                                          output_plane);
            uint32_t output_y;
            for (output_y = 0u; output_y < output_height; ++output_y) {
                uint32_t output_x;
                for (output_x = 0u; output_x < output_width; ++output_x) {
                    float value = bias == NULL ? 0.0f : bias[output_channel];
                    uint32_t input_channel;
                    for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
                        const float* input_data =
                            input + (size_t)(((uint64_t)batch * input_channels + input_channel) *
                                             input_plane);
                        const float* input_weights = output_weights +
                                                     (size_t)((uint64_t)input_channel *
                                                              kernel_height * kernel_width);
                        uint32_t kernel_y;
                        for (kernel_y = 0u; kernel_y < kernel_height; ++kernel_y) {
                            const int32_t input_y =
                                (int32_t)output_y + (int32_t)kernel_y - (int32_t)pad_y;
                            uint32_t kernel_x;
                            if (input_y < 0 || input_y >= (int32_t)input_height) continue;
                            for (kernel_x = 0u; kernel_x < kernel_width; ++kernel_x) {
                                const int32_t input_x =
                                    (int32_t)output_x + (int32_t)kernel_x - (int32_t)pad_x;
                                if (input_x >= 0 && input_x < (int32_t)input_width) {
                                    value += input_data[(size_t)((uint32_t)input_y * input_width +
                                                                 (uint32_t)input_x)] *
                                             input_weights[kernel_y * kernel_width + kernel_x];
                                }
                            }
                        }
                    }
                    output_data[(size_t)output_y * output_width + output_x] = value;
                }
            }
        }
    }
}

int main(void) {
    const int32_t normal_input_dimensions[4] = {2, 2, 4, 5};
    const int32_t normal_weight_dimensions[4] = {3, 2, 3, 3};
    const int32_t invalid_weight_dimensions[4] = {3, 1, 3, 3};
    const int32_t normal_output_dimensions[4] = {2, 3, 2, 3};
    const int32_t stride2_input_dimensions[4] = {1, 2, 5, 18};
    const int32_t stride2_weight_dimensions[4] = {3, 2, 3, 3};
    const int32_t stride2_output_dimensions[4] = {1, 3, 3, 9};
    const int32_t stride2_four_output_dimensions[4] = {1, 4, 3, 9};
    const int32_t unit_conv_input_dimensions[4] = {1, 3, 5, 19};
    const int32_t unit_conv_weight_dimensions[4] = {2, 3, 3, 3};
    const int32_t unit_conv_output_dimensions[4] = {1, 2, 5, 19};
    const int32_t unit_conv_four_output_dimensions[4] = {1, 4, 5, 19};
    const int32_t unit_conv2x2_weight_dimensions[4] = {2, 3, 2, 2};
    const int32_t unit_conv2x2_four_weight_dimensions[4] = {4, 3, 2, 2};
    const int32_t unit_conv2x2_kernel[2] = {2, 2};
    const int32_t conv7x7_input_dimensions[4] = {1, 2, 9, 19};
    const int32_t conv7x7_weight_dimensions[4] = {4, 2, 7, 7};
    const int32_t conv7x7_output_dimensions[4] = {1, 4, 9, 19};
    const int32_t conv7x7_kernel[2] = {7, 7};
    const int32_t conv7x7_pads[4] = {3, 3, 3, 3};
    const int32_t conv5x5_weight_dimensions[4] = {4, 2, 5, 5};
    const int32_t conv5x5_kernel[2] = {5, 5};
    const int32_t conv5x5_pads[4] = {2, 2, 2, 2};
    const int32_t conv7x1_weight_dimensions[4] = {4, 2, 7, 1};
    const int32_t conv7x1_kernel[2] = {7, 1};
    const int32_t conv7x1_pads[4] = {3, 0, 3, 0};
    const int32_t conv1x7_weight_dimensions[4] = {4, 2, 1, 7};
    const int32_t conv1x7_kernel[2] = {1, 7};
    const int32_t conv1x7_pads[4] = {0, 3, 0, 3};
    const int32_t conv5x1_weight_dimensions[4] = {4, 2, 5, 1};
    const int32_t conv5x1_kernel[2] = {5, 1};
    const int32_t conv5x1_pads[4] = {2, 0, 2, 0};
    const int32_t conv1x5_weight_dimensions[4] = {4, 2, 1, 5};
    const int32_t conv1x5_kernel[2] = {1, 5};
    const int32_t conv1x5_pads[4] = {0, 2, 0, 2};
    const int32_t unit_conv2x2_pads[4] = {0, 0, 1, 1};
    const int32_t invalid_output_dimensions[4] = {2, 3, 2, 2};
    const int32_t normal_kernel[2] = {3, 3};
    const int32_t normal_strides[2] = {2, 2};
    const int32_t unit_dilations[2] = {1, 1};
    const int32_t normal_pads[4] = {1, 1, 1, 1};
    const int32_t grouped_input_dimensions[4] = {1, 4, 4, 4};
    const int32_t grouped_weight_dimensions[4] = {6, 2, 3, 3};
    const int32_t grouped_output_dimensions[4] = {1, 6, 4, 4};
    const int32_t unit_strides[2] = {1, 1};
    const int32_t depthwise_input_dimensions[4] = {1, 3, 4, 5};
    const int32_t depthwise_weight_dimensions[4] = {3, 1, 3, 2};
    const int32_t depthwise_output_dimensions[4] = {1, 3, 4, 5};
    const int32_t depthwise_kernel[2] = {3, 2};
    const int32_t depthwise_dilations[2] = {1, 2};
    const int32_t unit_depthwise_dimensions[4] = {1, 2, 4, 10};
    const int32_t unit_depthwise_weight_dimensions[4] = {2, 1, 3, 3};
    const int32_t stride2x1_depthwise_input_dimensions[4] = {1, 2, 5, 19};
    const int32_t stride2x1_depthwise_output_dimensions[4] = {1, 2, 3, 19};
    const int32_t stride2x1_depthwise_strides[2] = {2, 1};
    const int32_t unit_depthwise5x5_dimensions[4] = {1, 2, 6, 19};
    const int32_t unit_depthwise5x5_weight_dimensions[4] = {2, 1, 5, 5};
    const int32_t depthwise5x5_kernel[2] = {5, 5};
    const int32_t pad2[4] = {2, 2, 2, 2};
    const int32_t unit_depthwise9x9_dimensions[4] = {1, 2, 10, 19};
    const int32_t unit_depthwise9x9_weight_dimensions[4] = {2, 1, 9, 9};
    const int32_t depthwise9x9_kernel[2] = {9, 9};
    const int32_t pad4[4] = {4, 4, 4, 4};
    const int32_t asymmetric_input_dimensions[4] = {1, 1, 2, 3};
    const int32_t asymmetric_weight_dimensions[4] = {1, 1, 2, 2};
    const int32_t asymmetric_output_dimensions[4] = {1, 1, 3, 3};
    const int32_t asymmetric_kernel[2] = {2, 2};
    const int32_t asymmetric_strides[2] = {1, 2};
    const int32_t asymmetric_dilations[2] = {2, 1};
    const int32_t asymmetric_pads[4] = {2, 1, 1, 2};
    const int32_t pointwise_input_dimensions[4] = {2, 4, 2, 5};
    const int32_t pointwise_weight_dimensions[4] = {6, 2, 1, 1};
    const int32_t pointwise_output_dimensions[4] = {2, 6, 2, 5};
    const int32_t packed_pointwise_input_dimensions[4] = {2, 5, 3, 7};
    const int32_t packed_pointwise_output_dimensions[4] = {2, 7, 3, 7};
    const int32_t point_kernel[2] = {1, 1};
    const int32_t no_pads[4] = {0, 0, 0, 0};
    const int32_t batch_norm_dimensions[4] = {2, 3, 2, 2};
    const int32_t transpose_conv_input_dimensions[4] = {1, 2, 2, 2};
    const int32_t transpose_conv_weight_dimensions[4] = {2, 1, 2, 2};
    const int32_t transpose_conv_output_dimensions[4] = {1, 1, 4, 4};
    const int32_t transpose_conv_kernel[2] = {2, 2};
    const int32_t transpose_conv_strides[2] = {2, 2};
    const int32_t transpose_conv_simd_input_dimensions[4] = {1, 2, 3, 8};
    const int32_t transpose_conv_simd_weight_dimensions[4] = {2, 3, 2, 2};
    const int32_t transpose_conv_simd_output_dimensions[4] = {1, 3, 6, 16};
    const float normal_bias[3] = {0.25f, -0.5f, 1.0f};
    const float stride2_bias[3] = {-0.125f, 0.625f, -0.875f};
    const float stride2_four_bias[4] = {-0.125f, 0.625f, -0.875f, 0.375f};
    const float unit_conv_bias[2] = {0.375f, -0.625f};
    const float unit_conv_four_bias[4] = {0.375f, -0.625f, 0.125f, -0.875f};
    const float conv7x7_bias[4] = {0.375f, -0.625f, 0.125f, -0.875f};
    const float unit_depthwise_bias[2] = {0.375f, -0.625f};
    const float pointwise_bias[6] = {0.25f, -0.5f, 1.0f, -1.25f, 0.75f, 0.5f};
    const float packed_pointwise_bias[7] = {0.25f, -0.5f, 1.0f, -1.25f, 0.75f, 0.5f, -0.125f};
    const float batch_norm_scale[3] = {1.5f, -0.75f, 0.25f};
    const float batch_norm_bias[3] = {0.1f, 0.5f, -1.0f};
    const float batch_norm_mean[3] = {-0.25f, 1.0f, 0.5f};
    const float batch_norm_variance[3] = {0.5f, 2.0f, 0.25f};
    const float invalid_variance[3] = {0.5f, -1.0f, 0.25f};
    const float transpose_conv_bias[1] = {0.125f};
    const float transpose_conv_simd_bias[3] = {0.125f, -0.25f, 0.5f};
    float normal_input[80];
    float normal_weights[54];
    float normal_output[36];
    float stride2_input[180];
    float stride2_weights[54];
    float stride2_output[81];
    float stride2_dispatched_output[81];
    float stride2_simd_output[81];
    float stride2_four_weights[72];
    float stride2_four_output[108];
    float stride2_four_simd_output[108];
    float unit_conv_input[285];
    float unit_conv_weights[54];
    float unit_conv_output[190];
    float unit_conv_dispatched_output[190];
    float unit_conv_simd_output[190];
    float unit_conv_four_weights[108];
    float unit_conv_four_output[380];
    float unit_conv_four_simd_output[380];
    float unit_conv2x2_weights[24];
    float unit_conv2x2_output[190];
    float unit_conv2x2_dispatched_output[190];
    float unit_conv2x2_simd_output[190];
    float unit_conv2x2_four_weights[48];
    float unit_conv2x2_four_output[380];
    float unit_conv2x2_four_dispatched_output[380];
    float unit_conv2x2_four_simd_output[380];
    float conv7x7_input[342];
    float conv7x7_weights[392];
    float conv7x7_output[684];
    float conv7x7_simd_output[684];
    float conv7x7_dispatched_output[684];
    float conv5x5_input[342];
    float conv5x5_weights[200];
    float conv5x5_output[684];
    float conv5x5_simd_output[684];
    float conv5x5_dispatched_output[684];
    float conv7x1_weights[56];
    float conv7x1_output[684];
    float conv7x1_simd_output[684];
    float conv7x1_dispatched_output[684];
    float conv1x7_weights[56];
    float conv1x7_output[684];
    float conv1x7_simd_output[684];
    float conv1x7_dispatched_output[684];
    float conv5x1_weights[40];
    float conv5x1_output[684];
    float conv5x1_simd_output[684];
    float conv5x1_dispatched_output[684];
    float conv1x5_weights[40];
    float conv1x5_output[684];
    float conv1x5_simd_output[684];
    float conv1x5_dispatched_output[684];
    float grouped_input[64];
    float grouped_weights[108];
    float grouped_output[96];
    float depthwise_input[60];
    float depthwise_weights[18];
    float depthwise_output[60];
    float unit_depthwise_input[80];
    float unit_depthwise_weights[18];
    float unit_depthwise_output[80];
    float unit_depthwise_dispatched_output[80];
    float unit_depthwise_simd_output[80];
    float stride2x1_depthwise_input[190];
    float stride2x1_depthwise_output[114];
    float stride2x1_depthwise_dispatched_output[114];
    float stride2x1_depthwise_simd_output[114];
    float unit_depthwise5x5_input[228];
    float unit_depthwise5x5_weights[50];
    float unit_depthwise5x5_output[228];
    float unit_depthwise5x5_dispatched_output[228];
    float unit_depthwise5x5_simd_output[228];
    float unit_depthwise9x9_input[380];
    float unit_depthwise9x9_weights[162];
    float unit_depthwise9x9_output[380];
    float unit_depthwise9x9_dispatched_output[380];
    float unit_depthwise9x9_simd_output[380];
    float asymmetric_input[6];
    float asymmetric_weights[4];
    float asymmetric_output[9];
    float pointwise_input[80];
    float pointwise_weights[12];
    float pointwise_output[120];
    float pointwise_dispatched_output[120];
    float pointwise_simd_output[120];
    float packed_pointwise_input[210];
    float packed_pointwise_weights[35];
    float packed_pointwise_packed_weights[40];
    float packed_pointwise_output[294];
    float packed_pointwise_simd_output[294];
    float batch_norm_input[24];
    float batch_norm_output[24];
    float batch_norm_in_place[24];
    float transpose_conv_input[8];
    float transpose_conv_weights[8];
    float transpose_conv_output[16];
    float transpose_conv_simd_input[48];
    float transpose_conv_simd_weights[24];
    float transpose_conv_simd_output[288];
    float transpose_conv_sse2_output[288];
    float transpose_conv_avx2_output[288];
    float transpose_conv_arch_output[288];
    float transpose_conv_range_output[288];
    lw_simd_level simd_level;
    lw_status status;

    fill_values(normal_input, 80u, 5u, 19u, 9, 4.0f);
    fill_values(normal_weights, 54u, 7u, 17u, 8, 6.0f);
    status = lw_scalar_conv2d_f32(normal_input, normal_weights, normal_bias, 3u, normal_output,
                                  normal_input_dimensions, normal_weight_dimensions,
                                  normal_output_dimensions, normal_kernel, normal_strides,
                                  unit_dilations, normal_pads, 1u);
    if (!expect_status("normal conv", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("conv", normal_output, 36u);

    fill_values(stride2_input, 180u, 11u, 41u, 20, 10.0f);
    fill_values(stride2_weights, 54u, 13u, 31u, 15, 8.0f);
    lw_scalar_conv3x3_stride2_pad1_f32(stride2_input, stride2_weights, stride2_bias, stride2_output,
                                       stride2_input_dimensions, stride2_output_dimensions);
    simd_level = lw_detect_simd_level();
    if (lw_simd_level_is_sse2(simd_level)) {
        lw_sse2_conv3x3_stride2_pad1_f32(stride2_input, stride2_weights, stride2_bias,
                                         stride2_simd_output, stride2_input_dimensions,
                                         stride2_output_dimensions);
        if (memcmp(stride2_output, stride2_simd_output, sizeof(stride2_output)) != 0) {
            fprintf(stderr, "SSE2 stride-2 Conv differs from scalar output\n");
            return 1;
        }
    }
    if (lw_simd_level_is_avx2(simd_level)) {
        lw_avx2_conv3x3_stride2_pad1_f32(stride2_input, stride2_weights, stride2_bias,
                                         stride2_simd_output, stride2_input_dimensions,
                                         stride2_output_dimensions);
        if (memcmp(stride2_output, stride2_simd_output, sizeof(stride2_output)) != 0) {
            fprintf(stderr, "AVX2 stride-2 Conv differs from scalar output\n");
            return 1;
        }
    }
    status = lw_scalar_conv2d_f32(stride2_input, stride2_weights, stride2_bias, 3u,
                                  stride2_dispatched_output, stride2_input_dimensions,
                                  stride2_weight_dimensions, stride2_output_dimensions,
                                  normal_kernel, normal_strides, unit_dilations, normal_pads, 1u);
    if (!expect_status("stride-2 conv", status, LW_STATUS_OK) ||
        memcmp(stride2_output, stride2_dispatched_output, sizeof(stride2_output)) != 0) {
        fprintf(stderr, "dispatched stride-2 Conv differs from scalar output\n");
        return 1;
    }
    print_values("stride2_conv", stride2_output, 81u);

    /* Four output channels exercise the AVX2 kernel that reuses each gathered
     * input vector across four independent NCHW output planes. */
    fill_values(stride2_four_weights, 72u, 19u, 47u, 23, 12.0f);
    lw_scalar_conv3x3_stride2_pad1_f32(stride2_input, stride2_four_weights, stride2_four_bias,
                                       stride2_four_output, stride2_input_dimensions,
                                       stride2_four_output_dimensions);
    if (lw_simd_level_is_avx2(simd_level)) {
        lw_avx2_conv3x3_stride2_pad1_f32(stride2_input, stride2_four_weights, stride2_four_bias,
                                         stride2_four_simd_output, stride2_input_dimensions,
                                         stride2_four_output_dimensions);
        if (memcmp(stride2_four_output, stride2_four_simd_output, sizeof(stride2_four_output)) !=
            0) {
            fprintf(stderr, "AVX2 four-output stride-2 Conv differs from scalar output\n");
            return 1;
        }
    }

    fill_values(unit_conv_input, 285u, 17u, 43u, 21, 11.0f);
    fill_values(unit_conv_weights, 54u, 19u, 37u, 18, 9.0f);
    lw_scalar_conv3x3_unit_pad1_f32(unit_conv_input, unit_conv_weights, unit_conv_bias,
                                    unit_conv_output, unit_conv_input_dimensions,
                                    unit_conv_output_dimensions);
    if (lw_simd_level_is_sse2(simd_level)) {
        lw_sse2_conv3x3_unit_pad1_f32(unit_conv_input, unit_conv_weights, unit_conv_bias,
                                      unit_conv_simd_output, unit_conv_input_dimensions,
                                      unit_conv_output_dimensions);
        if (memcmp(unit_conv_output, unit_conv_simd_output, sizeof(unit_conv_output)) != 0) {
            fprintf(stderr, "SSE2 unit-stride Conv differs from scalar output\n");
            return 1;
        }
    }
    if (lw_simd_level_is_neon(simd_level)) {
        lw_neon_conv3x3_unit_pad1_f32(unit_conv_input, unit_conv_weights, unit_conv_bias,
                                      unit_conv_simd_output, unit_conv_input_dimensions,
                                      unit_conv_output_dimensions);
        if (memcmp(unit_conv_output, unit_conv_simd_output, sizeof(unit_conv_output)) != 0) {
            fprintf(stderr, "NEON unit-stride Conv differs from scalar output\n");
            return 1;
        }
    }
    if (lw_simd_level_is_avx2(simd_level)) {
        lw_avx2_conv3x3_unit_pad1_f32(unit_conv_input, unit_conv_weights, unit_conv_bias,
                                      unit_conv_simd_output, unit_conv_input_dimensions,
                                      unit_conv_output_dimensions);
        if (memcmp(unit_conv_output, unit_conv_simd_output, sizeof(unit_conv_output)) != 0) {
            fprintf(stderr, "AVX2 unit-stride Conv differs from scalar output\n");
            return 1;
        }
    }
    status = lw_scalar_conv2d_f32(unit_conv_input, unit_conv_weights, unit_conv_bias, 2u,
                                  unit_conv_dispatched_output, unit_conv_input_dimensions,
                                  unit_conv_weight_dimensions, unit_conv_output_dimensions,
                                  normal_kernel, unit_strides, unit_dilations, normal_pads, 1u);
    if (!expect_status("unit-stride conv", status, LW_STATUS_OK) ||
        memcmp(unit_conv_output, unit_conv_dispatched_output, sizeof(unit_conv_output)) != 0) {
        fprintf(stderr, "dispatched unit-stride Conv differs from scalar output\n");
        return 1;
    }
    print_values("unit_stride_conv", unit_conv_output, 190u);

    fill_values(conv7x7_input, 342u, 17u, 43u, 21, 11.0f);
    fill_values(conv7x7_weights, 392u, 19u, 37u, 18, 9.0f);
    reference_conv7x7(conv7x7_input, conv7x7_weights, conv7x7_bias, conv7x7_output,
                     conv7x7_input_dimensions, conv7x7_output_dimensions);
    if (lw_simd_level_is_avx2(simd_level)) {
        lw_avx2_conv7x7_unit_pad3_f32(conv7x7_input, conv7x7_weights, conv7x7_bias,
                                      conv7x7_simd_output, conv7x7_input_dimensions,
                                      conv7x7_output_dimensions);
        if (memcmp(conv7x7_output, conv7x7_simd_output, sizeof(conv7x7_output)) != 0) {
            fprintf(stderr, "AVX2 7x7 unit-stride Conv differs from scalar output\n");
            return 1;
        }
    }
    status = lw_scalar_conv2d_f32(
        conv7x7_input, conv7x7_weights, conv7x7_bias, 4u, conv7x7_dispatched_output,
        conv7x7_input_dimensions, conv7x7_weight_dimensions, conv7x7_output_dimensions,
        conv7x7_kernel, unit_strides, unit_dilations, conv7x7_pads, 1u);
    if (!expect_status("7x7 unit-stride conv", status, LW_STATUS_OK) ||
        memcmp(conv7x7_output, conv7x7_dispatched_output, sizeof(conv7x7_output)) != 0) {
        fprintf(stderr, "dispatched 7x7 Conv differs from scalar output\n");
        return 1;
    }

    fill_values(conv5x5_input, 342u, 17u, 43u, 21, 11.0f);
    fill_values(conv5x5_weights, 200u, 19u, 37u, 18, 9.0f);
    reference_conv5x5(conv5x5_input, conv5x5_weights, conv7x7_bias, conv5x5_output,
                     conv7x7_input_dimensions, conv7x7_output_dimensions);
    if (lw_simd_level_is_avx2(simd_level)) {
        lw_avx2_conv5x5_unit_pad2_f32(conv5x5_input, conv5x5_weights, conv7x7_bias,
                                      conv5x5_simd_output, conv7x7_input_dimensions,
                                      conv7x7_output_dimensions);
        if (memcmp(conv5x5_output, conv5x5_simd_output, sizeof(conv5x5_output)) != 0) {
            fprintf(stderr, "AVX2 5x5 unit-stride Conv differs from scalar output\n");
            return 1;
        }
    }
    status = lw_scalar_conv2d_f32(
        conv5x5_input, conv5x5_weights, conv7x7_bias, 4u, conv5x5_dispatched_output,
        conv7x7_input_dimensions, conv5x5_weight_dimensions, conv7x7_output_dimensions,
        conv5x5_kernel, unit_strides, unit_dilations, conv5x5_pads, 1u);
    if (!expect_status("5x5 unit-stride conv", status, LW_STATUS_OK) ||
        memcmp(conv5x5_output, conv5x5_dispatched_output, sizeof(conv5x5_output)) != 0) {
        fprintf(stderr, "dispatched 5x5 Conv differs from scalar output\n");
        return 1;
    }

    /* Four output channels exercise the AVX2 unit-stride kernel that reuses
     * every input vector across four independent NCHW output planes. */
    fill_values(unit_conv_four_weights, 108u, 23u, 47u, 23, 12.0f);
    lw_scalar_conv3x3_unit_pad1_f32(unit_conv_input, unit_conv_four_weights, unit_conv_four_bias,
                                    unit_conv_four_output, unit_conv_input_dimensions,
                                    unit_conv_four_output_dimensions);
    if (lw_simd_level_is_avx2(simd_level)) {
        lw_avx2_conv3x3_unit_pad1_f32(unit_conv_input, unit_conv_four_weights, unit_conv_four_bias,
                                      unit_conv_four_simd_output, unit_conv_input_dimensions,
                                      unit_conv_four_output_dimensions);
        if (memcmp(unit_conv_four_output, unit_conv_four_simd_output,
                   sizeof(unit_conv_four_output)) != 0) {
            fprintf(stderr, "AVX2 four-output unit-stride Conv differs from scalar output\n");
            return 1;
        }
    }

    fill_values(unit_conv2x2_weights, 24u, 23u, 41u, 20, 10.0f);
    lw_scalar_conv2x2_unit_pad_end1_f32(unit_conv_input, unit_conv2x2_weights, unit_conv_bias,
                                        unit_conv2x2_output, unit_conv_input_dimensions,
                                        unit_conv_output_dimensions);
    if (lw_simd_level_is_sse2(simd_level)) {
        lw_sse2_conv2x2_unit_pad_end1_f32(unit_conv_input, unit_conv2x2_weights, unit_conv_bias,
                                          unit_conv2x2_simd_output, unit_conv_input_dimensions,
                                          unit_conv_output_dimensions);
        if (memcmp(unit_conv2x2_output, unit_conv2x2_simd_output, sizeof(unit_conv2x2_output)) !=
            0) {
            fprintf(stderr, "SSE2 2x2 Conv differs from scalar output\n");
            return 1;
        }
    }
    if (lw_simd_level_is_avx2(simd_level)) {
        lw_avx2_conv2x2_unit_pad_end1_f32(unit_conv_input, unit_conv2x2_weights, unit_conv_bias,
                                          unit_conv2x2_simd_output, unit_conv_input_dimensions,
                                          unit_conv_output_dimensions);
        if (memcmp(unit_conv2x2_output, unit_conv2x2_simd_output, sizeof(unit_conv2x2_output)) !=
            0) {
            fprintf(stderr, "AVX2 2x2 Conv differs from scalar output\n");
            return 1;
        }
    }
    status = lw_scalar_conv2d_f32(
        unit_conv_input, unit_conv2x2_weights, unit_conv_bias, 2u, unit_conv2x2_dispatched_output,
        unit_conv_input_dimensions, unit_conv2x2_weight_dimensions, unit_conv_output_dimensions,
        unit_conv2x2_kernel, unit_strides, unit_dilations, unit_conv2x2_pads, 1u);
    if (!expect_status("2x2 unit-stride conv", status, LW_STATUS_OK) ||
        memcmp(unit_conv2x2_output, unit_conv2x2_dispatched_output, sizeof(unit_conv2x2_output)) !=
            0) {
        fprintf(stderr, "dispatched 2x2 Conv differs from scalar output\n");
        return 1;
    }
    print_values("unit_stride_conv2x2", unit_conv2x2_output, 190u);

    fill_values(unit_conv2x2_four_weights, 48u, 27u, 43u, 21, 8.0f);
    lw_scalar_conv2x2_unit_pad_end1_f32(
        unit_conv_input, unit_conv2x2_four_weights, unit_conv_four_bias,
        unit_conv2x2_four_output, unit_conv_input_dimensions, unit_conv_four_output_dimensions);
    if (lw_simd_level_is_sse2(simd_level)) {
        lw_sse2_conv2x2_unit_pad_end1_f32(
            unit_conv_input, unit_conv2x2_four_weights, unit_conv_four_bias,
            unit_conv2x2_four_simd_output, unit_conv_input_dimensions,
            unit_conv_four_output_dimensions);
        if (memcmp(unit_conv2x2_four_output, unit_conv2x2_four_simd_output,
                   sizeof(unit_conv2x2_four_output)) != 0) {
            fprintf(stderr, "SSE2 four-output 2x2 Conv differs from scalar output\n");
            return 1;
        }
    }
    if (lw_simd_level_is_avx2(simd_level)) {
        lw_avx2_conv2x2_unit_pad_end1_f32(
            unit_conv_input, unit_conv2x2_four_weights, unit_conv_four_bias,
            unit_conv2x2_four_simd_output, unit_conv_input_dimensions,
            unit_conv_four_output_dimensions);
        if (memcmp(unit_conv2x2_four_output, unit_conv2x2_four_simd_output,
                   sizeof(unit_conv2x2_four_output)) != 0) {
            fprintf(stderr, "AVX2 four-output 2x2 Conv differs from scalar output\n");
            return 1;
        }
    }
    status = lw_scalar_conv2d_f32(
        unit_conv_input, unit_conv2x2_four_weights, unit_conv_four_bias, 4u,
        unit_conv2x2_four_dispatched_output, unit_conv_input_dimensions,
        unit_conv2x2_four_weight_dimensions, unit_conv_four_output_dimensions,
        unit_conv2x2_kernel, unit_strides, unit_dilations, unit_conv2x2_pads, 1u);
    if (!expect_status("four-output 2x2 unit-stride conv", status, LW_STATUS_OK) ||
        memcmp(unit_conv2x2_four_output, unit_conv2x2_four_dispatched_output,
               sizeof(unit_conv2x2_four_output)) != 0) {
        fprintf(stderr, "dispatched four-output 2x2 Conv differs from scalar output\n");
        return 1;
    }
    print_values("unit_stride_conv2x2_four", unit_conv2x2_four_output, 380u);

    fill_values(grouped_input, 64u, 3u, 23u, 11, 5.0f);
    fill_values(grouped_weights, 108u, 11u, 29u, 14, 7.0f);
    status = lw_scalar_conv2d_f32(grouped_input, grouped_weights, NULL, 0u, grouped_output,
                                  grouped_input_dimensions, grouped_weight_dimensions,
                                  grouped_output_dimensions, normal_kernel, unit_strides,
                                  unit_dilations, normal_pads, 2u);
    if (!expect_status("grouped conv", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("grouped_conv", grouped_output, 96u);

    fill_values(depthwise_input, 60u, 13u, 31u, 15, 8.0f);
    fill_values(depthwise_weights, 18u, 5u, 13u, 6, 5.0f);
    status = lw_scalar_conv2d_f32(depthwise_input, depthwise_weights, NULL, 0u, depthwise_output,
                                  depthwise_input_dimensions, depthwise_weight_dimensions,
                                  depthwise_output_dimensions, depthwise_kernel, unit_strides,
                                  depthwise_dilations, normal_pads, 3u);
    if (!expect_status("depthwise conv", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("depthwise_conv", depthwise_output, 60u);

    fill_values(unit_depthwise_input, 80u, 17u, 37u, 18, 9.0f);
    fill_values(unit_depthwise_weights, 18u, 7u, 19u, 9, 6.0f);
    lw_scalar_depthwise_conv3x3_unit_pad1_f32(unit_depthwise_input, unit_depthwise_weights,
                                              unit_depthwise_bias, unit_depthwise_output,
                                              unit_depthwise_dimensions);
    if (lw_simd_level_is_sse2(simd_level)) {
        lw_sse2_depthwise_conv3x3_unit_pad1_f32(unit_depthwise_input, unit_depthwise_weights,
                                                unit_depthwise_bias, unit_depthwise_simd_output,
                                                unit_depthwise_dimensions);
        if (memcmp(unit_depthwise_output, unit_depthwise_simd_output,
                   sizeof(unit_depthwise_output)) != 0) {
            fprintf(stderr, "SSE2 depthwise Conv differs from scalar output\n");
            return 1;
        }
    }
    if (lw_simd_level_is_avx2(simd_level)) {
        lw_avx2_depthwise_conv3x3_unit_pad1_f32(unit_depthwise_input, unit_depthwise_weights,
                                                unit_depthwise_bias, unit_depthwise_simd_output,
                                                unit_depthwise_dimensions);
        if (memcmp(unit_depthwise_output, unit_depthwise_simd_output,
                   sizeof(unit_depthwise_output)) != 0) {
            fprintf(stderr, "AVX2 depthwise Conv differs from scalar output\n");
            return 1;
        }
    }
    status = lw_scalar_conv2d_f32(unit_depthwise_input, unit_depthwise_weights, unit_depthwise_bias,
                                  2u, unit_depthwise_dispatched_output, unit_depthwise_dimensions,
                                  unit_depthwise_weight_dimensions, unit_depthwise_dimensions,
                                  normal_kernel, unit_strides, unit_dilations, normal_pads, 2u);
    if (!expect_status("unit depthwise conv", status, LW_STATUS_OK) ||
        memcmp(unit_depthwise_output, unit_depthwise_dispatched_output,
               sizeof(unit_depthwise_output)) != 0) {
        fprintf(stderr, "dispatched depthwise Conv differs from scalar output\n");
        return 1;
    }
    print_values("unit_depthwise_conv", unit_depthwise_output, 80u);

    fill_values(stride2x1_depthwise_input, 190u, 23u, 47u, 23, 11.0f);
    lw_scalar_depthwise_conv3x3_stride2x1_pad1_f32(
        stride2x1_depthwise_input, unit_depthwise_weights, unit_depthwise_bias,
        stride2x1_depthwise_output, stride2x1_depthwise_input_dimensions,
        stride2x1_depthwise_output_dimensions);
    if (lw_simd_level_is_sse2(simd_level)) {
        lw_sse2_depthwise_conv3x3_stride2x1_pad1_f32(
            stride2x1_depthwise_input, unit_depthwise_weights, unit_depthwise_bias,
            stride2x1_depthwise_simd_output, stride2x1_depthwise_input_dimensions,
            stride2x1_depthwise_output_dimensions);
        if (memcmp(stride2x1_depthwise_output, stride2x1_depthwise_simd_output,
                   sizeof(stride2x1_depthwise_output)) != 0) {
            fprintf(stderr, "SSE2 stride-2x1 depthwise Conv differs from scalar output\n");
            return 1;
        }
    }
    if (lw_simd_level_is_avx2(simd_level)) {
        lw_avx2_depthwise_conv3x3_stride2x1_pad1_f32(
            stride2x1_depthwise_input, unit_depthwise_weights, unit_depthwise_bias,
            stride2x1_depthwise_simd_output, stride2x1_depthwise_input_dimensions,
            stride2x1_depthwise_output_dimensions);
        if (memcmp(stride2x1_depthwise_output, stride2x1_depthwise_simd_output,
                   sizeof(stride2x1_depthwise_output)) != 0) {
            fprintf(stderr, "AVX2 stride-2x1 depthwise Conv differs from scalar output\n");
            return 1;
        }
    }
    status = lw_scalar_conv2d_f32(
        stride2x1_depthwise_input, unit_depthwise_weights, unit_depthwise_bias, 2u,
        stride2x1_depthwise_dispatched_output, stride2x1_depthwise_input_dimensions,
        unit_depthwise_weight_dimensions, stride2x1_depthwise_output_dimensions, normal_kernel,
        stride2x1_depthwise_strides, unit_dilations, normal_pads, 2u);
    if (!expect_status("stride-2x1 depthwise conv", status, LW_STATUS_OK) ||
        memcmp(stride2x1_depthwise_output, stride2x1_depthwise_dispatched_output,
               sizeof(stride2x1_depthwise_output)) != 0) {
        fprintf(stderr, "dispatched stride-2x1 depthwise Conv differs from scalar output\n");
        return 1;
    }
    print_values("stride2x1_depthwise_conv", stride2x1_depthwise_output, 114u);

    fill_values(unit_depthwise5x5_input, 228u, 29u, 53u, 26, 13.0f);
    fill_values(unit_depthwise5x5_weights, 50u, 31u, 47u, 23, 12.0f);
    lw_scalar_depthwise_conv5x5_unit_pad2_f32(unit_depthwise5x5_input, unit_depthwise5x5_weights,
                                              unit_depthwise_bias, unit_depthwise5x5_output,
                                              unit_depthwise5x5_dimensions);
    if (lw_simd_level_is_sse2(simd_level)) {
        lw_sse2_depthwise_conv5x5_unit_pad2_f32(unit_depthwise5x5_input, unit_depthwise5x5_weights,
                                                unit_depthwise_bias, unit_depthwise5x5_simd_output,
                                                unit_depthwise5x5_dimensions);
        if (memcmp(unit_depthwise5x5_output, unit_depthwise5x5_simd_output,
                   sizeof(unit_depthwise5x5_output)) != 0) {
            fprintf(stderr, "SSE2 depthwise 5x5 Conv differs from scalar output\n");
            return 1;
        }
    }
    if (lw_simd_level_is_avx2(simd_level)) {
        lw_avx2_depthwise_conv5x5_unit_pad2_f32(unit_depthwise5x5_input, unit_depthwise5x5_weights,
                                                unit_depthwise_bias, unit_depthwise5x5_simd_output,
                                                unit_depthwise5x5_dimensions);
        if (memcmp(unit_depthwise5x5_output, unit_depthwise5x5_simd_output,
                   sizeof(unit_depthwise5x5_output)) != 0) {
            fprintf(stderr, "AVX2 depthwise 5x5 Conv differs from scalar output\n");
            return 1;
        }
    }
    status = lw_scalar_conv2d_f32(unit_depthwise5x5_input, unit_depthwise5x5_weights,
                                  unit_depthwise_bias, 2u, unit_depthwise5x5_dispatched_output,
                                  unit_depthwise5x5_dimensions, unit_depthwise5x5_weight_dimensions,
                                  unit_depthwise5x5_dimensions, depthwise5x5_kernel, unit_strides,
                                  unit_dilations, pad2, 2u);
    if (!expect_status("unit depthwise 5x5 conv", status, LW_STATUS_OK) ||
        memcmp(unit_depthwise5x5_output, unit_depthwise5x5_dispatched_output,
               sizeof(unit_depthwise5x5_output)) != 0) {
        fprintf(stderr, "dispatched depthwise 5x5 Conv differs from scalar output\n");
        return 1;
    }
    print_values("unit_depthwise_conv5x5", unit_depthwise5x5_output, 228u);

    fill_values(unit_depthwise9x9_input, 380u, 37u, 67u, 31, 17.0f);
    fill_values(unit_depthwise9x9_weights, 162u, 41u, 71u, 33, 19.0f);
    reference_depthwise9x9(unit_depthwise9x9_input, unit_depthwise9x9_weights,
                           unit_depthwise_bias, unit_depthwise9x9_output,
                           unit_depthwise9x9_dimensions);
    if (lw_simd_level_is_avx2(simd_level)) {
        lw_avx2_depthwise_conv9x9_unit_pad4_f32(
            unit_depthwise9x9_input, unit_depthwise9x9_weights, unit_depthwise_bias,
            unit_depthwise9x9_simd_output, unit_depthwise9x9_dimensions);
        if (memcmp(unit_depthwise9x9_output, unit_depthwise9x9_simd_output,
                   sizeof(unit_depthwise9x9_output)) != 0) {
            fprintf(stderr, "AVX2 depthwise 9x9 Conv differs from scalar output\n");
            return 1;
        }
    }
    status = lw_scalar_conv2d_f32(
        unit_depthwise9x9_input, unit_depthwise9x9_weights, unit_depthwise_bias, 2u,
        unit_depthwise9x9_dispatched_output, unit_depthwise9x9_dimensions,
        unit_depthwise9x9_weight_dimensions, unit_depthwise9x9_dimensions, depthwise9x9_kernel,
        unit_strides, unit_dilations, pad4, 2u);
    if (!expect_status("unit depthwise 9x9 conv", status, LW_STATUS_OK) ||
        memcmp(unit_depthwise9x9_output, unit_depthwise9x9_dispatched_output,
               sizeof(unit_depthwise9x9_output)) != 0) {
        fprintf(stderr, "dispatched depthwise 9x9 Conv differs from scalar output\n");
        return 1;
    }
    print_values("unit_depthwise_conv9x9", unit_depthwise9x9_output, 380u);

    fill_values(conv7x1_weights, 56u, 43u, 79u, 37, 17.0f);
    reference_conv_axis(conv7x7_input, conv7x1_weights, conv7x7_bias, conv7x1_output,
                        conv7x7_input_dimensions, conv7x7_output_dimensions, 7u, 1u, 3u, 0u);
    if (lw_simd_level_is_avx2(simd_level)) {
        lw_avx2_conv7x1_unit_pad3_f32(conv7x7_input, conv7x1_weights, conv7x7_bias,
                                      conv7x1_simd_output, conv7x7_input_dimensions,
                                      conv7x7_output_dimensions);
        if (memcmp(conv7x1_output, conv7x1_simd_output, sizeof(conv7x1_output)) != 0) {
            fprintf(stderr, "AVX2 7x1 Conv differs from scalar output\n");
            return 1;
        }
    }
    status = lw_scalar_conv2d_f32(conv7x7_input, conv7x1_weights, conv7x7_bias, 4u,
                                  conv7x1_dispatched_output, conv7x7_input_dimensions,
                                  conv7x1_weight_dimensions, conv7x7_output_dimensions,
                                  conv7x1_kernel, unit_strides, unit_dilations, conv7x1_pads, 1u);
    if (!expect_status("7x1 conv", status, LW_STATUS_OK) ||
        memcmp(conv7x1_output, conv7x1_dispatched_output, sizeof(conv7x1_output)) != 0) {
        fprintf(stderr, "dispatched 7x1 Conv differs from scalar output\n");
        return 1;
    }
    print_values("conv7x1", conv7x1_output, 684u);

    fill_values(conv1x7_weights, 56u, 47u, 83u, 39, 19.0f);
    reference_conv_axis(conv7x7_input, conv1x7_weights, conv7x7_bias, conv1x7_output,
                        conv7x7_input_dimensions, conv7x7_output_dimensions, 1u, 7u, 0u, 3u);
    if (lw_simd_level_is_avx2(simd_level)) {
        lw_avx2_conv1x7_unit_pad3_f32(conv7x7_input, conv1x7_weights, conv7x7_bias,
                                      conv1x7_simd_output, conv7x7_input_dimensions,
                                      conv7x7_output_dimensions);
        if (memcmp(conv1x7_output, conv1x7_simd_output, sizeof(conv1x7_output)) != 0) {
            fprintf(stderr, "AVX2 1x7 Conv differs from scalar output\n");
            return 1;
        }
    }
    status = lw_scalar_conv2d_f32(conv7x7_input, conv1x7_weights, conv7x7_bias, 4u,
                                  conv1x7_dispatched_output, conv7x7_input_dimensions,
                                  conv1x7_weight_dimensions, conv7x7_output_dimensions,
                                  conv1x7_kernel, unit_strides, unit_dilations, conv1x7_pads, 1u);
    if (!expect_status("1x7 conv", status, LW_STATUS_OK) ||
        memcmp(conv1x7_output, conv1x7_dispatched_output, sizeof(conv1x7_output)) != 0) {
        fprintf(stderr, "dispatched 1x7 Conv differs from scalar output\n");
        return 1;
    }
    print_values("conv1x7", conv1x7_output, 684u);

    fill_values(conv5x1_weights, 40u, 53u, 89u, 41, 23.0f);
    reference_conv_axis(conv7x7_input, conv5x1_weights, conv7x7_bias, conv5x1_output,
                        conv7x7_input_dimensions, conv7x7_output_dimensions, 5u, 1u, 2u, 0u);
    if (lw_simd_level_is_avx2(simd_level)) {
        lw_avx2_conv5x1_unit_pad2_f32(conv7x7_input, conv5x1_weights, conv7x7_bias,
                                      conv5x1_simd_output, conv7x7_input_dimensions,
                                      conv7x7_output_dimensions);
        if (memcmp(conv5x1_output, conv5x1_simd_output, sizeof(conv5x1_output)) != 0) {
            fprintf(stderr, "AVX2 5x1 Conv differs from scalar output\n");
            return 1;
        }
    }
    status = lw_scalar_conv2d_f32(conv7x7_input, conv5x1_weights, conv7x7_bias, 4u,
                                  conv5x1_dispatched_output, conv7x7_input_dimensions,
                                  conv5x1_weight_dimensions, conv7x7_output_dimensions,
                                  conv5x1_kernel, unit_strides, unit_dilations, conv5x1_pads, 1u);
    if (!expect_status("5x1 conv", status, LW_STATUS_OK) ||
        memcmp(conv5x1_output, conv5x1_dispatched_output, sizeof(conv5x1_output)) != 0) {
        fprintf(stderr, "dispatched 5x1 Conv differs from scalar output\n");
        return 1;
    }
    print_values("conv5x1", conv5x1_output, 684u);

    fill_values(conv1x5_weights, 40u, 59u, 97u, 43, 29.0f);
    reference_conv_axis(conv7x7_input, conv1x5_weights, conv7x7_bias, conv1x5_output,
                        conv7x7_input_dimensions, conv7x7_output_dimensions, 1u, 5u, 0u, 2u);
    if (lw_simd_level_is_avx2(simd_level)) {
        lw_avx2_conv1x5_unit_pad2_f32(conv7x7_input, conv1x5_weights, conv7x7_bias,
                                      conv1x5_simd_output, conv7x7_input_dimensions,
                                      conv7x7_output_dimensions);
        if (memcmp(conv1x5_output, conv1x5_simd_output, sizeof(conv1x5_output)) != 0) {
            fprintf(stderr, "AVX2 1x5 Conv differs from scalar output\n");
            return 1;
        }
    }
    status = lw_scalar_conv2d_f32(conv7x7_input, conv1x5_weights, conv7x7_bias, 4u,
                                  conv1x5_dispatched_output, conv7x7_input_dimensions,
                                  conv1x5_weight_dimensions, conv7x7_output_dimensions,
                                  conv1x5_kernel, unit_strides, unit_dilations, conv1x5_pads, 1u);
    if (!expect_status("1x5 conv", status, LW_STATUS_OK) ||
        memcmp(conv1x5_output, conv1x5_dispatched_output, sizeof(conv1x5_output)) != 0) {
        fprintf(stderr, "dispatched 1x5 Conv differs from scalar output\n");
        return 1;
    }
    print_values("conv1x5", conv1x5_output, 684u);

    fill_values(asymmetric_input, 6u, 3u, 11u, 5, 4.0f);
    fill_values(asymmetric_weights, 4u, 5u, 13u, 6, 3.0f);
    status = lw_scalar_conv2d_f32(asymmetric_input, asymmetric_weights, NULL, 0u, asymmetric_output,
                                  asymmetric_input_dimensions, asymmetric_weight_dimensions,
                                  asymmetric_output_dimensions, asymmetric_kernel,
                                  asymmetric_strides, asymmetric_dilations, asymmetric_pads, 1u);
    if (!expect_status("asymmetric dilated conv", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("asymmetric_conv", asymmetric_output, 9u);

    fill_values(pointwise_input, 80u, 7u, 19u, 9, 5.0f);
    fill_values(pointwise_weights, 12u, 11u, 23u, 11, 6.0f);
    lw_scalar_conv1x1_unit_f32(pointwise_input, pointwise_weights, pointwise_bias, pointwise_output,
                               pointwise_input_dimensions, pointwise_output_dimensions, 2u, 2u, 3u);
    if (lw_simd_level_is_sse2(simd_level)) {
        lw_sse2_conv1x1_unit_f32(pointwise_input, pointwise_weights, pointwise_bias,
                                 pointwise_simd_output, pointwise_input_dimensions,
                                 pointwise_output_dimensions, 2u, 2u, 3u);
        if (memcmp(pointwise_output, pointwise_simd_output, sizeof(pointwise_output)) != 0) {
            fprintf(stderr, "SSE2 pointwise Conv differs from scalar output\n");
            return 1;
        }
    }
    if (lw_simd_level_is_avx2(simd_level)) {
        lw_avx2_conv1x1_unit_f32(pointwise_input, pointwise_weights, pointwise_bias,
                                 pointwise_simd_output, pointwise_input_dimensions,
                                 pointwise_output_dimensions, 2u, 2u, 3u);
        if (memcmp(pointwise_output, pointwise_simd_output, sizeof(pointwise_output)) != 0) {
            fprintf(stderr, "AVX2 pointwise Conv differs from scalar output\n");
            return 1;
        }
    }
    status = lw_scalar_conv2d_f32(pointwise_input, pointwise_weights, pointwise_bias, 6u,
                                  pointwise_dispatched_output, pointwise_input_dimensions,
                                  pointwise_weight_dimensions, pointwise_output_dimensions,
                                  point_kernel, unit_strides, unit_dilations, no_pads, 2u);
    if (!expect_status("grouped pointwise conv", status, LW_STATUS_OK)) {
        return 1;
    }
    if (memcmp(pointwise_output, pointwise_dispatched_output, sizeof(pointwise_output)) != 0) {
        fprintf(stderr, "dispatched pointwise Conv differs from scalar output\n");
        return 1;
    }
    print_values("grouped_pointwise_conv", pointwise_output, 120u);

    /* Exercise four-output-channel packing, including the x64 16-value spatial
     * main loop, its five-value tail, and a three-output-channel block. */
    fill_values(packed_pointwise_input, 210u, 13u, 43u, 21, 11.0f);
    fill_values(packed_pointwise_weights, 35u, 17u, 37u, 18, 9.0f);
    lw_scalar_conv1x1_unit_f32(packed_pointwise_input, packed_pointwise_weights,
                               packed_pointwise_bias, packed_pointwise_output,
                               packed_pointwise_input_dimensions,
                               packed_pointwise_output_dimensions, 1u, 5u, 7u);
    lw_pack_conv1x1_weights_f32(packed_pointwise_weights, 5u, 7u, packed_pointwise_packed_weights);
    lw_scalar_packed_conv1x1_f32(packed_pointwise_input, packed_pointwise_packed_weights,
                                 packed_pointwise_bias, packed_pointwise_simd_output,
                                 packed_pointwise_input_dimensions,
                                 packed_pointwise_output_dimensions);
    if (memcmp(packed_pointwise_output, packed_pointwise_simd_output,
               sizeof(packed_pointwise_output)) != 0) {
        fprintf(stderr, "scalar packed pointwise Conv differs from canonical output\n");
        return 1;
    }
    if (lw_simd_level_is_sse2(simd_level)) {
        lw_sse2_packed_conv1x1_f32(packed_pointwise_input, packed_pointwise_packed_weights,
                                   packed_pointwise_bias, packed_pointwise_simd_output,
                                   packed_pointwise_input_dimensions,
                                   packed_pointwise_output_dimensions);
        if (memcmp(packed_pointwise_output, packed_pointwise_simd_output,
                   sizeof(packed_pointwise_output)) != 0) {
            fprintf(stderr, "SSE2 packed pointwise Conv differs from canonical output\n");
            return 1;
        }
    }
    if (lw_simd_level_is_neon(simd_level)) {
        lw_neon_packed_conv1x1_f32(packed_pointwise_input, packed_pointwise_packed_weights,
                                    packed_pointwise_bias, packed_pointwise_simd_output,
                                    packed_pointwise_input_dimensions,
                                    packed_pointwise_output_dimensions);
        if (memcmp(packed_pointwise_output, packed_pointwise_simd_output,
                   sizeof(packed_pointwise_output)) != 0) {
            fprintf(stderr, "NEON packed pointwise Conv differs from canonical output\n");
            return 1;
        }
    }
    if (lw_simd_level_is_lsx(simd_level)) {
        lw_lsx_packed_conv1x1_f32(packed_pointwise_input, packed_pointwise_packed_weights,
                                   packed_pointwise_bias, packed_pointwise_simd_output,
                                   packed_pointwise_input_dimensions,
                                   packed_pointwise_output_dimensions);
        if (memcmp(packed_pointwise_output, packed_pointwise_simd_output,
                   sizeof(packed_pointwise_output)) != 0) {
            fprintf(stderr, "LSX packed pointwise Conv differs from canonical output\n");
            return 1;
        }
    }
    if (lw_simd_level_is_avx2(simd_level)) {
        lw_avx2_packed_conv1x1_f32(packed_pointwise_input, packed_pointwise_packed_weights,
                                   packed_pointwise_bias, packed_pointwise_simd_output,
                                   packed_pointwise_input_dimensions,
                                   packed_pointwise_output_dimensions);
        if (memcmp(packed_pointwise_output, packed_pointwise_simd_output,
                   sizeof(packed_pointwise_output)) != 0) {
            fprintf(stderr, "AVX2 packed pointwise Conv differs from canonical output\n");
            return 1;
        }
    }
    lw_packed_conv1x1_f32(packed_pointwise_input, packed_pointwise_packed_weights,
                          packed_pointwise_bias, packed_pointwise_simd_output,
                          packed_pointwise_input_dimensions, packed_pointwise_output_dimensions);
    if (memcmp(packed_pointwise_output, packed_pointwise_simd_output,
               sizeof(packed_pointwise_output)) != 0) {
        fprintf(stderr, "dispatched packed pointwise Conv differs from canonical output\n");
        return 1;
    }
    print_values("packed_pointwise_conv", packed_pointwise_output, 294u);

    fill_values(transpose_conv_input, 8u, 3u, 13u, 6, 4.0f);
    fill_values(transpose_conv_weights, 8u, 5u, 17u, 8, 6.0f);
    status = lw_scalar_conv_transpose2d_f32(
        transpose_conv_input, transpose_conv_weights, transpose_conv_bias, 1u,
        transpose_conv_output, transpose_conv_input_dimensions, transpose_conv_weight_dimensions,
        transpose_conv_output_dimensions, transpose_conv_kernel, transpose_conv_strides,
        unit_dilations, no_pads, 1u);
    if (!expect_status("transpose conv", status, LW_STATUS_OK)) {
        return 1;
    }

    /* Width eight exercises the vector body of both dedicated transpose-Conv
     * kernels; compare each implementation with the dispatched result. */
    fill_values(transpose_conv_simd_input, 48u, 17u, 37u, 18, 9.0f);
    fill_values(transpose_conv_simd_weights, 24u, 11u, 29u, 14, 7.0f);
    lw_scalar_conv_transpose2x2_stride2_f32(
        transpose_conv_simd_input, transpose_conv_simd_weights, transpose_conv_simd_bias,
        transpose_conv_arch_output, transpose_conv_simd_input_dimensions,
        transpose_conv_simd_output_dimensions);
    memset(transpose_conv_range_output, 0xa5, sizeof(transpose_conv_range_output));
    lw_scalar_conv_transpose2x2_stride2_range_f32(
        transpose_conv_simd_input, transpose_conv_simd_weights, transpose_conv_simd_bias,
        transpose_conv_range_output, transpose_conv_simd_input_dimensions,
        transpose_conv_simd_output_dimensions, 0u, 1u);
    lw_scalar_conv_transpose2x2_stride2_range_f32(
        transpose_conv_simd_input, transpose_conv_simd_weights, transpose_conv_simd_bias,
        transpose_conv_range_output, transpose_conv_simd_input_dimensions,
        transpose_conv_simd_output_dimensions, 1u, 3u);
    if (memcmp(transpose_conv_arch_output, transpose_conv_range_output,
               sizeof(transpose_conv_range_output)) != 0) {
        fprintf(stderr, "scalar ranged transpose Conv differs from full output\n");
        return 1;
    }
    status = lw_scalar_conv_transpose2d_f32(
        transpose_conv_simd_input, transpose_conv_simd_weights, transpose_conv_simd_bias, 3u,
        transpose_conv_simd_output, transpose_conv_simd_input_dimensions,
        transpose_conv_simd_weight_dimensions, transpose_conv_simd_output_dimensions,
        transpose_conv_kernel, transpose_conv_strides, unit_dilations, no_pads, 1u);
    if (!expect_status("transpose conv SIMD shape", status, LW_STATUS_OK)) {
        return 1;
    }
    if (memcmp(transpose_conv_arch_output, transpose_conv_simd_output,
               sizeof(transpose_conv_simd_output)) != 0) {
        fprintf(stderr, "dispatched transpose Conv differs from scalar output\n");
        return 1;
    }
    if (lw_simd_level_is_sse2(simd_level)) {
        lw_sse2_conv_transpose2x2_stride2_f32(
            transpose_conv_simd_input, transpose_conv_simd_weights, transpose_conv_simd_bias,
            transpose_conv_sse2_output, transpose_conv_simd_input_dimensions,
            transpose_conv_simd_output_dimensions);
        if (memcmp(transpose_conv_arch_output, transpose_conv_sse2_output,
                   sizeof(transpose_conv_simd_output)) != 0) {
            fprintf(stderr, "SSE2 transpose Conv differs from dispatched output\n");
            return 1;
        }
        memset(transpose_conv_range_output, 0xa5, sizeof(transpose_conv_range_output));
        lw_sse2_conv_transpose2x2_stride2_range_f32(
            transpose_conv_simd_input, transpose_conv_simd_weights, transpose_conv_simd_bias,
            transpose_conv_range_output, transpose_conv_simd_input_dimensions,
            transpose_conv_simd_output_dimensions, 0u, 1u);
        lw_sse2_conv_transpose2x2_stride2_range_f32(
            transpose_conv_simd_input, transpose_conv_simd_weights, transpose_conv_simd_bias,
            transpose_conv_range_output, transpose_conv_simd_input_dimensions,
            transpose_conv_simd_output_dimensions, 1u, 3u);
        if (memcmp(transpose_conv_arch_output, transpose_conv_range_output,
                   sizeof(transpose_conv_range_output)) != 0) {
            fprintf(stderr, "SSE2 ranged transpose Conv differs from full output\n");
            return 1;
        }
    }
    if (lw_simd_level_is_neon(simd_level)) {
        lw_neon_conv_transpose2x2_stride2_f32(
            transpose_conv_simd_input, transpose_conv_simd_weights, transpose_conv_simd_bias,
            transpose_conv_arch_output, transpose_conv_simd_input_dimensions,
            transpose_conv_simd_output_dimensions);
        if (memcmp(transpose_conv_simd_output, transpose_conv_arch_output,
                   sizeof(transpose_conv_simd_output)) != 0) {
            fprintf(stderr, "NEON transpose Conv differs from dispatched output\n");
            return 1;
        }
        memset(transpose_conv_range_output, 0xa5, sizeof(transpose_conv_range_output));
        lw_neon_conv_transpose2x2_stride2_range_f32(
            transpose_conv_simd_input, transpose_conv_simd_weights, transpose_conv_simd_bias,
            transpose_conv_range_output, transpose_conv_simd_input_dimensions,
            transpose_conv_simd_output_dimensions, 0u, 1u);
        lw_neon_conv_transpose2x2_stride2_range_f32(
            transpose_conv_simd_input, transpose_conv_simd_weights, transpose_conv_simd_bias,
            transpose_conv_range_output, transpose_conv_simd_input_dimensions,
            transpose_conv_simd_output_dimensions, 1u, 3u);
        if (memcmp(transpose_conv_simd_output, transpose_conv_range_output,
                   sizeof(transpose_conv_range_output)) != 0) {
            fprintf(stderr, "NEON ranged transpose Conv differs from full output\n");
            return 1;
        }
    }
    if (lw_simd_level_is_avx2(simd_level)) {
        lw_avx2_conv_transpose2x2_stride2_f32(
            transpose_conv_simd_input, transpose_conv_simd_weights, transpose_conv_simd_bias,
            transpose_conv_avx2_output, transpose_conv_simd_input_dimensions,
            transpose_conv_simd_output_dimensions);
        if (memcmp(transpose_conv_arch_output, transpose_conv_avx2_output,
                   sizeof(transpose_conv_simd_output)) != 0) {
            fprintf(stderr, "AVX2 transpose Conv differs from dispatched output\n");
            return 1;
        }
        memset(transpose_conv_range_output, 0xa5, sizeof(transpose_conv_range_output));
        lw_avx2_conv_transpose2x2_stride2_range_f32(
            transpose_conv_simd_input, transpose_conv_simd_weights, transpose_conv_simd_bias,
            transpose_conv_range_output, transpose_conv_simd_input_dimensions,
            transpose_conv_simd_output_dimensions, 0u, 1u);
        lw_avx2_conv_transpose2x2_stride2_range_f32(
            transpose_conv_simd_input, transpose_conv_simd_weights, transpose_conv_simd_bias,
            transpose_conv_range_output, transpose_conv_simd_input_dimensions,
            transpose_conv_simd_output_dimensions, 1u, 3u);
        if (memcmp(transpose_conv_arch_output, transpose_conv_range_output,
                   sizeof(transpose_conv_range_output)) != 0) {
            fprintf(stderr, "AVX2 ranged transpose Conv differs from full output\n");
            return 1;
        }
    }
    print_values("conv_transpose", transpose_conv_output, 16u);

    status = lw_scalar_conv_transpose2d_f32(
        transpose_conv_input, transpose_conv_weights, transpose_conv_bias, 1u, transpose_conv_input,
        transpose_conv_input_dimensions, transpose_conv_weight_dimensions,
        transpose_conv_output_dimensions, transpose_conv_kernel, transpose_conv_strides,
        unit_dilations, no_pads, 1u);
    if (!expect_status("transpose conv alias", status, LW_STATUS_INVALID_ARGUMENT)) {
        return 1;
    }
    status = lw_scalar_conv_transpose2d_f32(
        transpose_conv_input, transpose_conv_weights, transpose_conv_bias, 1u,
        transpose_conv_output, transpose_conv_input_dimensions, transpose_conv_weight_dimensions,
        transpose_conv_output_dimensions, transpose_conv_kernel, transpose_conv_strides,
        unit_dilations, no_pads, 3u);
    if (!expect_status("transpose conv groups", status, LW_STATUS_INVALID_SHAPE)) {
        return 1;
    }

    fill_values(batch_norm_input, 24u, 7u, 21u, 10, 4.0f);
    status = lw_scalar_batch_normalization_f32(batch_norm_input, batch_norm_scale, batch_norm_bias,
                                               batch_norm_mean, batch_norm_variance, 3u, 1.0e-5f,
                                               batch_norm_output, 4u, batch_norm_dimensions);
    if (!expect_status("batch normalization", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("batch_norm", batch_norm_output, 24u);

    memcpy(batch_norm_in_place, batch_norm_input, sizeof(batch_norm_input));
    status = lw_scalar_batch_normalization_f32(
        batch_norm_in_place, batch_norm_scale, batch_norm_bias, batch_norm_mean,
        batch_norm_variance, 3u, 1.0e-5f, batch_norm_in_place, 4u, batch_norm_dimensions);
    if (!expect_status("in-place batch normalization", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("batch_norm_in_place", batch_norm_in_place, 24u);

    status = lw_scalar_conv2d_f32(normal_input, normal_weights, normal_bias, 3u, normal_output,
                                  normal_input_dimensions, invalid_weight_dimensions,
                                  normal_output_dimensions, normal_kernel, normal_strides,
                                  unit_dilations, normal_pads, 1u);
    if (!expect_status("conv input channels", status, LW_STATUS_INVALID_SHAPE)) {
        return 1;
    }
    status = lw_scalar_conv2d_f32(normal_input, normal_weights, normal_bias, 3u, normal_output,
                                  normal_input_dimensions, normal_weight_dimensions,
                                  invalid_output_dimensions, normal_kernel, normal_strides,
                                  unit_dilations, normal_pads, 1u);
    if (!expect_status("conv output shape", status, LW_STATUS_INVALID_SHAPE)) {
        return 1;
    }
    status = lw_scalar_conv2d_f32(normal_input, normal_weights, normal_bias, 2u, normal_output,
                                  normal_input_dimensions, normal_weight_dimensions,
                                  normal_output_dimensions, normal_kernel, normal_strides,
                                  unit_dilations, normal_pads, 1u);
    if (!expect_status("conv bias count", status, LW_STATUS_INVALID_SHAPE)) {
        return 1;
    }
    status = lw_scalar_conv2d_f32(normal_input, normal_weights, normal_bias, 3u, normal_output,
                                  normal_input_dimensions, normal_weight_dimensions,
                                  normal_output_dimensions, normal_kernel, normal_strides,
                                  unit_dilations, normal_pads, 0u);
    if (!expect_status("conv zero groups", status, LW_STATUS_INVALID_SHAPE)) {
        return 1;
    }
    status = lw_scalar_conv2d_f32(normal_input, normal_weights, normal_bias, 3u, normal_input,
                                  normal_input_dimensions, normal_weight_dimensions,
                                  normal_output_dimensions, normal_kernel, normal_strides,
                                  unit_dilations, normal_pads, 1u);
    if (!expect_status("conv alias", status, LW_STATUS_INVALID_ARGUMENT)) {
        return 1;
    }
    status = lw_scalar_batch_normalization_f32(batch_norm_input, batch_norm_scale, batch_norm_bias,
                                               batch_norm_mean, batch_norm_variance, 2u, 1.0e-5f,
                                               batch_norm_output, 4u, batch_norm_dimensions);
    if (!expect_status("batch normalization parameter count", status, LW_STATUS_INVALID_SHAPE)) {
        return 1;
    }
    status = lw_scalar_batch_normalization_f32(batch_norm_input, batch_norm_scale, batch_norm_bias,
                                               batch_norm_mean, invalid_variance, 3u, 1.0e-5f,
                                               batch_norm_output, 4u, batch_norm_dimensions);
    if (!expect_status("batch normalization variance", status, LW_STATUS_INVALID_ARGUMENT)) {
        return 1;
    }
    status = lw_scalar_batch_normalization_f32(batch_norm_input, batch_norm_scale, batch_norm_bias,
                                               batch_norm_mean, batch_norm_variance, 3u, 0.0f,
                                               batch_norm_output, 4u, batch_norm_dimensions);
    if (!expect_status("batch normalization epsilon", status, LW_STATUS_INVALID_SHAPE)) {
        return 1;
    }
    if (is_32_bit_process()) {
        const int32_t large_dimensions[4] = {1, 1, 65536, 65536};
        const int32_t point_weight_dimensions[4] = {1, 1, 1, 1};
        status = lw_scalar_conv2d_f32(normal_input, normal_weights, NULL, 0u, normal_output,
                                      large_dimensions, point_weight_dimensions, large_dimensions,
                                      point_kernel, unit_strides, unit_dilations, no_pads, 1u);
        if (!expect_status("conv 32-bit byte overflow", status, LW_STATUS_OUT_OF_BOUNDS)) {
            return 1;
        }
    }
    return 0;
}
