#include "simd_kernels.h"
#include "simd_platform.h"

#include "scalar_kernels.h"

#include <stddef.h>

#define LW_COMPILES_SSE2_CONV3X3 LW_SIMD_HAS_SSE2_INTRINSICS

LW_SIMD_SSE2_TARGET
void lw_sse2_conv3x3_unit_pad1_f32(const float* input, const float* weights, const float* bias,
                                   float* output, const int32_t input_dimensions[4],
                                   const int32_t output_dimensions[4]) {
#if LW_COMPILES_SSE2_CONV3X3
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
            __m128 initial_values = _mm_set1_ps(initial);
            uint64_t spatial = 0u;
            uint32_t input_channel;
            for (; spatial + 4u <= channel_plane; spatial += 4u) {
                _mm_storeu_ps(output_channel_data + (size_t)spatial, initial_values);
            }
            for (; spatial < channel_plane; ++spatial) {
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
                        __m128 weight_values = _mm_set1_ps(weight);
                        uint32_t output_y;
                        for (output_y = output_y_begin; output_y < output_y_end; ++output_y) {
                            uint32_t input_y = output_y + kernel_y - 1u;
                            const float* input_row =
                                input_channel_data + (size_t)((uint64_t)input_y * width);
                            float* output_row =
                                output_channel_data + (size_t)((uint64_t)output_y * width);
                            uint32_t output_x = output_x_begin;
                            for (; output_x + 4u <= output_x_end; output_x += 4u) {
                                uint32_t input_x = output_x + kernel_x - 1u;
                                __m128 input_values = _mm_loadu_ps(input_row + input_x);
                                __m128 output_values = _mm_loadu_ps(output_row + output_x);
                                output_values = _mm_add_ps(output_values,
                                                           _mm_mul_ps(input_values, weight_values));
                                _mm_storeu_ps(output_row + output_x, output_values);
                            }
                            for (; output_x < output_x_end; ++output_x) {
                                uint32_t input_x = output_x + kernel_x - 1u;
                                output_row[output_x] += input_row[input_x] * weight;
                            }
                        }
                    }
                }
            }
        }
    }
#else
    lw_scalar_conv3x3_unit_pad1_f32(input, weights, bias, output, input_dimensions,
                                    output_dimensions);
#endif
}

#if LW_COMPILES_SSE2_CONV3X3 && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("sse2")))
#endif
void lw_sse2_conv3x3_stride2_pad1_f32(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    const int32_t input_dimensions[4],
    const int32_t output_dimensions[4]) {
#if LW_COMPILES_SSE2_CONV3X3
    uint32_t input_height = (uint32_t)input_dimensions[2];
    uint32_t input_width = (uint32_t)input_dimensions[3];
    uint32_t output_height = (uint32_t)output_dimensions[2];
    uint32_t output_width = (uint32_t)output_dimensions[3];
    uint64_t input_channel_plane = (uint64_t)input_height * input_width;
    uint64_t output_channel_plane = (uint64_t)output_height * output_width;
    uint32_t batch;
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
            __m128 initial_values = _mm_set1_ps(initial);
            uint64_t spatial = 0u;
            uint32_t input_channel;
            for (; spatial + 4u <= output_channel_plane; spatial += 4u) {
                _mm_storeu_ps(output_channel_data + (size_t)spatial, initial_values);
            }
            for (; spatial < output_channel_plane; ++spatial) {
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
                    uint32_t output_y_begin = kernel_y == 0u ? 1u : 0u;
                    uint32_t output_y_end = output_height;
                    uint32_t kernel_x;
                    if (kernel_y == 2u && (uint64_t)(output_y_end - 1u) * 2u + 1u >= input_height) {
                        --output_y_end;
                    }
                    for (kernel_x = 0u; kernel_x < 3u; ++kernel_x) {
                        uint32_t output_x_begin = kernel_x == 0u ? 1u : 0u;
                        uint32_t output_x_end = output_width;
                        float weight = weight_channel_data[(size_t)kernel_y * 3u + kernel_x];
                        __m128 weight_values = _mm_set1_ps(weight);
                        uint32_t output_y;
                        if (kernel_x == 2u &&
                            (uint64_t)(output_x_end - 1u) * 2u + 1u >= input_width) {
                            --output_x_end;
                        }
                        for (output_y = output_y_begin; output_y < output_y_end; ++output_y) {
                            uint32_t input_y = output_y * 2u - 1u + kernel_y;
                            const float* input_row =
                                input_channel_data + (size_t)((uint64_t)input_y * input_width);
                            float* output_row =
                                output_channel_data + (size_t)((uint64_t)output_y * output_width);
                            uint32_t output_x = output_x_begin;
                            for (; output_x + 4u <= output_x_end; output_x += 4u) {
                                uint32_t input_x = output_x * 2u - 1u + kernel_x;
                                if ((uint64_t)input_x + 7u >= input_width) {
                                    break;
                                }
                                {
                                    __m128 first = _mm_loadu_ps(input_row + input_x);
                                    __m128 second = _mm_loadu_ps(input_row + input_x + 4u);
                                    __m128 input_values =
                                        _mm_shuffle_ps(first, second, _MM_SHUFFLE(2, 0, 2, 0));
                                    __m128 output_values = _mm_loadu_ps(output_row + output_x);
                                    output_values = _mm_add_ps(
                                        output_values, _mm_mul_ps(input_values, weight_values));
                                    _mm_storeu_ps(output_row + output_x, output_values);
                                }
                            }
                            for (; output_x < output_x_end; ++output_x) {
                                uint32_t input_x = output_x * 2u - 1u + kernel_x;
                                output_row[output_x] += input_row[input_x] * weight;
                            }
                        }
                    }
                }
            }
        }
    }
#else
    lw_scalar_conv3x3_stride2_pad1_f32(input, weights, bias, output, input_dimensions,
                                       output_dimensions);
#endif
}
