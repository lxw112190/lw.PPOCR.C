#include "simd_kernels.h"
#include "simd_platform.h"

#include <stddef.h>

#define LW_COMPILES_SSE2_DEPTHWISE3X3 LW_SIMD_HAS_SSE2_INTRINSICS

LW_SIMD_SSE2_TARGET
void lw_sse2_depthwise_conv3x3_unit_pad1_f32(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    const int32_t dimensions[4]) {
#if LW_COMPILES_SSE2_DEPTHWISE3X3
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
            __m128 initial_values = _mm_set1_ps(initial);
            uint64_t spatial = 0u;
            uint32_t kernel_y;
            for (; spatial + 4u <= channel_plane; spatial += 4u) {
                _mm_storeu_ps(output_channel + (size_t)spatial, initial_values);
            }
            for (; spatial < channel_plane; ++spatial) {
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
                    __m128 weight_values = _mm_set1_ps(weight);
                    uint32_t output_y;
                    for (output_y = output_y_begin; output_y < output_y_end; ++output_y) {
                        uint32_t input_y = output_y + kernel_y - 1u;
                        const float* input_row =
                            input_channel + (size_t)((uint64_t)input_y * width);
                        float* output_row = output_channel + (size_t)((uint64_t)output_y * width);
                        uint32_t output_x = output_x_begin;
                        for (; output_x + 4u <= output_x_end; output_x += 4u) {
                            uint32_t input_x = output_x + kernel_x - 1u;
                            __m128 output_values = _mm_loadu_ps(output_row + output_x);
                            __m128 input_values = _mm_loadu_ps(input_row + input_x);
                            output_values =
                                _mm_add_ps(output_values, _mm_mul_ps(input_values, weight_values));
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
#else
    lw_scalar_depthwise_conv3x3_unit_pad1_f32(input, weights, bias, output, dimensions);
#endif
}
