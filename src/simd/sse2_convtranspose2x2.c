#include "simd_kernels.h"
#include "simd_platform.h"

#include "scalar_kernels.h"

#include <stddef.h>

#define LW_COMPILES_SSE2_CONV_TRANSPOSE2X2 LW_SIMD_HAS_SSE2_INTRINSICS

LW_SIMD_SSE2_TARGET
void lw_sse2_conv_transpose2x2_stride2_f32(
    const float* input, const float* weights, const float* bias, float* output,
    const int32_t input_dimensions[4], const int32_t output_dimensions[4]) {
#if LW_COMPILES_SSE2_CONV_TRANSPOSE2X2
    uint32_t input_channels = (uint32_t)input_dimensions[1];
    uint32_t output_channels = (uint32_t)output_dimensions[1];
    uint32_t input_height = (uint32_t)input_dimensions[2];
    uint32_t input_width = (uint32_t)input_dimensions[3];
    uint32_t output_width = (uint32_t)output_dimensions[3];
    uint64_t input_plane = (uint64_t)input_height * input_width;
    uint64_t output_plane = (uint64_t)(uint32_t)output_dimensions[2] * output_width;
    uint32_t batch;

    /* The AVX2 implementation handles eight values at once.  This SSE2
     * counterpart keeps the identical no-overlap 2x2 expansion strategy for
     * CPUs that expose SSE2 but not AVX2. */
    for (batch = 0u; batch < (uint32_t)input_dimensions[0]; ++batch) {
        const float* batch_input =
            input + (size_t)((uint64_t)batch * input_channels * input_plane);
        float* batch_output =
            output + (size_t)((uint64_t)batch * output_channels * output_plane);
        uint32_t output_channel;
        for (output_channel = 0u; output_channel < output_channels; ++output_channel) {
            float* output_channel_data =
                batch_output + (size_t)((uint64_t)output_channel * output_plane);
            float initial_value = bias == NULL ? 0.0f : bias[output_channel];
            __m128 initial = _mm_set1_ps(initial_value);
            uint64_t output_index = 0u;
            uint32_t input_channel;
            for (; output_index + 4u <= output_plane; output_index += 4u) {
                _mm_storeu_ps(output_channel_data + (size_t)output_index, initial);
            }
            for (; output_index < output_plane; ++output_index) {
                output_channel_data[(size_t)output_index] = initial_value;
            }

            for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
                const float* input_channel_data =
                    batch_input + (size_t)((uint64_t)input_channel * input_plane);
                const float* channel_weights =
                    weights + (size_t)(((uint64_t)input_channel * output_channels + output_channel) * 4u);
                uint32_t input_y;
                for (input_y = 0u; input_y < input_height; ++input_y) {
                    const float* input_row =
                        input_channel_data + (size_t)((uint64_t)input_y * input_width);
                    float* output_row0 =
                        output_channel_data + (size_t)((uint64_t)(input_y * 2u) * output_width);
                    float* output_row1 = output_row0 + output_width;
                    uint32_t input_x = 0u;
                    for (; input_x + 4u <= input_width; input_x += 4u) {
                        __m128 values = _mm_loadu_ps(input_row + input_x);
                        __m128 zero = _mm_setzero_ps();
                        __m128 even = _mm_unpacklo_ps(values, zero);
                        __m128 even_high = _mm_unpackhi_ps(values, zero);
                        __m128 odd = _mm_unpacklo_ps(zero, values);
                        __m128 odd_high = _mm_unpackhi_ps(zero, values);
                        uint32_t output_x = input_x * 2u;
                        __m128 product;

                        product = _mm_mul_ps(even, _mm_set1_ps(channel_weights[0]));
                        _mm_storeu_ps(output_row0 + output_x,
                                      _mm_add_ps(_mm_loadu_ps(output_row0 + output_x), product));
                        product = _mm_mul_ps(even_high, _mm_set1_ps(channel_weights[0]));
                        _mm_storeu_ps(output_row0 + output_x + 4u,
                                      _mm_add_ps(_mm_loadu_ps(output_row0 + output_x + 4u), product));

                        product = _mm_mul_ps(odd, _mm_set1_ps(channel_weights[1]));
                        _mm_storeu_ps(output_row0 + output_x,
                                      _mm_add_ps(_mm_loadu_ps(output_row0 + output_x), product));
                        product = _mm_mul_ps(odd_high, _mm_set1_ps(channel_weights[1]));
                        _mm_storeu_ps(output_row0 + output_x + 4u,
                                      _mm_add_ps(_mm_loadu_ps(output_row0 + output_x + 4u), product));

                        product = _mm_mul_ps(even, _mm_set1_ps(channel_weights[2]));
                        _mm_storeu_ps(output_row1 + output_x,
                                      _mm_add_ps(_mm_loadu_ps(output_row1 + output_x), product));
                        product = _mm_mul_ps(even_high, _mm_set1_ps(channel_weights[2]));
                        _mm_storeu_ps(output_row1 + output_x + 4u,
                                      _mm_add_ps(_mm_loadu_ps(output_row1 + output_x + 4u), product));

                        product = _mm_mul_ps(odd, _mm_set1_ps(channel_weights[3]));
                        _mm_storeu_ps(output_row1 + output_x,
                                      _mm_add_ps(_mm_loadu_ps(output_row1 + output_x), product));
                        product = _mm_mul_ps(odd_high, _mm_set1_ps(channel_weights[3]));
                        _mm_storeu_ps(output_row1 + output_x + 4u,
                                      _mm_add_ps(_mm_loadu_ps(output_row1 + output_x + 4u), product));
                    }
                    for (; input_x < input_width; ++input_x) {
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
#else
    (void)input;
    (void)weights;
    (void)bias;
    (void)output;
    (void)input_dimensions;
    (void)output_dimensions;
#endif
}
