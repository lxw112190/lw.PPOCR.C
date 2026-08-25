#include "simd_kernels.h"

#include "scalar_kernels.h"

#include <stddef.h>

#if defined(_M_IX86) || defined(_M_X64) || \
    defined(__i386__) || defined(__x86_64__)
#  include <immintrin.h>
#  define LW_COMPILES_AVX2_CONV3X3 1
#else
#  define LW_COMPILES_AVX2_CONV3X3 0
#endif

#if LW_COMPILES_AVX2_CONV3X3 && \
    (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,no-fma")))
#endif
void lw_avx2_conv3x3_stride2_pad1_f32(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    const int32_t input_dimensions[4],
    const int32_t output_dimensions[4]) {
#if LW_COMPILES_AVX2_CONV3X3
    uint32_t input_height = (uint32_t)input_dimensions[2];
    uint32_t input_width = (uint32_t)input_dimensions[3];
    uint32_t output_height = (uint32_t)output_dimensions[2];
    uint32_t output_width = (uint32_t)output_dimensions[3];
    uint64_t input_channel_plane = (uint64_t)input_height * input_width;
    uint64_t output_channel_plane = (uint64_t)output_height * output_width;
    uint32_t batch;
    for (batch = 0u; batch < (uint32_t)input_dimensions[0]; ++batch) {
        const float* batch_input = input + (size_t)(
            (uint64_t)batch * (uint32_t)input_dimensions[1] *
            input_channel_plane);
        float* batch_output = output + (size_t)(
            (uint64_t)batch * (uint32_t)output_dimensions[1] *
            output_channel_plane);
        uint32_t output_channel;
        for (output_channel = 0u;
             output_channel < (uint32_t)output_dimensions[1];
             ++output_channel) {
            const float* output_channel_weights = weights + (size_t)(
                (uint64_t)output_channel *
                (uint32_t)input_dimensions[1] * 9u);
            float* output_channel_data = batch_output + (size_t)(
                (uint64_t)output_channel * output_channel_plane);
            float initial = bias == NULL ? 0.0f : bias[output_channel];
            __m256 initial_values = _mm256_set1_ps(initial);
            uint64_t spatial = 0u;
            uint32_t input_channel;
            for (; spatial + 8u <= output_channel_plane; spatial += 8u) {
                _mm256_storeu_ps(output_channel_data + (size_t)spatial,
                                 initial_values);
            }
            for (; spatial < output_channel_plane; ++spatial) {
                output_channel_data[(size_t)spatial] = initial;
            }
            for (input_channel = 0u;
                 input_channel < (uint32_t)input_dimensions[1];
                 ++input_channel) {
                const float* input_channel_data = batch_input + (size_t)(
                    (uint64_t)input_channel * input_channel_plane);
                const float* weight_channel_data = output_channel_weights +
                    (size_t)((uint64_t)input_channel * 9u);
                uint32_t kernel_y;
                for (kernel_y = 0u; kernel_y < 3u; ++kernel_y) {
                    uint32_t output_y_begin = kernel_y == 0u ? 1u : 0u;
                    uint32_t output_y_end = output_height;
                    uint32_t kernel_x;
                    if (kernel_y == 2u &&
                        (uint64_t)(output_y_end - 1u) * 2u + 1u >=
                            input_height) {
                        --output_y_end;
                    }
                    for (kernel_x = 0u; kernel_x < 3u; ++kernel_x) {
                        uint32_t output_x_begin = kernel_x == 0u ? 1u : 0u;
                        uint32_t output_x_end = output_width;
                        float weight = weight_channel_data[
                            (size_t)kernel_y * 3u + kernel_x];
                        __m256 weight_values = _mm256_set1_ps(weight);
                        uint32_t output_y;
                        if (kernel_x == 2u &&
                            (uint64_t)(output_x_end - 1u) * 2u + 1u >=
                                input_width) {
                            --output_x_end;
                        }
                        for (output_y = output_y_begin;
                             output_y < output_y_end; ++output_y) {
                            uint32_t input_y = output_y * 2u - 1u + kernel_y;
                            const float* input_row = input_channel_data +
                                (size_t)((uint64_t)input_y * input_width);
                            float* output_row = output_channel_data +
                                (size_t)((uint64_t)output_y * output_width);
                            uint32_t output_x = output_x_begin;
                            for (; output_x + 8u <= output_x_end; output_x += 8u) {
                                uint32_t input_x =
                                    output_x * 2u - 1u + kernel_x;
                                if ((uint64_t)input_x + 15u >= input_width) {
                                    break;
                                }
                                {
                                    __m256 first =
                                        _mm256_loadu_ps(input_row + input_x);
                                    __m256 second =
                                        _mm256_loadu_ps(input_row + input_x + 8u);
                                    __m128 first_even = _mm_shuffle_ps(
                                        _mm256_castps256_ps128(first),
                                        _mm256_extractf128_ps(first, 1),
                                        _MM_SHUFFLE(2, 0, 2, 0));
                                    __m128 second_even = _mm_shuffle_ps(
                                        _mm256_castps256_ps128(second),
                                        _mm256_extractf128_ps(second, 1),
                                        _MM_SHUFFLE(2, 0, 2, 0));
                                    __m256 input_values =
                                        _mm256_castps128_ps256(first_even);
                                    __m256 output_values;
                                    input_values = _mm256_insertf128_ps(
                                        input_values, second_even, 1);
                                    output_values = _mm256_loadu_ps(
                                        output_row + output_x);
                                    output_values = _mm256_add_ps(
                                        output_values,
                                        _mm256_mul_ps(input_values, weight_values));
                                    _mm256_storeu_ps(
                                        output_row + output_x, output_values);
                                }
                            }
                            for (; output_x < output_x_end; ++output_x) {
                                uint32_t input_x =
                                    output_x * 2u - 1u + kernel_x;
                                output_row[output_x] +=
                                    input_row[input_x] * weight;
                            }
                        }
                    }
                }
            }
        }
    }
#else
    lw_scalar_conv3x3_stride2_pad1_f32(
        input, weights, bias, output, input_dimensions, output_dimensions);
#endif
}
