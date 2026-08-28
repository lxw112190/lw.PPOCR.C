#include "simd_kernels.h"

#include "scalar_kernels.h"

#include <stddef.h>

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
#  include <immintrin.h>
#  define LW_COMPILES_AVX2_CONV3X3 1
#else
#  define LW_COMPILES_AVX2_CONV3X3 0
#endif

#if LW_COMPILES_AVX2_CONV3X3
#  if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,no-fma")))
#  endif
static void
conv3x3_unit_pad1_four_outputs_f32(const float* input, const float* weights, const float* bias,
                                   float* output, const int32_t input_dimensions[4],
                                   const int32_t output_dimensions[4]) {
    const uint32_t input_channels = (uint32_t)input_dimensions[1];
    const uint32_t height = (uint32_t)input_dimensions[2];
    const uint32_t width = (uint32_t)input_dimensions[3];
    const uint32_t output_channels = (uint32_t)output_dimensions[1];
    const uint64_t channel_plane = (uint64_t)height * width;
    const uint64_t weights_per_output = (uint64_t)input_channels * 9u;
    uint32_t batch;
    for (batch = 0u; batch < (uint32_t)input_dimensions[0]; ++batch) {
        const float* batch_input =
            input + (size_t)((uint64_t)batch * input_channels * channel_plane);
        float* batch_output = output + (size_t)((uint64_t)batch * output_channels * channel_plane);
        uint32_t output_channel;
        for (output_channel = 0u; output_channel < output_channels; output_channel += 4u) {
            const float* weights_0 =
                weights + (size_t)((uint64_t)(output_channel + 0u) * weights_per_output);
            const float* weights_1 =
                weights + (size_t)((uint64_t)(output_channel + 1u) * weights_per_output);
            const float* weights_2 =
                weights + (size_t)((uint64_t)(output_channel + 2u) * weights_per_output);
            const float* weights_3 =
                weights + (size_t)((uint64_t)(output_channel + 3u) * weights_per_output);
            float* output_0 =
                batch_output + (size_t)((uint64_t)(output_channel + 0u) * channel_plane);
            float* output_1 =
                batch_output + (size_t)((uint64_t)(output_channel + 1u) * channel_plane);
            float* output_2 =
                batch_output + (size_t)((uint64_t)(output_channel + 2u) * channel_plane);
            float* output_3 =
                batch_output + (size_t)((uint64_t)(output_channel + 3u) * channel_plane);
            const float initial_0 = bias == NULL ? 0.0f : bias[output_channel + 0u];
            const float initial_1 = bias == NULL ? 0.0f : bias[output_channel + 1u];
            const float initial_2 = bias == NULL ? 0.0f : bias[output_channel + 2u];
            const float initial_3 = bias == NULL ? 0.0f : bias[output_channel + 3u];
            uint32_t output_y;
            for (output_y = 0u; output_y < height; ++output_y) {
                const size_t output_row_offset = (size_t)((uint64_t)output_y * width);
                uint32_t output_x = 0u;
                while (output_x < width) {
                    const int full_height = output_y != 0u && output_y + 1u < height;
                    const int full_width = output_x != 0u && output_x + 8u < width;
                    if (full_height && full_width) {
                        __m256 accumulators_0 = _mm256_set1_ps(initial_0);
                        __m256 accumulators_1 = _mm256_set1_ps(initial_1);
                        __m256 accumulators_2 = _mm256_set1_ps(initial_2);
                        __m256 accumulators_3 = _mm256_set1_ps(initial_3);
                        uint32_t input_channel;
                        for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
                            const float* input_channel_data =
                                batch_input + (size_t)((uint64_t)input_channel * channel_plane);
                            const size_t weight_offset = (size_t)((uint64_t)input_channel * 9u);
                            uint32_t kernel_y;
                            for (kernel_y = 0u; kernel_y < 3u; ++kernel_y) {
                                const uint32_t input_y = output_y + kernel_y - 1u;
                                const float* input_row =
                                    input_channel_data + (size_t)((uint64_t)input_y * width);
                                uint32_t kernel_x;
                                for (kernel_x = 0u; kernel_x < 3u; ++kernel_x) {
                                    const uint32_t input_x = output_x + kernel_x - 1u;
                                    const size_t weight_index =
                                        weight_offset + (size_t)kernel_y * 3u + kernel_x;
                                    const __m256 input_values =
                                        _mm256_loadu_ps(input_row + input_x);
                                    accumulators_0 = _mm256_add_ps(
                                        accumulators_0,
                                        _mm256_mul_ps(input_values,
                                                      _mm256_set1_ps(weights_0[weight_index])));
                                    accumulators_1 = _mm256_add_ps(
                                        accumulators_1,
                                        _mm256_mul_ps(input_values,
                                                      _mm256_set1_ps(weights_1[weight_index])));
                                    accumulators_2 = _mm256_add_ps(
                                        accumulators_2,
                                        _mm256_mul_ps(input_values,
                                                      _mm256_set1_ps(weights_2[weight_index])));
                                    accumulators_3 = _mm256_add_ps(
                                        accumulators_3,
                                        _mm256_mul_ps(input_values,
                                                      _mm256_set1_ps(weights_3[weight_index])));
                                }
                            }
                        }
                        _mm256_storeu_ps(output_0 + output_row_offset + output_x, accumulators_0);
                        _mm256_storeu_ps(output_1 + output_row_offset + output_x, accumulators_1);
                        _mm256_storeu_ps(output_2 + output_row_offset + output_x, accumulators_2);
                        _mm256_storeu_ps(output_3 + output_row_offset + output_x, accumulators_3);
                        output_x += 8u;
                    } else {
                        float accumulator_0 = initial_0;
                        float accumulator_1 = initial_1;
                        float accumulator_2 = initial_2;
                        float accumulator_3 = initial_3;
                        uint32_t input_channel;
                        for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
                            const float* input_channel_data =
                                batch_input + (size_t)((uint64_t)input_channel * channel_plane);
                            const size_t weight_offset = (size_t)((uint64_t)input_channel * 9u);
                            uint32_t kernel_y;
                            for (kernel_y = 0u; kernel_y < 3u; ++kernel_y) {
                                const int64_t input_y = (int64_t)output_y + kernel_y - 1;
                                uint32_t kernel_x;
                                if (input_y < 0 || input_y >= (int64_t)height) {
                                    continue;
                                }
                                for (kernel_x = 0u; kernel_x < 3u; ++kernel_x) {
                                    const int64_t input_x = (int64_t)output_x + kernel_x - 1;
                                    if (input_x >= 0 && input_x < (int64_t)width) {
                                        const size_t input_index =
                                            (size_t)((uint64_t)(uint32_t)input_y * width +
                                                     (uint32_t)input_x);
                                        const size_t weight_index =
                                            weight_offset + (size_t)kernel_y * 3u + kernel_x;
                                        const float value = input_channel_data[input_index];
                                        accumulator_0 += value * weights_0[weight_index];
                                        accumulator_1 += value * weights_1[weight_index];
                                        accumulator_2 += value * weights_2[weight_index];
                                        accumulator_3 += value * weights_3[weight_index];
                                    }
                                }
                            }
                        }
                        output_0[output_row_offset + output_x] = accumulator_0;
                        output_1[output_row_offset + output_x] = accumulator_1;
                        output_2[output_row_offset + output_x] = accumulator_2;
                        output_3[output_row_offset + output_x] = accumulator_3;
                        ++output_x;
                    }
                }
            }
        }
    }
}
#endif

#if LW_COMPILES_AVX2_CONV3X3 && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,no-fma")))
#endif
void lw_avx2_conv3x3_unit_pad1_f32(const float* input, const float* weights, const float* bias,
                                   float* output, const int32_t input_dimensions[4],
                                   const int32_t output_dimensions[4]) {
#if LW_COMPILES_AVX2_CONV3X3
    if ((uint32_t)output_dimensions[1] % 4u == 0u) {
        conv3x3_unit_pad1_four_outputs_f32(input, weights, bias, output, input_dimensions,
                                           output_dimensions);
        return;
    }
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
            __m256 initial_values = _mm256_set1_ps(initial);
            uint64_t spatial = 0u;
            uint32_t input_channel;
            for (; spatial + 8u <= channel_plane; spatial += 8u) {
                _mm256_storeu_ps(output_channel_data + (size_t)spatial, initial_values);
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
                        __m256 weight_values = _mm256_set1_ps(weight);
                        uint32_t output_y;
                        for (output_y = output_y_begin; output_y < output_y_end; ++output_y) {
                            uint32_t input_y = output_y + kernel_y - 1u;
                            const float* input_row =
                                input_channel_data + (size_t)((uint64_t)input_y * width);
                            float* output_row =
                                output_channel_data + (size_t)((uint64_t)output_y * width);
                            uint32_t output_x = output_x_begin;
                            for (; output_x + 8u <= output_x_end; output_x += 8u) {
                                uint32_t input_x = output_x + kernel_x - 1u;
                                __m256 input_values = _mm256_loadu_ps(input_row + input_x);
                                __m256 output_values = _mm256_loadu_ps(output_row + output_x);
                                output_values = _mm256_add_ps(
                                    output_values, _mm256_mul_ps(input_values, weight_values));
                                _mm256_storeu_ps(output_row + output_x, output_values);
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

#if LW_COMPILES_AVX2_CONV3X3
#  if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,no-fma")))
#  endif
static void
conv3x3_stride2_pad1_output_stream_f32(const float* input, const float* weights, const float* bias,
                                       float* output, const int32_t input_dimensions[4],
                                       const int32_t output_dimensions[4]) {
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
            __m256 initial_values = _mm256_set1_ps(initial);
            uint64_t spatial = 0u;
            uint32_t input_channel;
            for (; spatial + 8u <= output_channel_plane; spatial += 8u) {
                _mm256_storeu_ps(output_channel_data + (size_t)spatial, initial_values);
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
                        __m256 weight_values = _mm256_set1_ps(weight);
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
                            for (; output_x + 8u <= output_x_end; output_x += 8u) {
                                uint32_t input_x = output_x * 2u - 1u + kernel_x;
                                if ((uint64_t)input_x + 15u >= input_width) {
                                    break;
                                }
                                {
                                    __m256 first = _mm256_loadu_ps(input_row + input_x);
                                    __m256 second = _mm256_loadu_ps(input_row + input_x + 8u);
                                    __m128 first_even = _mm_shuffle_ps(
                                        _mm256_castps256_ps128(first),
                                        _mm256_extractf128_ps(first, 1), _MM_SHUFFLE(2, 0, 2, 0));
                                    __m128 second_even = _mm_shuffle_ps(
                                        _mm256_castps256_ps128(second),
                                        _mm256_extractf128_ps(second, 1), _MM_SHUFFLE(2, 0, 2, 0));
                                    __m256 input_values = _mm256_castps128_ps256(first_even);
                                    __m256 output_values;
                                    input_values =
                                        _mm256_insertf128_ps(input_values, second_even, 1);
                                    output_values = _mm256_loadu_ps(output_row + output_x);
                                    output_values = _mm256_add_ps(
                                        output_values, _mm256_mul_ps(input_values, weight_values));
                                    _mm256_storeu_ps(output_row + output_x, output_values);
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
}

#  if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,no-fma")))
#  endif
static void
conv3x3_stride2_pad1_four_outputs_f32(const float* input, const float* weights, const float* bias,
                                      float* output, const int32_t input_dimensions[4],
                                      const int32_t output_dimensions[4]) {
    const uint32_t input_channels = (uint32_t)input_dimensions[1];
    const uint32_t input_height = (uint32_t)input_dimensions[2];
    const uint32_t input_width = (uint32_t)input_dimensions[3];
    const uint32_t output_channels = (uint32_t)output_dimensions[1];
    const uint32_t output_height = (uint32_t)output_dimensions[2];
    const uint32_t output_width = (uint32_t)output_dimensions[3];
    const uint64_t input_channel_plane = (uint64_t)input_height * input_width;
    const uint64_t output_channel_plane = (uint64_t)output_height * output_width;
    const uint64_t weights_per_output = (uint64_t)input_channels * 9u;
    uint32_t batch;
    for (batch = 0u; batch < (uint32_t)input_dimensions[0]; ++batch) {
        const float* batch_input =
            input + (size_t)((uint64_t)batch * input_channels * input_channel_plane);
        float* batch_output =
            output + (size_t)((uint64_t)batch * output_channels * output_channel_plane);
        uint32_t output_channel;
        for (output_channel = 0u; output_channel < output_channels; output_channel += 4u) {
            const float* weights_0 =
                weights + (size_t)((uint64_t)(output_channel + 0u) * weights_per_output);
            const float* weights_1 =
                weights + (size_t)((uint64_t)(output_channel + 1u) * weights_per_output);
            const float* weights_2 =
                weights + (size_t)((uint64_t)(output_channel + 2u) * weights_per_output);
            const float* weights_3 =
                weights + (size_t)((uint64_t)(output_channel + 3u) * weights_per_output);
            float* output_0 =
                batch_output + (size_t)((uint64_t)(output_channel + 0u) * output_channel_plane);
            float* output_1 =
                batch_output + (size_t)((uint64_t)(output_channel + 1u) * output_channel_plane);
            float* output_2 =
                batch_output + (size_t)((uint64_t)(output_channel + 2u) * output_channel_plane);
            float* output_3 =
                batch_output + (size_t)((uint64_t)(output_channel + 3u) * output_channel_plane);
            const float initial_0 = bias == NULL ? 0.0f : bias[output_channel + 0u];
            const float initial_1 = bias == NULL ? 0.0f : bias[output_channel + 1u];
            const float initial_2 = bias == NULL ? 0.0f : bias[output_channel + 2u];
            const float initial_3 = bias == NULL ? 0.0f : bias[output_channel + 3u];
            uint32_t output_y;
            for (output_y = 0u; output_y < output_height; ++output_y) {
                const size_t output_row_offset = (size_t)((uint64_t)output_y * output_width);
                uint32_t output_x = 0u;
                while (output_x < output_width) {
                    const int full_height =
                        output_y != 0u && (uint64_t)output_y * 2u + 1u < input_height;
                    const int full_width = output_x != 0u && output_x + 8u <= output_width &&
                                           ((uint64_t)output_x + 7u) * 2u + 1u < input_width;
                    if (full_height && full_width) {
                        __m256 accumulators_0 = _mm256_set1_ps(initial_0);
                        __m256 accumulators_1 = _mm256_set1_ps(initial_1);
                        __m256 accumulators_2 = _mm256_set1_ps(initial_2);
                        __m256 accumulators_3 = _mm256_set1_ps(initial_3);
                        uint32_t input_channel;
                        for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
                            const float* input_channel_data =
                                batch_input +
                                (size_t)((uint64_t)input_channel * input_channel_plane);
                            const size_t weight_offset = (size_t)((uint64_t)input_channel * 9u);
                            uint32_t kernel_y;
                            for (kernel_y = 0u; kernel_y < 3u; ++kernel_y) {
                                const uint32_t input_y = output_y * 2u - 1u + kernel_y;
                                const float* input_row =
                                    input_channel_data + (size_t)((uint64_t)input_y * input_width);
                                uint32_t kernel_x;
                                for (kernel_x = 0u; kernel_x < 3u; ++kernel_x) {
                                    const uint32_t input_x = output_x * 2u - 1u + kernel_x;
                                    __m256 first = _mm256_loadu_ps(input_row + input_x);
                                    __m256 second = _mm256_loadu_ps(input_row + input_x + 8u);
                                    __m128 first_even = _mm_shuffle_ps(
                                        _mm256_castps256_ps128(first),
                                        _mm256_extractf128_ps(first, 1), _MM_SHUFFLE(2, 0, 2, 0));
                                    __m128 second_even = _mm_shuffle_ps(
                                        _mm256_castps256_ps128(second),
                                        _mm256_extractf128_ps(second, 1), _MM_SHUFFLE(2, 0, 2, 0));
                                    __m256 input_values = _mm256_castps128_ps256(first_even);
                                    const size_t weight_index =
                                        weight_offset + (size_t)kernel_y * 3u + kernel_x;
                                    input_values =
                                        _mm256_insertf128_ps(input_values, second_even, 1);
                                    accumulators_0 = _mm256_add_ps(
                                        accumulators_0,
                                        _mm256_mul_ps(input_values,
                                                      _mm256_set1_ps(weights_0[weight_index])));
                                    accumulators_1 = _mm256_add_ps(
                                        accumulators_1,
                                        _mm256_mul_ps(input_values,
                                                      _mm256_set1_ps(weights_1[weight_index])));
                                    accumulators_2 = _mm256_add_ps(
                                        accumulators_2,
                                        _mm256_mul_ps(input_values,
                                                      _mm256_set1_ps(weights_2[weight_index])));
                                    accumulators_3 = _mm256_add_ps(
                                        accumulators_3,
                                        _mm256_mul_ps(input_values,
                                                      _mm256_set1_ps(weights_3[weight_index])));
                                }
                            }
                        }
                        _mm256_storeu_ps(output_0 + output_row_offset + output_x, accumulators_0);
                        _mm256_storeu_ps(output_1 + output_row_offset + output_x, accumulators_1);
                        _mm256_storeu_ps(output_2 + output_row_offset + output_x, accumulators_2);
                        _mm256_storeu_ps(output_3 + output_row_offset + output_x, accumulators_3);
                        output_x += 8u;
                    } else {
                        float accumulator_0 = initial_0;
                        float accumulator_1 = initial_1;
                        float accumulator_2 = initial_2;
                        float accumulator_3 = initial_3;
                        uint32_t input_channel;
                        for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
                            const float* input_channel_data =
                                batch_input +
                                (size_t)((uint64_t)input_channel * input_channel_plane);
                            const size_t weight_offset = (size_t)((uint64_t)input_channel * 9u);
                            uint32_t kernel_y;
                            for (kernel_y = 0u; kernel_y < 3u; ++kernel_y) {
                                const int64_t input_y = (int64_t)output_y * 2 - 1 + kernel_y;
                                uint32_t kernel_x;
                                if (input_y < 0 || input_y >= input_height) {
                                    continue;
                                }
                                for (kernel_x = 0u; kernel_x < 3u; ++kernel_x) {
                                    const int64_t input_x = (int64_t)output_x * 2 - 1 + kernel_x;
                                    if (input_x >= 0 && input_x < input_width) {
                                        const uint64_t input_offset =
                                            (uint64_t)(uint32_t)input_y * input_width +
                                            (uint32_t)input_x;
                                        const float value =
                                            input_channel_data[(size_t)input_offset];
                                        const size_t weight_index =
                                            weight_offset + (size_t)kernel_y * 3u + kernel_x;
                                        accumulator_0 += value * weights_0[weight_index];
                                        accumulator_1 += value * weights_1[weight_index];
                                        accumulator_2 += value * weights_2[weight_index];
                                        accumulator_3 += value * weights_3[weight_index];
                                    }
                                }
                            }
                        }
                        output_0[output_row_offset + output_x] = accumulator_0;
                        output_1[output_row_offset + output_x] = accumulator_1;
                        output_2[output_row_offset + output_x] = accumulator_2;
                        output_3[output_row_offset + output_x] = accumulator_3;
                        ++output_x;
                    }
                }
            }
        }
    }
}
#endif

#if LW_COMPILES_AVX2_CONV3X3 && (defined(__GNUC__) || defined(__clang__))
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
    if ((uint32_t)output_dimensions[1] % 4u == 0u) {
        conv3x3_stride2_pad1_four_outputs_f32(input, weights, bias, output, input_dimensions,
                                              output_dimensions);
        return;
    }
    if (input_dimensions[1] > 4) {
        conv3x3_stride2_pad1_output_stream_f32(input, weights, bias, output, input_dimensions,
                                               output_dimensions);
        return;
    }
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
            uint32_t output_y;
            for (output_y = 0u; output_y < output_height; ++output_y) {
                float* output_row =
                    output_channel_data + (size_t)((uint64_t)output_y * output_width);
                uint32_t output_x = 0u;
                while (output_x < output_width) {
                    int full_height = output_y != 0u && (uint64_t)output_y * 2u + 1u < input_height;
                    int full_width = output_x != 0u && output_x + 8u <= output_width &&
                                     ((uint64_t)output_x + 7u) * 2u + 1u < input_width;
                    if (full_height && full_width) {
                        __m256 accumulators = _mm256_set1_ps(initial);
                        uint32_t input_channel;
                        for (input_channel = 0u; input_channel < (uint32_t)input_dimensions[1];
                             ++input_channel) {
                            const float* input_channel_data =
                                batch_input +
                                (size_t)((uint64_t)input_channel * input_channel_plane);
                            const float* weight_channel_data =
                                output_channel_weights + (size_t)((uint64_t)input_channel * 9u);
                            uint32_t kernel_y;
                            for (kernel_y = 0u; kernel_y < 3u; ++kernel_y) {
                                uint32_t input_y = output_y * 2u - 1u + kernel_y;
                                const float* input_row =
                                    input_channel_data + (size_t)((uint64_t)input_y * input_width);
                                uint32_t kernel_x;
                                for (kernel_x = 0u; kernel_x < 3u; ++kernel_x) {
                                    uint32_t input_x = output_x * 2u - 1u + kernel_x;
                                    __m256 first = _mm256_loadu_ps(input_row + input_x);
                                    __m256 second = _mm256_loadu_ps(input_row + input_x + 8u);
                                    __m128 first_even = _mm_shuffle_ps(
                                        _mm256_castps256_ps128(first),
                                        _mm256_extractf128_ps(first, 1), _MM_SHUFFLE(2, 0, 2, 0));
                                    __m128 second_even = _mm_shuffle_ps(
                                        _mm256_castps256_ps128(second),
                                        _mm256_extractf128_ps(second, 1), _MM_SHUFFLE(2, 0, 2, 0));
                                    __m256 input_values = _mm256_castps128_ps256(first_even);
                                    __m256 weight_values = _mm256_set1_ps(
                                        weight_channel_data[(size_t)kernel_y * 3u + kernel_x]);
                                    input_values =
                                        _mm256_insertf128_ps(input_values, second_even, 1);
                                    accumulators = _mm256_add_ps(
                                        accumulators, _mm256_mul_ps(input_values, weight_values));
                                }
                            }
                        }
                        _mm256_storeu_ps(output_row + output_x, accumulators);
                        output_x += 8u;
                    } else {
                        float accumulator = initial;
                        uint32_t input_channel;
                        for (input_channel = 0u; input_channel < (uint32_t)input_dimensions[1];
                             ++input_channel) {
                            const float* input_channel_data =
                                batch_input +
                                (size_t)((uint64_t)input_channel * input_channel_plane);
                            const float* weight_channel_data =
                                output_channel_weights + (size_t)((uint64_t)input_channel * 9u);
                            uint32_t kernel_y;
                            for (kernel_y = 0u; kernel_y < 3u; ++kernel_y) {
                                int64_t input_y = (int64_t)output_y * 2 - 1 + kernel_y;
                                uint32_t kernel_x;
                                if (input_y < 0 || input_y >= input_height) {
                                    continue;
                                }
                                for (kernel_x = 0u; kernel_x < 3u; ++kernel_x) {
                                    int64_t input_x = (int64_t)output_x * 2 - 1 + kernel_x;
                                    if (input_x >= 0 && input_x < input_width) {
                                        uint64_t input_offset =
                                            (uint64_t)(uint32_t)input_y * input_width +
                                            (uint32_t)input_x;
                                        accumulator +=
                                            input_channel_data[(size_t)input_offset] *
                                            weight_channel_data[(size_t)kernel_y * 3u + kernel_x];
                                    }
                                }
                            }
                        }
                        output_row[output_x++] = accumulator;
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
