#include "simd_kernels.h"

#include "scalar_kernels.h"

#include <stddef.h>

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
#  include <immintrin.h>
#  define LW_COMPILES_AVX2_CONV2X2 1
#else
#  define LW_COMPILES_AVX2_CONV2X2 0
#endif

#if LW_COMPILES_AVX2_CONV2X2 && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,no-fma")))
#endif
void lw_avx2_conv2x2_unit_pad_end1_f32(const float* input, const float* weights, const float* bias,
                                       float* output, const int32_t input_dimensions[4],
                                       const int32_t output_dimensions[4]) {
#if LW_COMPILES_AVX2_CONV2X2
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
                    output_channel_weights + (size_t)((uint64_t)input_channel * 4u);
                uint32_t kernel_y;
                for (kernel_y = 0u; kernel_y < 2u; ++kernel_y) {
                    uint32_t output_y_end = height - kernel_y;
                    uint32_t kernel_x;
                    for (kernel_x = 0u; kernel_x < 2u; ++kernel_x) {
                        uint32_t output_x_end = width - kernel_x;
                        __m256 weight_values =
                            _mm256_set1_ps(weight_channel_data[kernel_y * 2u + kernel_x]);
                        uint32_t output_y;
                        for (output_y = 0u; output_y < output_y_end; ++output_y) {
                            const float* input_row =
                                input_channel_data +
                                (size_t)((uint64_t)(output_y + kernel_y) * width + kernel_x);
                            float* output_row =
                                output_channel_data + (size_t)((uint64_t)output_y * width);
                            uint32_t output_x = 0u;
                            for (; output_x + 8u <= output_x_end; output_x += 8u) {
                                __m256 input_values = _mm256_loadu_ps(input_row + output_x);
                                __m256 output_values = _mm256_loadu_ps(output_row + output_x);
                                output_values = _mm256_add_ps(
                                    output_values, _mm256_mul_ps(input_values, weight_values));
                                _mm256_storeu_ps(output_row + output_x, output_values);
                            }
                            for (; output_x < output_x_end; ++output_x) {
                                output_row[output_x] +=
                                    input_row[output_x] *
                                    weight_channel_data[kernel_y * 2u + kernel_x];
                            }
                        }
                    }
                }
            }
        }
    }
#else
    lw_scalar_conv2x2_unit_pad_end1_f32(input, weights, bias, output, input_dimensions,
                                        output_dimensions);
#endif
}
