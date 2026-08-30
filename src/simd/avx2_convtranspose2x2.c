#include "simd_kernels.h"

#include "scalar_kernels.h"

#include <stddef.h>

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
#  include <immintrin.h>
#  define LW_COMPILES_AVX2_CONV_TRANSPOSE2X2 1
#else
#  define LW_COMPILES_AVX2_CONV_TRANSPOSE2X2 0
#endif

#if LW_COMPILES_AVX2_CONV_TRANSPOSE2X2 && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,no-fma")))
#endif
void lw_avx2_conv_transpose2x2_stride2_f32(
    const float* input, const float* weights, const float* bias, float* output,
    const int32_t input_dimensions[4], const int32_t output_dimensions[4]) {
#if LW_COMPILES_AVX2_CONV_TRANSPOSE2X2
    uint32_t input_channels = (uint32_t)input_dimensions[1];
    uint32_t output_channels = (uint32_t)output_dimensions[1];
    uint32_t input_height = (uint32_t)input_dimensions[2];
    uint32_t input_width = (uint32_t)input_dimensions[3];
    uint32_t output_width = (uint32_t)output_dimensions[3];
    uint64_t input_plane = (uint64_t)input_height * input_width;
    uint64_t output_plane = (uint64_t)(uint32_t)output_dimensions[2] * output_width;
    uint32_t batch;

    /* With kernel=2, stride=2 and zero padding, each input point writes one
     * non-overlapping 2x2 output tile.  Eight adjacent input values expand to
     * two contiguous AVX output vectors per tile row, avoiding gathers and
     * preserving each output element's input-channel accumulation order. */
    for (batch = 0u; batch < (uint32_t)input_dimensions[0]; ++batch) {
        const float* batch_input =
            input + (size_t)((uint64_t)batch * input_channels * input_plane);
        float* batch_output =
            output + (size_t)((uint64_t)batch * output_channels * output_plane);
        uint32_t output_channel;
        for (output_channel = 0u; output_channel < output_channels; ++output_channel) {
            float* output_channel_data =
                batch_output + (size_t)((uint64_t)output_channel * output_plane);
            __m256 initial = _mm256_set1_ps(bias == NULL ? 0.0f : bias[output_channel]);
            const float initial_scalar = bias == NULL ? 0.0f : bias[output_channel];
            const __m256 zero = _mm256_setzero_ps();
            uint32_t input_channel;

            for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
                const float* input_channel_data =
                    batch_input + (size_t)((uint64_t)input_channel * input_plane);
                const float* channel_weights =
                    weights + (size_t)(((uint64_t)input_channel * output_channels + output_channel) * 4u);
                const __m256 weight0 = _mm256_set1_ps(channel_weights[0]);
                const __m256 weight1 = _mm256_set1_ps(channel_weights[1]);
                const __m256 weight2 = _mm256_set1_ps(channel_weights[2]);
                const __m256 weight3 = _mm256_set1_ps(channel_weights[3]);
                uint32_t input_y;
                for (input_y = 0u; input_y < input_height; ++input_y) {
                    const float* input_row =
                        input_channel_data + (size_t)((uint64_t)input_y * input_width);
                    float* output_row0 =
                        output_channel_data + (size_t)((uint64_t)(input_y * 2u) * output_width);
                    float* output_row1 = output_row0 + output_width;
                    uint32_t input_x = 0u;
                    for (; input_x + 8u <= input_width; input_x += 8u) {
                        __m256 values = _mm256_loadu_ps(input_row + input_x);
                        __m256 even_lo_pairs = _mm256_unpacklo_ps(values, zero);
                        __m256 even_hi_pairs = _mm256_unpackhi_ps(values, zero);
                        __m256 odd_lo_pairs = _mm256_unpacklo_ps(zero, values);
                        __m256 odd_hi_pairs = _mm256_unpackhi_ps(zero, values);
                        /* AVX unpack is lane-local: combine the two 128-bit
                         * halves so vectors hold x[0..3] then x[4..7]. */
                        __m256 even_low =
                            _mm256_permute2f128_ps(even_lo_pairs, even_hi_pairs, 0x20);
                        __m256 even_high =
                            _mm256_permute2f128_ps(even_lo_pairs, even_hi_pairs, 0x31);
                        __m256 odd_low =
                            _mm256_permute2f128_ps(odd_lo_pairs, odd_hi_pairs, 0x20);
                        __m256 odd_high =
                            _mm256_permute2f128_ps(odd_lo_pairs, odd_hi_pairs, 0x31);
                        uint32_t output_x = input_x * 2u;
                        __m256 row0_low = input_channel == 0u
                                              ? initial
                                              : _mm256_loadu_ps(output_row0 + output_x);
                        __m256 row0_high = input_channel == 0u
                                               ? initial
                                               : _mm256_loadu_ps(output_row0 + output_x + 8u);
                        __m256 row1_low = input_channel == 0u
                                              ? initial
                                              : _mm256_loadu_ps(output_row1 + output_x);
                        __m256 row1_high = input_channel == 0u
                                               ? initial
                                               : _mm256_loadu_ps(output_row1 + output_x + 8u);

                        row0_low =
                            _mm256_add_ps(row0_low, _mm256_mul_ps(even_low, weight0));
                        row0_high =
                            _mm256_add_ps(row0_high, _mm256_mul_ps(even_high, weight0));
                        row0_low = _mm256_add_ps(row0_low, _mm256_mul_ps(odd_low, weight1));
                        row0_high = _mm256_add_ps(row0_high, _mm256_mul_ps(odd_high, weight1));
                        row1_low =
                            _mm256_add_ps(row1_low, _mm256_mul_ps(even_low, weight2));
                        row1_high =
                            _mm256_add_ps(row1_high, _mm256_mul_ps(even_high, weight2));
                        row1_low = _mm256_add_ps(row1_low, _mm256_mul_ps(odd_low, weight3));
                        row1_high = _mm256_add_ps(row1_high, _mm256_mul_ps(odd_high, weight3));

                        _mm256_storeu_ps(output_row0 + output_x, row0_low);
                        _mm256_storeu_ps(output_row0 + output_x + 8u, row0_high);
                        _mm256_storeu_ps(output_row1 + output_x, row1_low);
                        _mm256_storeu_ps(output_row1 + output_x + 8u, row1_high);
                    }
                    for (; input_x < input_width; ++input_x) {
                        float value = input_row[input_x];
                        uint32_t output_x = input_x * 2u;
                        float row00 =
                            input_channel == 0u ? initial_scalar : output_row0[output_x];
                        float row01 =
                            input_channel == 0u ? initial_scalar : output_row0[output_x + 1u];
                        float row10 =
                            input_channel == 0u ? initial_scalar : output_row1[output_x];
                        float row11 =
                            input_channel == 0u ? initial_scalar : output_row1[output_x + 1u];
                        output_row0[output_x] = row00 + value * channel_weights[0];
                        output_row0[output_x + 1u] = row01 + value * channel_weights[1];
                        output_row1[output_x] = row10 + value * channel_weights[2];
                        output_row1[output_x + 1u] = row11 + value * channel_weights[3];
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
