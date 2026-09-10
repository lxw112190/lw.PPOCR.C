#include "simd_kernels.h"

#include "packed_conv_internal.h"

#include <stddef.h>

#if defined(_M_X64) || defined(__x86_64__)
#  include <immintrin.h>
#  define LW_COMPILES_AVX2_FMA_PACKED_CONV1X1_8X8 1
#else
#  define LW_COMPILES_AVX2_FMA_PACKED_CONV1X1_8X8 0
#endif

#if LW_COMPILES_AVX2_FMA_PACKED_CONV1X1_8X8 && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,fma")))
#endif
void lw_avx2_fma_packed_conv1x1_8x8_f32(
    const float* input, const float* packed_weights, const float* bias, float* output,
    const int32_t input_dimensions[4], const int32_t output_dimensions[4]) {
#if LW_COMPILES_AVX2_FMA_PACKED_CONV1X1_8X8
    const uint32_t input_channels = (uint32_t)input_dimensions[1];
    const uint32_t output_channels = (uint32_t)output_dimensions[1];
    const uint64_t channel_plane =
        (uint64_t)(uint32_t)input_dimensions[2] * (uint32_t)input_dimensions[3];
    const uint32_t batch_count = (uint32_t)input_dimensions[0];
    uint32_t batch;

    /* The candidate owns complete pairs of packed 4-output blocks. Keep all
     * other shapes on the validated 4-output FMA kernel. */
    if (input == NULL || packed_weights == NULL || output == NULL ||
        output_channels < 8u || (output_channels & 7u) != 0u || channel_plane < 8u) {
        lw_avx2_fma_packed_conv1x1_f32(input, packed_weights, bias, output,
                                       input_dimensions, output_dimensions);
        return;
    }

    for (batch = 0u; batch < batch_count; ++batch) {
        const float* batch_input =
            input + (size_t)((uint64_t)batch * input_channels * channel_plane);
        float* batch_output =
            output + (size_t)((uint64_t)batch * output_channels * channel_plane);
        uint32_t output_base;
        for (output_base = 0u; output_base < output_channels; output_base += 8u) {
            const uint32_t packed_block = output_base / LW_PACKED_CONV1X1_OUTPUT_TILE;
            const float* packed0 =
                packed_weights +
                (size_t)((uint64_t)packed_block * input_channels *
                          LW_PACKED_CONV1X1_OUTPUT_TILE);
            const float* packed1 =
                packed0 + (size_t)((uint64_t)input_channels *
                                   LW_PACKED_CONV1X1_OUTPUT_TILE);
            uint64_t spatial = 0u;
            for (; spatial + 8u <= channel_plane; spatial += 8u) {
                __m256 accumulator0 = _mm256_set1_ps(bias == NULL ? 0.0f : bias[output_base]);
                __m256 accumulator1 = _mm256_set1_ps(bias == NULL ? 0.0f : bias[output_base + 1u]);
                __m256 accumulator2 = _mm256_set1_ps(bias == NULL ? 0.0f : bias[output_base + 2u]);
                __m256 accumulator3 = _mm256_set1_ps(bias == NULL ? 0.0f : bias[output_base + 3u]);
                __m256 accumulator4 = _mm256_set1_ps(bias == NULL ? 0.0f : bias[output_base + 4u]);
                __m256 accumulator5 = _mm256_set1_ps(bias == NULL ? 0.0f : bias[output_base + 5u]);
                __m256 accumulator6 = _mm256_set1_ps(bias == NULL ? 0.0f : bias[output_base + 6u]);
                __m256 accumulator7 = _mm256_set1_ps(bias == NULL ? 0.0f : bias[output_base + 7u]);
                const float* input_ptr = batch_input + (size_t)spatial;
                const float* packed_ptr0 = packed0;
                const float* packed_ptr1 = packed1;
                uint32_t input_channel;
                for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
                    const __m256 input_values = _mm256_loadu_ps(input_ptr);
                    accumulator0 = _mm256_fmadd_ps(input_values, _mm256_set1_ps(packed_ptr0[0]), accumulator0);
                    accumulator1 = _mm256_fmadd_ps(input_values, _mm256_set1_ps(packed_ptr0[1]), accumulator1);
                    accumulator2 = _mm256_fmadd_ps(input_values, _mm256_set1_ps(packed_ptr0[2]), accumulator2);
                    accumulator3 = _mm256_fmadd_ps(input_values, _mm256_set1_ps(packed_ptr0[3]), accumulator3);
                    accumulator4 = _mm256_fmadd_ps(input_values, _mm256_set1_ps(packed_ptr1[0]), accumulator4);
                    accumulator5 = _mm256_fmadd_ps(input_values, _mm256_set1_ps(packed_ptr1[1]), accumulator5);
                    accumulator6 = _mm256_fmadd_ps(input_values, _mm256_set1_ps(packed_ptr1[2]), accumulator6);
                    accumulator7 = _mm256_fmadd_ps(input_values, _mm256_set1_ps(packed_ptr1[3]), accumulator7);
                    input_ptr += (size_t)channel_plane;
                    packed_ptr0 += LW_PACKED_CONV1X1_OUTPUT_TILE;
                    packed_ptr1 += LW_PACKED_CONV1X1_OUTPUT_TILE;
                }
                float* output0 =
                    batch_output + (size_t)((uint64_t)output_base * channel_plane + spatial);
                float* output1 = output0 + (size_t)channel_plane;
                float* output2 = output1 + (size_t)channel_plane;
                float* output3 = output2 + (size_t)channel_plane;
                float* output4 = output3 + (size_t)channel_plane;
                float* output5 = output4 + (size_t)channel_plane;
                float* output6 = output5 + (size_t)channel_plane;
                float* output7 = output6 + (size_t)channel_plane;
                _mm256_storeu_ps(output0, accumulator0);
                _mm256_storeu_ps(output1, accumulator1);
                _mm256_storeu_ps(output2, accumulator2);
                _mm256_storeu_ps(output3, accumulator3);
                _mm256_storeu_ps(output4, accumulator4);
                _mm256_storeu_ps(output5, accumulator5);
                _mm256_storeu_ps(output6, accumulator6);
                _mm256_storeu_ps(output7, accumulator7);
            }

            for (; spatial < channel_plane; ++spatial) {
                float accumulator0 = bias == NULL ? 0.0f : bias[output_base];
                float accumulator1 = bias == NULL ? 0.0f : bias[output_base + 1u];
                float accumulator2 = bias == NULL ? 0.0f : bias[output_base + 2u];
                float accumulator3 = bias == NULL ? 0.0f : bias[output_base + 3u];
                float accumulator4 = bias == NULL ? 0.0f : bias[output_base + 4u];
                float accumulator5 = bias == NULL ? 0.0f : bias[output_base + 5u];
                float accumulator6 = bias == NULL ? 0.0f : bias[output_base + 6u];
                float accumulator7 = bias == NULL ? 0.0f : bias[output_base + 7u];
                const float* input_ptr = batch_input + (size_t)spatial;
                const float* packed_ptr0 = packed0;
                const float* packed_ptr1 = packed1;
                uint32_t input_channel;
                for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
                    const float input_value = *input_ptr;
                    accumulator0 += input_value * packed_ptr0[0];
                    accumulator1 += input_value * packed_ptr0[1];
                    accumulator2 += input_value * packed_ptr0[2];
                    accumulator3 += input_value * packed_ptr0[3];
                    accumulator4 += input_value * packed_ptr1[0];
                    accumulator5 += input_value * packed_ptr1[1];
                    accumulator6 += input_value * packed_ptr1[2];
                    accumulator7 += input_value * packed_ptr1[3];
                    input_ptr += (size_t)channel_plane;
                    packed_ptr0 += LW_PACKED_CONV1X1_OUTPUT_TILE;
                    packed_ptr1 += LW_PACKED_CONV1X1_OUTPUT_TILE;
                }
                float* output0 =
                    batch_output + (size_t)((uint64_t)output_base * channel_plane + spatial);
                float* output1 = output0 + (size_t)channel_plane;
                float* output2 = output1 + (size_t)channel_plane;
                float* output3 = output2 + (size_t)channel_plane;
                float* output4 = output3 + (size_t)channel_plane;
                float* output5 = output4 + (size_t)channel_plane;
                float* output6 = output5 + (size_t)channel_plane;
                float* output7 = output6 + (size_t)channel_plane;
                *output0 = accumulator0;
                *output1 = accumulator1;
                *output2 = accumulator2;
                *output3 = accumulator3;
                *output4 = accumulator4;
                *output5 = accumulator5;
                *output6 = accumulator6;
                *output7 = accumulator7;
            }
        }
    }
#else
    lw_avx2_fma_packed_conv1x1_f32(input, packed_weights, bias, output,
                                   input_dimensions, output_dimensions);
#endif
}
