#include "simd_kernels.h"

#include "packed_conv_internal.h"

#include <stddef.h>

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
#  include <immintrin.h>
#  define LW_COMPILES_AVX2_PACKED_CONV1X1 1
#else
#  define LW_COMPILES_AVX2_PACKED_CONV1X1 0
#endif

#if LW_COMPILES_AVX2_PACKED_CONV1X1 && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,fma")))
#endif
void lw_avx2_fma_packed_conv1x1_f32(const float* input, const float* packed_weights,
                                const float* bias, float* output,
                                const int32_t input_dimensions[4],
                                const int32_t output_dimensions[4]) {
#if LW_COMPILES_AVX2_PACKED_CONV1X1
    uint32_t input_channels = (uint32_t)input_dimensions[1];
    uint32_t output_channels = (uint32_t)output_dimensions[1];
    uint64_t channel_plane =
        (uint64_t)(uint32_t)input_dimensions[2] * (uint32_t)input_dimensions[3];
    uint32_t output_blocks =
        (output_channels + LW_PACKED_CONV1X1_OUTPUT_TILE - 1u) / LW_PACKED_CONV1X1_OUTPUT_TILE;
    uint32_t batch;
    for (batch = 0u; batch < (uint32_t)input_dimensions[0]; ++batch) {
        const float* batch_input =
            input + (size_t)((uint64_t)batch * input_channels * channel_plane);
        float* batch_output = output + (size_t)((uint64_t)batch * output_channels * channel_plane);
        uint32_t output_block;
        for (output_block = 0u; output_block < output_blocks; ++output_block) {
            uint32_t output_base = output_block * LW_PACKED_CONV1X1_OUTPUT_TILE;
            uint32_t valid_outputs = output_channels - output_base;
            uint64_t spatial = 0u;
            if (valid_outputs > LW_PACKED_CONV1X1_OUTPUT_TILE) {
                valid_outputs = LW_PACKED_CONV1X1_OUTPUT_TILE;
            }
#  if defined(_M_X64) || defined(__x86_64__)
            /* x64 exposes sixteen YMM registers, so two spatial vectors can
             * stay live for each of the four output planes. This halves the
             * packed-weight broadcasts and loop bookkeeping compared with the
             * 4x8 kernel below. x86 retains 4x8 because it has only eight
             * architectural vector registers and would otherwise spill. */
            if (valid_outputs == LW_PACKED_CONV1X1_OUTPUT_TILE) {
                for (; spatial + 16u <= channel_plane; spatial += 16u) {
                    __m256 accumulator0_low =
                        _mm256_set1_ps(bias == NULL ? 0.0f : bias[output_base]);
                    __m256 accumulator0_high = accumulator0_low;
                    __m256 accumulator1_low =
                        _mm256_set1_ps(bias == NULL ? 0.0f : bias[output_base + 1u]);
                    __m256 accumulator1_high = accumulator1_low;
                    __m256 accumulator2_low =
                        _mm256_set1_ps(bias == NULL ? 0.0f : bias[output_base + 2u]);
                    __m256 accumulator2_high = accumulator2_low;
                    __m256 accumulator3_low =
                        _mm256_set1_ps(bias == NULL ? 0.0f : bias[output_base + 3u]);
                    __m256 accumulator3_high = accumulator3_low;
                    const float* input_ptr = batch_input + (size_t)spatial;
                    const float* packed_ptr =
                        packed_weights +
                        (size_t)((uint64_t)output_block * input_channels *
                                  LW_PACKED_CONV1X1_OUTPUT_TILE);
                    uint32_t input_channel;
                    for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
                        __m256 input_low = _mm256_loadu_ps(input_ptr);
                        __m256 input_high = _mm256_loadu_ps(input_ptr + 8u);
                        __m256 weight = _mm256_set1_ps(packed_ptr[0]);
                        accumulator0_low =
                            _mm256_fmadd_ps(input_low, weight, accumulator0_low);
                        accumulator0_high =
                            _mm256_fmadd_ps(input_high, weight, accumulator0_high);
                        weight = _mm256_set1_ps(packed_ptr[1]);
                        accumulator1_low =
                            _mm256_fmadd_ps(input_low, weight, accumulator1_low);
                        accumulator1_high =
                            _mm256_fmadd_ps(input_high, weight, accumulator1_high);
                        weight = _mm256_set1_ps(packed_ptr[2]);
                        accumulator2_low =
                            _mm256_fmadd_ps(input_low, weight, accumulator2_low);
                        accumulator2_high =
                            _mm256_fmadd_ps(input_high, weight, accumulator2_high);
                        weight = _mm256_set1_ps(packed_ptr[3]);
                        accumulator3_low =
                            _mm256_fmadd_ps(input_low, weight, accumulator3_low);
                        accumulator3_high =
                            _mm256_fmadd_ps(input_high, weight, accumulator3_high);
                        input_ptr += (size_t)channel_plane;
                        packed_ptr += LW_PACKED_CONV1X1_OUTPUT_TILE;
                    }
                    float* output0 =
                        batch_output + (size_t)((uint64_t)output_base * channel_plane + spatial);
                    float* output1 = output0 + (size_t)channel_plane;
                    float* output2 = output1 + (size_t)channel_plane;
                    float* output3 = output2 + (size_t)channel_plane;
                    _mm256_storeu_ps(output0, accumulator0_low);
                    _mm256_storeu_ps(output0 + 8u, accumulator0_high);
                    _mm256_storeu_ps(output1, accumulator1_low);
                    _mm256_storeu_ps(output1 + 8u, accumulator1_high);
                    _mm256_storeu_ps(output2, accumulator2_low);
                    _mm256_storeu_ps(output2 + 8u, accumulator2_high);
                    _mm256_storeu_ps(output3, accumulator3_low);
                    _mm256_storeu_ps(output3 + 8u, accumulator3_high);
                }
            }
#  endif
            for (; spatial + 8u <= channel_plane; spatial += 8u) {
                __m256 accumulator0 = _mm256_set1_ps(bias == NULL ? 0.0f : bias[output_base]);
                __m256 accumulator1 = _mm256_set1_ps(
                    bias == NULL || valid_outputs <= 1u ? 0.0f : bias[output_base + 1u]);
                __m256 accumulator2 = _mm256_set1_ps(
                    bias == NULL || valid_outputs <= 2u ? 0.0f : bias[output_base + 2u]);
                __m256 accumulator3 = _mm256_set1_ps(
                    bias == NULL || valid_outputs <= 3u ? 0.0f : bias[output_base + 3u]);
                const float* input_ptr = batch_input + (size_t)spatial;
                const float* packed_ptr =
                    packed_weights +
                    (size_t)((uint64_t)output_block * input_channels *
                              LW_PACKED_CONV1X1_OUTPUT_TILE);
                uint32_t input_channel;
                for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
                    __m256 input_values = _mm256_loadu_ps(input_ptr);
                    accumulator0 = _mm256_fmadd_ps(
                        input_values, _mm256_set1_ps(packed_ptr[0]), accumulator0);
                    accumulator1 = _mm256_fmadd_ps(
                        input_values, _mm256_set1_ps(packed_ptr[1]), accumulator1);
                    accumulator2 = _mm256_fmadd_ps(
                        input_values, _mm256_set1_ps(packed_ptr[2]), accumulator2);
                    accumulator3 = _mm256_fmadd_ps(
                        input_values, _mm256_set1_ps(packed_ptr[3]), accumulator3);
                    input_ptr += (size_t)channel_plane;
                    packed_ptr += LW_PACKED_CONV1X1_OUTPUT_TILE;
                }
                float* output0 =
                    batch_output + (size_t)((uint64_t)output_base * channel_plane + spatial);
                float* output1 = output0 + (size_t)channel_plane;
                float* output2 = output1 + (size_t)channel_plane;
                float* output3 = output2 + (size_t)channel_plane;
                _mm256_storeu_ps(output0, accumulator0);
                if (valid_outputs > 1u) {
                    _mm256_storeu_ps(output1, accumulator1);
                }
                if (valid_outputs > 2u) {
                    _mm256_storeu_ps(output2, accumulator2);
                }
                if (valid_outputs > 3u) {
                    _mm256_storeu_ps(output3, accumulator3);
                }
            }
            for (; spatial < channel_plane; ++spatial) {
                float accumulator0 = bias == NULL ? 0.0f : bias[output_base];
                float accumulator1 =
                    bias == NULL || valid_outputs <= 1u ? 0.0f : bias[output_base + 1u];
                float accumulator2 =
                    bias == NULL || valid_outputs <= 2u ? 0.0f : bias[output_base + 2u];
                float accumulator3 =
                    bias == NULL || valid_outputs <= 3u ? 0.0f : bias[output_base + 3u];
                const float* input_ptr = batch_input + (size_t)spatial;
                const float* packed_ptr =
                    packed_weights +
                    (size_t)((uint64_t)output_block * input_channels *
                              LW_PACKED_CONV1X1_OUTPUT_TILE);
                uint32_t input_channel;
                for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
                    float input_value = *input_ptr;
                    accumulator0 += input_value * packed_ptr[0];
                    accumulator1 += input_value * packed_ptr[1];
                    accumulator2 += input_value * packed_ptr[2];
                    accumulator3 += input_value * packed_ptr[3];
                    input_ptr += (size_t)channel_plane;
                    packed_ptr += LW_PACKED_CONV1X1_OUTPUT_TILE;
                }
                float* output0 =
                    batch_output + (size_t)((uint64_t)output_base * channel_plane + spatial);
                float* output1 = output0 + (size_t)channel_plane;
                float* output2 = output1 + (size_t)channel_plane;
                float* output3 = output2 + (size_t)channel_plane;
                *output0 = accumulator0;
                if (valid_outputs > 1u)
                    *output1 = accumulator1;
                if (valid_outputs > 2u)
                    *output2 = accumulator2;
                if (valid_outputs > 3u)
                    *output3 = accumulator3;
            }
        }
    }
#else
    lw_scalar_packed_conv1x1_f32(input, packed_weights, bias, output, input_dimensions,
                                 output_dimensions);
#endif
}
