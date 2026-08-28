#include "simd_kernels.h"
#include "simd_platform.h"

#include "packed_conv_internal.h"

#include <stddef.h>

#define LW_COMPILES_SSE2_PACKED_CONV1X1 LW_SIMD_HAS_SSE2_INTRINSICS

LW_SIMD_SSE2_TARGET
void lw_sse2_packed_conv1x1_f32(const float* input, const float* packed_weights,
                                const float* bias, float* output,
                                const int32_t input_dimensions[4],
                                const int32_t output_dimensions[4]) {
#if LW_COMPILES_SSE2_PACKED_CONV1X1
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
            for (; spatial + 4u <= channel_plane; spatial += 4u) {
                __m128 accumulator0 = _mm_set1_ps(bias == NULL ? 0.0f : bias[output_base]);
                __m128 accumulator1 = _mm_set1_ps(
                    bias == NULL || valid_outputs <= 1u ? 0.0f : bias[output_base + 1u]);
                __m128 accumulator2 = _mm_set1_ps(
                    bias == NULL || valid_outputs <= 2u ? 0.0f : bias[output_base + 2u]);
                __m128 accumulator3 = _mm_set1_ps(
                    bias == NULL || valid_outputs <= 3u ? 0.0f : bias[output_base + 3u]);
                uint32_t input_channel;
                for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
                    const float* input_channel_data =
                        batch_input + (size_t)((uint64_t)input_channel * channel_plane + spatial);
                    const float* packed =
                        packed_weights +
                        (size_t)(((uint64_t)output_block * input_channels + input_channel) *
                                 LW_PACKED_CONV1X1_OUTPUT_TILE);
                    __m128 input_values = _mm_loadu_ps(input_channel_data);
                    accumulator0 =
                        _mm_add_ps(accumulator0, _mm_mul_ps(input_values, _mm_set1_ps(packed[0])));
                    accumulator1 =
                        _mm_add_ps(accumulator1, _mm_mul_ps(input_values, _mm_set1_ps(packed[1])));
                    accumulator2 =
                        _mm_add_ps(accumulator2, _mm_mul_ps(input_values, _mm_set1_ps(packed[2])));
                    accumulator3 =
                        _mm_add_ps(accumulator3, _mm_mul_ps(input_values, _mm_set1_ps(packed[3])));
                }
                _mm_storeu_ps(batch_output +
                                  (size_t)((uint64_t)output_base * channel_plane + spatial),
                              accumulator0);
                if (valid_outputs > 1u)
                    _mm_storeu_ps(
                        batch_output +
                            (size_t)((uint64_t)(output_base + 1u) * channel_plane + spatial),
                        accumulator1);
                if (valid_outputs > 2u)
                    _mm_storeu_ps(
                        batch_output +
                            (size_t)((uint64_t)(output_base + 2u) * channel_plane + spatial),
                        accumulator2);
                if (valid_outputs > 3u)
                    _mm_storeu_ps(
                        batch_output +
                            (size_t)((uint64_t)(output_base + 3u) * channel_plane + spatial),
                        accumulator3);
            }
            for (; spatial < channel_plane; ++spatial) {
                float accumulator0 = bias == NULL ? 0.0f : bias[output_base];
                float accumulator1 =
                    bias == NULL || valid_outputs <= 1u ? 0.0f : bias[output_base + 1u];
                float accumulator2 =
                    bias == NULL || valid_outputs <= 2u ? 0.0f : bias[output_base + 2u];
                float accumulator3 =
                    bias == NULL || valid_outputs <= 3u ? 0.0f : bias[output_base + 3u];
                uint32_t input_channel;
                for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
                    float input_value =
                        batch_input[(size_t)((uint64_t)input_channel * channel_plane + spatial)];
                    const float* packed =
                        packed_weights +
                        (size_t)(((uint64_t)output_block * input_channels + input_channel) *
                                 LW_PACKED_CONV1X1_OUTPUT_TILE);
                    accumulator0 += input_value * packed[0];
                    accumulator1 += input_value * packed[1];
                    accumulator2 += input_value * packed[2];
                    accumulator3 += input_value * packed[3];
                }
                batch_output[(size_t)((uint64_t)output_base * channel_plane + spatial)] =
                    accumulator0;
                if (valid_outputs > 1u)
                    batch_output[(size_t)((uint64_t)(output_base + 1u) * channel_plane + spatial)] =
                        accumulator1;
                if (valid_outputs > 2u)
                    batch_output[(size_t)((uint64_t)(output_base + 2u) * channel_plane + spatial)] =
                        accumulator2;
                if (valid_outputs > 3u)
                    batch_output[(size_t)((uint64_t)(output_base + 3u) * channel_plane + spatial)] =
                        accumulator3;
            }
        }
    }
#else
    lw_scalar_packed_conv1x1_f32(input, packed_weights, bias, output, input_dimensions,
                                 output_dimensions);
#endif
}
