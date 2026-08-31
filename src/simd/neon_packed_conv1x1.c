#include "simd_kernels.h"

#include "packed_conv_internal.h"

#include <stddef.h>

#if defined(_M_ARM64) || defined(__aarch64__)
#  include <arm_neon.h>
#  define LW_COMPILES_NEON_PACKED_CONV1X1 1
#else
#  define LW_COMPILES_NEON_PACKED_CONV1X1 0
#endif

void lw_neon_packed_conv1x1_f32(const float* input, const float* packed_weights,
                                const float* bias, float* output,
                                const int32_t input_dimensions[4],
                                const int32_t output_dimensions[4]) {
#if LW_COMPILES_NEON_PACKED_CONV1X1
    const uint32_t input_channels = (uint32_t)input_dimensions[1];
    const uint32_t output_channels = (uint32_t)output_dimensions[1];
    const uint64_t channel_plane =
        (uint64_t)(uint32_t)input_dimensions[2] * (uint32_t)input_dimensions[3];
    const uint32_t output_blocks =
        (output_channels + LW_PACKED_CONV1X1_OUTPUT_TILE - 1u) /
        LW_PACKED_CONV1X1_OUTPUT_TILE;
    uint32_t batch;
    for (batch = 0u; batch < (uint32_t)input_dimensions[0]; ++batch) {
        const float* batch_input =
            input + (size_t)((uint64_t)batch * input_channels * channel_plane);
        float* batch_output = output + (size_t)((uint64_t)batch * output_channels * channel_plane);
        uint32_t output_block;
        for (output_block = 0u; output_block < output_blocks; ++output_block) {
            const uint32_t output_base = output_block * LW_PACKED_CONV1X1_OUTPUT_TILE;
            uint32_t valid_outputs = output_channels - output_base;
            uint64_t spatial = 0u;
            if (valid_outputs > LW_PACKED_CONV1X1_OUTPUT_TILE) {
                valid_outputs = LW_PACKED_CONV1X1_OUTPUT_TILE;
            }
            for (; spatial + 4u <= channel_plane; spatial += 4u) {
                float32x4_t accumulator0 =
                    vdupq_n_f32(bias == NULL ? 0.0f : bias[output_base]);
                float32x4_t accumulator1 = vdupq_n_f32(
                    bias == NULL || valid_outputs <= 1u ? 0.0f : bias[output_base + 1u]);
                float32x4_t accumulator2 = vdupq_n_f32(
                    bias == NULL || valid_outputs <= 2u ? 0.0f : bias[output_base + 2u]);
                float32x4_t accumulator3 = vdupq_n_f32(
                    bias == NULL || valid_outputs <= 3u ? 0.0f : bias[output_base + 3u]);
                uint32_t input_channel;
                for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
                    const float* input_channel_data =
                        batch_input +
                        (size_t)((uint64_t)input_channel * channel_plane + spatial);
                    const float* packed =
                        packed_weights +
                        (size_t)(((uint64_t)output_block * input_channels + input_channel) *
                                 LW_PACKED_CONV1X1_OUTPUT_TILE);
                    const float32x4_t input_values = vld1q_f32(input_channel_data);
                    accumulator0 =
                        vaddq_f32(accumulator0, vmulq_n_f32(input_values, packed[0]));
                    accumulator1 =
                        vaddq_f32(accumulator1, vmulq_n_f32(input_values, packed[1]));
                    accumulator2 =
                        vaddq_f32(accumulator2, vmulq_n_f32(input_values, packed[2]));
                    accumulator3 =
                        vaddq_f32(accumulator3, vmulq_n_f32(input_values, packed[3]));
                }
                vst1q_f32(batch_output +
                              (size_t)((uint64_t)output_base * channel_plane + spatial),
                          accumulator0);
                if (valid_outputs > 1u) {
                    vst1q_f32(batch_output +
                                  (size_t)((uint64_t)(output_base + 1u) * channel_plane + spatial),
                              accumulator1);
                }
                if (valid_outputs > 2u) {
                    vst1q_f32(batch_output +
                                  (size_t)((uint64_t)(output_base + 2u) * channel_plane + spatial),
                              accumulator2);
                }
                if (valid_outputs > 3u) {
                    vst1q_f32(batch_output +
                                  (size_t)((uint64_t)(output_base + 3u) * channel_plane + spatial),
                              accumulator3);
                }
            }
            for (; spatial < channel_plane; ++spatial) {
                float accumulators[LW_PACKED_CONV1X1_OUTPUT_TILE] = {
                    bias == NULL ? 0.0f : bias[output_base],
                    bias == NULL || valid_outputs <= 1u ? 0.0f : bias[output_base + 1u],
                    bias == NULL || valid_outputs <= 2u ? 0.0f : bias[output_base + 2u],
                    bias == NULL || valid_outputs <= 3u ? 0.0f : bias[output_base + 3u]};
                uint32_t input_channel;
                uint32_t output_lane;
                for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
                    const float input_value =
                        batch_input[(size_t)((uint64_t)input_channel * channel_plane + spatial)];
                    const float* packed =
                        packed_weights +
                        (size_t)(((uint64_t)output_block * input_channels + input_channel) *
                                 LW_PACKED_CONV1X1_OUTPUT_TILE);
                    for (output_lane = 0u; output_lane < valid_outputs; ++output_lane) {
                        accumulators[output_lane] += input_value * packed[output_lane];
                    }
                }
                for (output_lane = 0u; output_lane < valid_outputs; ++output_lane) {
                    batch_output[(size_t)(((uint64_t)output_base + output_lane) * channel_plane +
                                          spatial)] = accumulators[output_lane];
                }
            }
        }
    }
#else
    lw_scalar_packed_conv1x1_f32(input, packed_weights, bias, output, input_dimensions,
                                 output_dimensions);
#endif
}
