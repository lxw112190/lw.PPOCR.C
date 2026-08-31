#include "simd_kernels.h"

#include "packed_conv_internal.h"

#include <stddef.h>

#if defined(__loongarch_sx)
#  include <lsxintrin.h>
#  define LW_COMPILES_LSX_PACKED_CONV1X1 1
#else
#  define LW_COMPILES_LSX_PACKED_CONV1X1 0
#endif

#if LW_COMPILES_LSX_PACKED_CONV1X1
static __m128 lw_lsx_broadcast_f32(float value) {
    return (__m128)__lsx_vldrepl_w(&value, 0);
}

static __m128 lw_lsx_load_f32(const float* values) {
    return (__m128)__lsx_vld(values, 0);
}

static void lw_lsx_store_f32(float* output, __m128 values) {
    __lsx_vst((__m128i)values, output, 0);
}
#endif

void lw_lsx_packed_conv1x1_f32(const float* input, const float* packed_weights,
                               const float* bias, float* output,
                               const int32_t input_dimensions[4],
                               const int32_t output_dimensions[4]) {
#if LW_COMPILES_LSX_PACKED_CONV1X1
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
                __m128 accumulator0 =
                    lw_lsx_broadcast_f32(bias == NULL ? 0.0f : bias[output_base]);
                __m128 accumulator1 = lw_lsx_broadcast_f32(
                    bias == NULL || valid_outputs <= 1u ? 0.0f : bias[output_base + 1u]);
                __m128 accumulator2 = lw_lsx_broadcast_f32(
                    bias == NULL || valid_outputs <= 2u ? 0.0f : bias[output_base + 2u]);
                __m128 accumulator3 = lw_lsx_broadcast_f32(
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
                    const __m128 input_values = lw_lsx_load_f32(input_channel_data);
                    accumulator0 = __lsx_vfadd_s(
                        accumulator0,
                        __lsx_vfmul_s(input_values, lw_lsx_broadcast_f32(packed[0])));
                    accumulator1 = __lsx_vfadd_s(
                        accumulator1,
                        __lsx_vfmul_s(input_values, lw_lsx_broadcast_f32(packed[1])));
                    accumulator2 = __lsx_vfadd_s(
                        accumulator2,
                        __lsx_vfmul_s(input_values, lw_lsx_broadcast_f32(packed[2])));
                    accumulator3 = __lsx_vfadd_s(
                        accumulator3,
                        __lsx_vfmul_s(input_values, lw_lsx_broadcast_f32(packed[3])));
                }
                lw_lsx_store_f32(batch_output +
                                     (size_t)((uint64_t)output_base * channel_plane + spatial),
                                 accumulator0);
                if (valid_outputs > 1u) {
                    lw_lsx_store_f32(
                        batch_output +
                            (size_t)((uint64_t)(output_base + 1u) * channel_plane + spatial),
                        accumulator1);
                }
                if (valid_outputs > 2u) {
                    lw_lsx_store_f32(
                        batch_output +
                            (size_t)((uint64_t)(output_base + 2u) * channel_plane + spatial),
                        accumulator2);
                }
                if (valid_outputs > 3u) {
                    lw_lsx_store_f32(
                        batch_output +
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
