#include "packed_conv_internal.h"

/* NCHW pointwise-Conv packing and the portable microkernel fallback. */

#include "cpu_features.h"
#include "simd_kernels.h"

#include <limits.h>
#include <stddef.h>

int lw_packed_conv1x1_weight_count(uint32_t input_channels, uint32_t output_channels,
                                   uint64_t* weight_count) {
    uint64_t output_blocks;
    if (input_channels == 0u || output_channels == 0u || weight_count == NULL) {
        return 0;
    }
    output_blocks = ((uint64_t)output_channels + LW_PACKED_CONV1X1_OUTPUT_TILE - 1u) /
                    LW_PACKED_CONV1X1_OUTPUT_TILE;
    if (output_blocks > UINT64_MAX / input_channels ||
        output_blocks * input_channels > UINT64_MAX / LW_PACKED_CONV1X1_OUTPUT_TILE) {
        return 0;
    }
    *weight_count = output_blocks * input_channels * LW_PACKED_CONV1X1_OUTPUT_TILE;
    return *weight_count <= (uint64_t)(SIZE_MAX / sizeof(float));
}

void lw_pack_conv1x1_weights_f32(const float* weights, uint32_t input_channels,
                                 uint32_t output_channels, float* packed_weights) {
    uint32_t output_block;
    uint32_t output_blocks =
        (output_channels + LW_PACKED_CONV1X1_OUTPUT_TILE - 1u) / LW_PACKED_CONV1X1_OUTPUT_TILE;
    for (output_block = 0u; output_block < output_blocks; ++output_block) {
        uint32_t input_channel;
        for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
            uint32_t output_lane;
            for (output_lane = 0u; output_lane < LW_PACKED_CONV1X1_OUTPUT_TILE; ++output_lane) {
                uint32_t output_channel =
                    output_block * LW_PACKED_CONV1X1_OUTPUT_TILE + output_lane;
                uint64_t packed_index = ((uint64_t)output_block * input_channels + input_channel) *
                                            LW_PACKED_CONV1X1_OUTPUT_TILE +
                                        output_lane;
                packed_weights[(size_t)packed_index] =
                    output_channel < output_channels
                        ? weights[(size_t)((uint64_t)output_channel * input_channels +
                                           input_channel)]
                        : 0.0f;
            }
        }
    }
}

void lw_scalar_packed_conv1x1_f32(const float* input, const float* packed_weights,
                                  const float* bias, float* output,
                                  const int32_t input_dimensions[4],
                                  const int32_t output_dimensions[4]) {
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
            uint64_t spatial;
            if (valid_outputs > LW_PACKED_CONV1X1_OUTPUT_TILE) {
                valid_outputs = LW_PACKED_CONV1X1_OUTPUT_TILE;
            }
            for (spatial = 0u; spatial < channel_plane; ++spatial) {
                float accumulators[LW_PACKED_CONV1X1_OUTPUT_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};
                uint32_t output_lane;
                uint32_t input_channel;
                for (output_lane = 0u; output_lane < valid_outputs; ++output_lane) {
                    accumulators[output_lane] =
                        bias == NULL ? 0.0f : bias[output_base + output_lane];
                }
                for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
                    float input_value =
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
}

void lw_packed_conv1x1_f32(const float* input, const float* packed_weights, const float* bias,
                           float* output, const int32_t input_dimensions[4],
                           const int32_t output_dimensions[4]) {
    lw_simd_level simd_level = lw_detect_simd_level();
    if (simd_level >= LW_SIMD_LEVEL_AVX2) {
        lw_avx2_packed_conv1x1_f32(input, packed_weights, bias, output, input_dimensions,
                                   output_dimensions);
    } else if (simd_level >= LW_SIMD_LEVEL_SSE2) {
        lw_sse2_packed_conv1x1_f32(input, packed_weights, bias, output, input_dimensions,
                                   output_dimensions);
    } else {
        lw_scalar_packed_conv1x1_f32(input, packed_weights, bias, output, input_dimensions,
                                     output_dimensions);
    }
}
