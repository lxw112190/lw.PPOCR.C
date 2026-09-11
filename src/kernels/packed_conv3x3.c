#include "packed_conv3x3_internal.h"

#include "cpu_features.h"
#include "simd_kernels.h"

#include <stddef.h>

int lw_packed_conv3x3_stride2_weight_count(uint32_t input_channels,
                                           uint32_t output_channels,
                                           uint64_t* weight_count) {
    uint64_t output_blocks;
    if (input_channels == 0u || output_channels == 0u || weight_count == NULL) {
        return 0;
    }
    output_blocks =
        ((uint64_t)output_channels + LW_PACKED_CONV3X3_STRIDE2_OUTPUT_TILE - 1u) /
        LW_PACKED_CONV3X3_STRIDE2_OUTPUT_TILE;
    if (output_blocks > UINT64_MAX / input_channels ||
        output_blocks * input_channels > UINT64_MAX / 9u ||
        output_blocks * input_channels * 9u >
            UINT64_MAX / LW_PACKED_CONV3X3_STRIDE2_OUTPUT_TILE) {
        return 0;
    }
    *weight_count = output_blocks * input_channels * 9u *
                    LW_PACKED_CONV3X3_STRIDE2_OUTPUT_TILE;
    return *weight_count <= (uint64_t)(SIZE_MAX / sizeof(float));
}

void lw_pack_conv3x3_stride2_weights_f32(const float* weights, uint32_t input_channels,
                                         uint32_t output_channels, float* packed_weights) {
    const uint32_t output_blocks =
        (output_channels + LW_PACKED_CONV3X3_STRIDE2_OUTPUT_TILE - 1u) /
        LW_PACKED_CONV3X3_STRIDE2_OUTPUT_TILE;
    uint32_t output_block;
    for (output_block = 0u; output_block < output_blocks; ++output_block) {
        uint32_t input_channel;
        for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
            uint32_t kernel_index;
            for (kernel_index = 0u; kernel_index < 9u; ++kernel_index) {
                uint32_t output_lane;
                for (output_lane = 0u;
                     output_lane < LW_PACKED_CONV3X3_STRIDE2_OUTPUT_TILE;
                     ++output_lane) {
                    const uint32_t output_channel =
                        output_block * LW_PACKED_CONV3X3_STRIDE2_OUTPUT_TILE + output_lane;
                    const uint64_t packed_index =
                        (((uint64_t)output_block * input_channels + input_channel) * 9u +
                         kernel_index) *
                            LW_PACKED_CONV3X3_STRIDE2_OUTPUT_TILE +
                        output_lane;
                    packed_weights[(size_t)packed_index] =
                        output_channel < output_channels
                            ? weights[(size_t)(((uint64_t)output_channel * input_channels +
                                                input_channel) *
                                                   9u +
                                               kernel_index)]
                            : 0.0f;
                }
            }
        }
    }
}

void lw_scalar_packed_conv3x3_stride2_pad1_f32(
    const float* input, const float* packed_weights, const float* bias, float* output,
    const int32_t input_dimensions[4], const int32_t output_dimensions[4]) {
    const uint32_t input_channels = (uint32_t)input_dimensions[1];
    const uint32_t input_height = (uint32_t)input_dimensions[2];
    const uint32_t input_width = (uint32_t)input_dimensions[3];
    const uint32_t output_channels = (uint32_t)output_dimensions[1];
    const uint32_t output_height = (uint32_t)output_dimensions[2];
    const uint32_t output_width = (uint32_t)output_dimensions[3];
    const uint64_t input_plane = (uint64_t)input_height * input_width;
    const uint64_t output_plane = (uint64_t)output_height * output_width;
    const uint32_t output_blocks =
        (output_channels + LW_PACKED_CONV3X3_STRIDE2_OUTPUT_TILE - 1u) /
        LW_PACKED_CONV3X3_STRIDE2_OUTPUT_TILE;
    uint32_t batch;
    for (batch = 0u; batch < (uint32_t)input_dimensions[0]; ++batch) {
        const float* batch_input =
            input + (size_t)((uint64_t)batch * input_channels * input_plane);
        float* batch_output =
            output + (size_t)((uint64_t)batch * output_channels * output_plane);
        uint32_t output_block;
        for (output_block = 0u; output_block < output_blocks; ++output_block) {
            const uint32_t output_base =
                output_block * LW_PACKED_CONV3X3_STRIDE2_OUTPUT_TILE;
            uint32_t valid_outputs = output_channels - output_base;
            uint32_t output_lane;
            if (valid_outputs > LW_PACKED_CONV3X3_STRIDE2_OUTPUT_TILE) {
                valid_outputs = LW_PACKED_CONV3X3_STRIDE2_OUTPUT_TILE;
            }
            for (output_lane = 0u; output_lane < valid_outputs; ++output_lane) {
                uint32_t output_y;
                for (output_y = 0u; output_y < output_height; ++output_y) {
                    uint32_t output_x;
                    for (output_x = 0u; output_x < output_width; ++output_x) {
                        float accumulator =
                            bias == NULL ? 0.0f : bias[output_base + output_lane];
                        uint32_t input_channel;
                        for (input_channel = 0u; input_channel < input_channels;
                             ++input_channel) {
                            const float* input_channel_data =
                                batch_input + (size_t)((uint64_t)input_channel * input_plane);
                            uint32_t kernel_y;
                            for (kernel_y = 0u; kernel_y < 3u; ++kernel_y) {
                                const int64_t input_y =
                                    (int64_t)output_y * 2 - 1 + kernel_y;
                                uint32_t kernel_x;
                                if (input_y < 0 || input_y >= (int64_t)input_height) {
                                    continue;
                                }
                                for (kernel_x = 0u; kernel_x < 3u; ++kernel_x) {
                                    const int64_t input_x =
                                        (int64_t)output_x * 2 - 1 + kernel_x;
                                    if (input_x >= 0 && input_x < (int64_t)input_width) {
                                        const uint32_t kernel_index = kernel_y * 3u + kernel_x;
                                        const uint64_t packed_index =
                                            (((uint64_t)output_block * input_channels +
                                              input_channel) *
                                                 9u +
                                             kernel_index) *
                                                LW_PACKED_CONV3X3_STRIDE2_OUTPUT_TILE +
                                            output_lane;
                                        accumulator +=
                                            input_channel_data[(size_t)((uint64_t)(uint32_t)input_y *
                                                                        input_width +
                                                                        (uint32_t)input_x)] *
                                            packed_weights[(size_t)packed_index];
                                    }
                                }
                            }
                        }
                        batch_output[(size_t)(((uint64_t)output_base + output_lane) *
                                                  output_plane +
                                              (uint64_t)output_y * output_width + output_x)] =
                            accumulator;
                    }
                }
            }
        }
    }
}

void lw_packed_conv3x3_stride2_pad1_f32(
    const float* input, const float* packed_weights, const float* bias, float* output,
    const int32_t input_dimensions[4], const int32_t output_dimensions[4]) {
    const lw_simd_level simd_level = lw_detect_simd_level();
#if defined(LW_EXPERIMENTAL_AVX2_FMA_DISPATCH)
    const lw_cpu_capabilities capabilities = lw_get_cpu_capabilities();
    if (lw_simd_level_is_avx2(simd_level) && capabilities.has_avx2_fma &&
        input_dimensions[0] == 1 && input_dimensions[1] == 24 &&
        input_dimensions[2] == 24 &&
        (input_dimensions[3] == 160 || input_dimensions[3] == 480) &&
        output_dimensions[1] == 48 && output_dimensions[2] == 12 &&
        (output_dimensions[3] == 80 || output_dimensions[3] == 240)) {
        lw_avx2_fma_packed_conv3x3_stride2_pad1_f32(
            input, packed_weights, bias, output, input_dimensions, output_dimensions);
        return;
    }
#endif
    if (lw_simd_level_is_avx2(simd_level) &&
        (uint32_t)output_dimensions[1] % LW_PACKED_CONV3X3_STRIDE2_OUTPUT_TILE == 0u) {
        lw_avx2_packed_conv3x3_stride2_pad1_f32(
            input, packed_weights, bias, output, input_dimensions, output_dimensions);
    } else {
        lw_scalar_packed_conv3x3_stride2_pad1_f32(
            input, packed_weights, bias, output, input_dimensions, output_dimensions);
    }
}
