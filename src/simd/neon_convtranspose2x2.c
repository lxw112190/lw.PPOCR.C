#include "simd_kernels.h"

#include "scalar_kernels.h"

#include <stddef.h>

#if defined(_M_ARM64) || defined(__aarch64__)
#  include <arm_neon.h>
#  define LW_COMPILES_NEON_CONV_TRANSPOSE2X2 1
#else
#  define LW_COMPILES_NEON_CONV_TRANSPOSE2X2 0
#endif

void lw_neon_conv_transpose2x2_stride2_range_f32(
    const float* input, const float* weights, const float* bias, float* output,
    const int32_t input_dimensions[4], const int32_t output_dimensions[4],
    uint32_t output_channel_begin, uint32_t output_channel_end) {
#if LW_COMPILES_NEON_CONV_TRANSPOSE2X2
    const uint32_t input_channels = (uint32_t)input_dimensions[1];
    const uint32_t output_channels = (uint32_t)output_dimensions[1];
    const uint32_t input_height = (uint32_t)input_dimensions[2];
    const uint32_t input_width = (uint32_t)input_dimensions[3];
    const uint32_t output_width = (uint32_t)output_dimensions[3];
    const uint64_t input_plane = (uint64_t)input_height * input_width;
    const uint64_t output_plane =
        (uint64_t)(uint32_t)output_dimensions[2] * output_width;
    const float32x4_t zero = vdupq_n_f32(0.0f);
    uint32_t batch;
    for (batch = 0u; batch < (uint32_t)input_dimensions[0]; ++batch) {
        const float* batch_input =
            input + (size_t)((uint64_t)batch * input_channels * input_plane);
        float* batch_output =
            output + (size_t)((uint64_t)batch * output_channels * output_plane);
        uint32_t output_channel;
        for (output_channel = output_channel_begin; output_channel < output_channel_end;
             ++output_channel) {
            float* output_channel_data =
                batch_output + (size_t)((uint64_t)output_channel * output_plane);
            const float initial_value = bias == NULL ? 0.0f : bias[output_channel];
            const float32x4_t initial = vdupq_n_f32(initial_value);
            uint64_t output_index = 0u;
            uint32_t input_channel;
            for (; output_index + 4u <= output_plane; output_index += 4u) {
                vst1q_f32(output_channel_data + (size_t)output_index, initial);
            }
            for (; output_index < output_plane; ++output_index) {
                output_channel_data[(size_t)output_index] = initial_value;
            }

            for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
                const float* input_channel_data =
                    batch_input + (size_t)((uint64_t)input_channel * input_plane);
                const float* channel_weights =
                    weights + (size_t)(((uint64_t)input_channel * output_channels +
                                        output_channel) * 4u);
                const float weight0 = channel_weights[0];
                const float weight1 = channel_weights[1];
                const float weight2 = channel_weights[2];
                const float weight3 = channel_weights[3];
                uint32_t input_y;
                for (input_y = 0u; input_y < input_height; ++input_y) {
                    const float* input_row =
                        input_channel_data + (size_t)((uint64_t)input_y * input_width);
                    float* output_row0 =
                        output_channel_data + (size_t)((uint64_t)(input_y * 2u) * output_width);
                    float* output_row1 = output_row0 + output_width;
                    uint32_t input_x = 0u;
                    for (; input_x + 4u <= input_width; input_x += 4u) {
                        const float32x4_t values = vld1q_f32(input_row + input_x);
                        const float32x4x2_t even = vzipq_f32(values, zero);
                        const float32x4x2_t odd = vzipq_f32(zero, values);
                        const uint32_t output_x = input_x * 2u;
                        float32x4_t row0_low = vld1q_f32(output_row0 + output_x);
                        float32x4_t row0_high = vld1q_f32(output_row0 + output_x + 4u);
                        float32x4_t row1_low = vld1q_f32(output_row1 + output_x);
                        float32x4_t row1_high = vld1q_f32(output_row1 + output_x + 4u);

                        row0_low = vaddq_f32(row0_low, vmulq_n_f32(even.val[0], weight0));
                        row0_high = vaddq_f32(row0_high, vmulq_n_f32(even.val[1], weight0));
                        row0_low = vaddq_f32(row0_low, vmulq_n_f32(odd.val[0], weight1));
                        row0_high = vaddq_f32(row0_high, vmulq_n_f32(odd.val[1], weight1));
                        row1_low = vaddq_f32(row1_low, vmulq_n_f32(even.val[0], weight2));
                        row1_high = vaddq_f32(row1_high, vmulq_n_f32(even.val[1], weight2));
                        row1_low = vaddq_f32(row1_low, vmulq_n_f32(odd.val[0], weight3));
                        row1_high = vaddq_f32(row1_high, vmulq_n_f32(odd.val[1], weight3));

                        vst1q_f32(output_row0 + output_x, row0_low);
                        vst1q_f32(output_row0 + output_x + 4u, row0_high);
                        vst1q_f32(output_row1 + output_x, row1_low);
                        vst1q_f32(output_row1 + output_x + 4u, row1_high);
                    }
                    for (; input_x < input_width; ++input_x) {
                        const float value = input_row[input_x];
                        const uint32_t output_x = input_x * 2u;
                        output_row0[output_x] += value * weight0;
                        output_row0[output_x + 1u] += value * weight1;
                        output_row1[output_x] += value * weight2;
                        output_row1[output_x + 1u] += value * weight3;
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
    (void)output_channel_begin;
    (void)output_channel_end;
#endif
}

void lw_neon_conv_transpose2x2_stride2_f32(
    const float* input, const float* weights, const float* bias, float* output,
    const int32_t input_dimensions[4], const int32_t output_dimensions[4]) {
    lw_neon_conv_transpose2x2_stride2_range_f32(
        input, weights, bias, output, input_dimensions, output_dimensions, 0u,
        (uint32_t)output_dimensions[1]);
}
