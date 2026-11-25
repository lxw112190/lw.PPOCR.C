#include "simd_kernels.h"

#include "scalar_kernels.h"

#include <stddef.h>

#if defined(_M_ARM64) || defined(__aarch64__)
#  include <arm_neon.h>
#  define LW_COMPILES_NEON_CONV3X3 1
#else
#  define LW_COMPILES_NEON_CONV3X3 0
#endif

void lw_neon_conv3x3_unit_pad1_f32(const float* input, const float* weights, const float* bias,
                                   float* output, const int32_t input_dimensions[4],
                                   const int32_t output_dimensions[4]) {
#if LW_COMPILES_NEON_CONV3X3
    const uint32_t height = (uint32_t)input_dimensions[2];
    const uint32_t width = (uint32_t)input_dimensions[3];
    const uint64_t channel_plane = (uint64_t)height * width;
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
                weights + (size_t)((uint64_t)output_channel *
                                   (uint32_t)input_dimensions[1] * 9u);
            float* output_channel_data =
                batch_output + (size_t)((uint64_t)output_channel * channel_plane);
            const float initial = bias == NULL ? 0.0f : bias[output_channel];
            const float32x4_t initial_values = vdupq_n_f32(initial);
            uint64_t spatial = 0u;
            uint32_t input_channel;
            for (; spatial + 4u <= channel_plane; spatial += 4u) {
                vst1q_f32(output_channel_data + (size_t)spatial, initial_values);
            }
            for (; spatial < channel_plane; ++spatial) {
                output_channel_data[(size_t)spatial] = initial;
            }
            for (input_channel = 0u; input_channel < (uint32_t)input_dimensions[1];
                 ++input_channel) {
                const float* input_channel_data =
                    batch_input + (size_t)((uint64_t)input_channel * channel_plane);
                const float* weight_channel_data =
                    output_channel_weights + (size_t)((uint64_t)input_channel * 9u);
                uint32_t kernel_y;
                for (kernel_y = 0u; kernel_y < 3u; ++kernel_y) {
                    const uint32_t output_y_begin = kernel_y == 0u ? 1u : 0u;
                    const uint32_t output_y_end = kernel_y == 2u ? height - 1u : height;
                    uint32_t kernel_x;
                    for (kernel_x = 0u; kernel_x < 3u; ++kernel_x) {
                        const uint32_t output_x_begin = kernel_x == 0u ? 1u : 0u;
                        const uint32_t output_x_end = kernel_x == 2u ? width - 1u : width;
                        const float weight = weight_channel_data[kernel_y * 3u + kernel_x];
                        uint32_t output_y;
                        for (output_y = output_y_begin; output_y < output_y_end; ++output_y) {
                            const uint32_t input_y = output_y + kernel_y - 1u;
                            const float* input_row =
                                input_channel_data + (size_t)((uint64_t)input_y * width);
                            float* output_row =
                                output_channel_data + (size_t)((uint64_t)output_y * width);
                            uint32_t output_x = output_x_begin;
                            for (; output_x + 4u <= output_x_end; output_x += 4u) {
                                const uint32_t input_x = output_x + kernel_x - 1u;
                                const float32x4_t input_values = vld1q_f32(input_row + input_x);
                                const float32x4_t output_values =
                                    vld1q_f32(output_row + output_x);
                                vst1q_f32(output_row + output_x,
                                          vaddq_f32(output_values,
                                                    vmulq_n_f32(input_values, weight)));
                            }
                            for (; output_x < output_x_end; ++output_x) {
                                const uint32_t input_x = output_x + kernel_x - 1u;
                                output_row[output_x] += input_row[input_x] * weight;
                            }
                        }
                    }
                }
            }
        }
    }
#else
    lw_scalar_conv3x3_unit_pad1_f32(input, weights, bias, output, input_dimensions,
                                    output_dimensions);
#endif
}

void lw_neon_depthwise_conv3x3_pad1_f32(
    const float* input, const float* weights, const float* bias, float* output,
    const int32_t input_dimensions[4], const int32_t output_dimensions[4], uint32_t stride_y) {
#if LW_COMPILES_NEON_CONV3X3
    const uint32_t channels = (uint32_t)input_dimensions[1];
    const uint32_t height = (uint32_t)input_dimensions[2];
    const uint32_t width = (uint32_t)input_dimensions[3];
    const uint32_t output_height = (uint32_t)output_dimensions[2];
    if (width >= 6u) {
        const size_t input_plane = (size_t)height * width;
        const size_t output_plane = (size_t)output_height * width;
        for (uint32_t batch = 0u; batch < (uint32_t)input_dimensions[0]; ++batch) {
            for (uint32_t channel = 0u; channel < channels; ++channel) {
                const size_t plane = (size_t)batch * channels + channel;
                const float* src = input + plane * input_plane;
                const float* kernel = weights + (size_t)channel * 9u;
                float* dst = output + plane * output_plane;
                const float initial = bias == NULL ? 0.0f : bias[channel];
                const float32x4_t initial_vector = vdupq_n_f32(initial);
                for (uint32_t y = 0u; y < output_height; ++y) {
                    const uint32_t center = y * stride_y;
                    const uint32_t first_tap = center == 0u ? 1u : 0u;
                    const uint32_t end_tap = center + 1u < height ? 3u : 2u;
                    float* out = dst + (size_t)y * width;
                    uint32_t x = 1u;
                    float left = initial;
                    for (uint32_t ky = first_tap; ky < end_tap; ++ky) {
                        const float* row = src + (size_t)(center + ky - 1u) * width;
                        const float* tap = kernel + ky * 3u;
                        left += row[0] * tap[1];
                        left += row[1] * tap[2];
                    }
                    out[0] = left;
                    for (; x + 4u < width; x += 4u) {
                        float32x4_t sum = initial_vector;
                        for (uint32_t ky = first_tap; ky < end_tap; ++ky) {
                            const float* row = src + (size_t)(center + ky - 1u) * width + x - 1u;
                            const float* tap = kernel + ky * 3u;
                            sum = vaddq_f32(sum, vmulq_n_f32(vld1q_f32(row), tap[0]));
                            sum = vaddq_f32(sum, vmulq_n_f32(vld1q_f32(row + 1u), tap[1]));
                            sum = vaddq_f32(sum, vmulq_n_f32(vld1q_f32(row + 2u), tap[2]));
                        }
                        vst1q_f32(out + x, sum);
                    }
                    for (; x < width; ++x) {
                        float sum = initial;
                        for (uint32_t ky = first_tap; ky < end_tap; ++ky) {
                            const float* row = src + (size_t)(center + ky - 1u) * width;
                            const float* tap = kernel + ky * 3u;
                            sum += row[x - 1u] * tap[0];
                            sum += row[x] * tap[1];
                            if (x + 1u < width) {
                                sum += row[x + 1u] * tap[2];
                            }
                        }
                        out[x] = sum;
                    }
                }
            }
        }
        return;
    }
#endif
    if (stride_y == 1u) {
        lw_scalar_depthwise_conv3x3_unit_pad1_f32(input, weights, bias, output, input_dimensions);
    } else {
        lw_scalar_depthwise_conv3x3_stride2x1_pad1_f32(
            input, weights, bias, output, input_dimensions, output_dimensions);
    }
}

#if LW_COMPILES_NEON_CONV3X3
static float stride2_edge_pixel(const float* input, const float* weights, float sum,
                                size_t input_plane, uint32_t channels,
                                uint32_t input_width, uint32_t input_height,
                                uint32_t output_y, uint32_t output_x) {
    const int64_t y = (int64_t)output_y * 2 - 1;
    const int64_t x = (int64_t)output_x * 2 - 1;
    uint32_t c;
    for (c = 0u; c < channels; ++c) {
        uint32_t ky;
        for (ky = 0u; ky < 3u; ++ky) {
            const int64_t iy = y + ky;
            uint32_t kx;
            const float* row;
            if (iy < 0 || iy >= input_height) {
                continue;
            }
            row = input + (size_t)iy * input_width;
            for (kx = 0u; kx < 3u; ++kx) {
                const int64_t ix = x + kx;
                if (ix >= 0 && ix < input_width) {
                    sum += row[(size_t)ix] * weights[ky * 3u + kx];
                }
            }
        }
        input += input_plane;
        weights += 9u;
    }
    return sum;
}
#endif

void lw_neon_conv3x3_stride2_pad1_f32(const float* input, const float* weights, const float* bias,
                                      float* output, const int32_t input_dimensions[4],
                                      const int32_t output_dimensions[4]) {
#if LW_COMPILES_NEON_CONV3X3
    const uint32_t channels = (uint32_t)input_dimensions[1];
    const uint32_t input_height = (uint32_t)input_dimensions[2];
    const uint32_t input_width = (uint32_t)input_dimensions[3];
    const uint32_t output_channels = (uint32_t)output_dimensions[1];
    const uint32_t output_height = (uint32_t)output_dimensions[2];
    const uint32_t output_width = (uint32_t)output_dimensions[3];
    const size_t input_plane = (size_t)input_height * input_width;
    const size_t output_plane = (size_t)output_height * output_width;
    /* A complete horizontal tap window ends at 2 * x + 1. */
    const uint32_t interior_end = input_width / 2u;
    uint32_t batch;
    if (interior_end < 5u) {
        lw_scalar_conv3x3_stride2_pad1_f32(input, weights, bias, output, input_dimensions,
                                           output_dimensions);
        return;
    }
    for (batch = 0u; batch < (uint32_t)input_dimensions[0]; ++batch) {
        const float* batch_input = input + (size_t)batch * channels * input_plane;
        float* batch_output = output + (size_t)batch * output_channels * output_plane;
        uint32_t oc;
        /* Adjacent output channels use the same three input vectors per tap row.
         * Keep separate accumulators so each channel retains its tap order. */
        for (oc = 0u; oc < output_channels; oc += 2u) {
            const float* kernel = weights + (size_t)oc * channels * 9u;
            const float initial = bias == NULL ? 0.0f : bias[oc];
            const int paired = oc + 1u < output_channels;
            const float* next_kernel = paired ? kernel + (size_t)channels * 9u : kernel;
            const float next_initial = paired && bias != NULL ? bias[oc + 1u] : initial;
            uint32_t y;
            for (y = 0u; y < output_height; ++y) {
                float* dst = batch_output + (size_t)oc * output_plane + (size_t)y * output_width;
                float* next_dst = paired ? dst + output_plane : NULL;
                const uint32_t ky_begin = y == 0u ? 1u : 0u;
                const uint32_t ky_end = (uint64_t)y * 2u + 1u >= input_height ? 2u : 3u;
                uint32_t x = 1u;
                dst[0] = stride2_edge_pixel(batch_input, kernel, initial, input_plane, channels,
                                            input_width, input_height, y, 0u);
                if (paired) {
                    next_dst[0] = stride2_edge_pixel(batch_input, next_kernel, next_initial,
                                                    input_plane, channels, input_width,
                                                    input_height, y, 0u);
                }
                for (; x + 4u <= interior_end; x += 4u) {
                    const size_t left = (size_t)x * 2u - 1u;
                    const float* src = batch_input;
                    const float* w = kernel;
                    const float* next_w = next_kernel;
                    float32x4_t sum = vdupq_n_f32(initial);
                    float32x4_t next_sum = vdupq_n_f32(next_initial);
                    uint32_t c;
                    for (c = 0u; c < channels; ++c) {
                        uint32_t ky;
                        for (ky = ky_begin; ky < ky_end; ++ky) {
                            const size_t iy = (size_t)y * 2u + ky - 1u;
                            const float* row = src + iy * input_width + left;
                            /* Deinterleave the left and center taps and slide the left
                             * vector forward for the right tap. Preserve tap order. */
                            const float32x4x2_t pair = vld2q_f32(row);
                            const float32x4_t right =
                                vextq_f32(pair.val[0], vdupq_n_f32(row[8]), 1);
                            const float* tap = w + ky * 3u;
                            sum = vaddq_f32(sum, vmulq_n_f32(pair.val[0], tap[0]));
                            sum = vaddq_f32(sum, vmulq_n_f32(pair.val[1], tap[1]));
                            sum = vaddq_f32(sum, vmulq_n_f32(right, tap[2]));
                            if (paired) {
                                const float* next_tap = next_w + ky * 3u;
                                next_sum = vaddq_f32(next_sum, vmulq_n_f32(pair.val[0], next_tap[0]));
                                next_sum = vaddq_f32(next_sum, vmulq_n_f32(pair.val[1], next_tap[1]));
                                next_sum = vaddq_f32(next_sum, vmulq_n_f32(right, next_tap[2]));
                            }
                        }
                        src += input_plane;
                        w += 9u;
                        next_w += 9u;
                    }
                    vst1q_f32(dst + x, sum);
                    if (paired) {
                        vst1q_f32(next_dst + x, next_sum);
                    }
                }
                for (; x < output_width; ++x) {
                    dst[x] = stride2_edge_pixel(batch_input, kernel, initial, input_plane, channels,
                                                input_width, input_height, y, x);
                    if (paired) {
                        next_dst[x] = stride2_edge_pixel(batch_input, next_kernel, next_initial,
                                                        input_plane, channels, input_width,
                                                        input_height, y, x);
                    }
                }
            }
        }
    }
#else
    lw_scalar_conv3x3_stride2_pad1_f32(input, weights, bias, output, input_dimensions,
                                       output_dimensions);
#endif
}
