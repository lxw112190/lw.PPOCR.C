#include "simd_kernels.h"

#include <stddef.h>

#if defined(_M_ARM64) || defined(__aarch64__)
#include <arm_neon.h>
#define LW_COMPILES_NEON_DEPTHWISE5X5 1
#else
#define LW_COMPILES_NEON_DEPTHWISE5X5 0
#endif

#if LW_COMPILES_NEON_DEPTHWISE5X5
static float border_pixel(const float* source, const float* weights, float sum,
                          uint32_t width, uint32_t x, uint32_t y,
                          uint32_t first_y, uint32_t end_y) {
    const uint32_t first_x = x < 2u ? 2u - x : 0u;
    const uint32_t end_x = width - x >= 3u ? 5u : width - x + 2u;
    for (uint32_t ky = first_y; ky < end_y; ++ky) {
        const float* row = source + (size_t)(y + ky - 2u) * width;
        const float* taps = weights + ky * 5u;
        for (uint32_t kx = first_x; kx < end_x; ++kx) {
            sum += row[x + kx - 2u] * taps[kx];
        }
    }
    return sum;
}
#endif

void lw_neon_depthwise_conv5x5_unit_pad2_f32(const float* input, const float* weights,
                                            const float* bias, float* output,
                                            const int32_t dimensions[4]) {
#if LW_COMPILES_NEON_DEPTHWISE5X5
    const uint32_t channels = (uint32_t)dimensions[1];
    const uint32_t height = (uint32_t)dimensions[2];
    const uint32_t width = (uint32_t)dimensions[3];
    const size_t plane = (size_t)height * width;
    if (width >= 8u) {
        for (uint32_t batch = 0u; batch < (uint32_t)dimensions[0]; ++batch) {
            for (uint32_t channel = 0u; channel < channels; ++channel) {
                const size_t offset = ((size_t)batch * channels + channel) * plane;
                const float* source = input + offset;
                const float* kernel = weights + (size_t)channel * 25u;
                float* destination = output + offset;
                const float initial = bias == NULL ? 0.0f : bias[channel];
                const float32x4_t initial_vector = vdupq_n_f32(initial);
                for (uint32_t y = 0u; y < height; ++y) {
                    const uint32_t first_y = y < 2u ? 2u - y : 0u;
                    const uint32_t end_y = height - y >= 3u ? 5u : height - y + 2u;
                    float* row_out = destination + (size_t)y * width;
                    uint32_t x = 2u;
                    row_out[0] = border_pixel(source, kernel, initial, width, 0u, y, first_y, end_y);
                    row_out[1] = border_pixel(source, kernel, initial, width, 1u, y, first_y, end_y);
                    /* Accumulate rows and taps in canonical order and store each tile once. */
                    for (; x + 5u < width; x += 4u) {
                        float32x4_t sum = initial_vector;
                        for (uint32_t ky = first_y; ky < end_y; ++ky) {
                            const float* row = source + (size_t)(y + ky - 2u) * width + x - 2u;
                            const float* taps = kernel + ky * 5u;
                            sum = vaddq_f32(sum, vmulq_n_f32(vld1q_f32(row), taps[0]));
                            sum = vaddq_f32(sum, vmulq_n_f32(vld1q_f32(row + 1u), taps[1]));
                            sum = vaddq_f32(sum, vmulq_n_f32(vld1q_f32(row + 2u), taps[2]));
                            sum = vaddq_f32(sum, vmulq_n_f32(vld1q_f32(row + 3u), taps[3]));
                            sum = vaddq_f32(sum, vmulq_n_f32(vld1q_f32(row + 4u), taps[4]));
                        }
                        vst1q_f32(row_out + x, sum);
                    }
                    for (; x < width; ++x) {
                        row_out[x] = border_pixel(source, kernel, initial, width, x, y, first_y,
                                                  end_y);
                    }
                }
            }
        }
        return;
    }
#endif
    lw_scalar_depthwise_conv5x5_unit_pad2_f32(input, weights, bias, output, dimensions);
}
