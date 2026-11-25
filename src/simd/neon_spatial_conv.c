#include "simd_kernels.h"

#include <string.h>

#define LW_NEON_SPATIAL_MAX_CHANNELS 192u
#define LW_NEON_SPATIAL_PIXELS 16u

#if defined(_M_ARM64) || defined(__aarch64__)
#  include <arm_neon.h>
#  define LW_COMPILES_NEON_SPATIAL_CONV 1
#else
#  define LW_COMPILES_NEON_SPATIAL_CONV 0
#endif

int lw_neon_spatial_conv_supported(const int32_t input[4], const int32_t output[4],
                                  uint32_t kernel, uint32_t stride) {
#if LW_COMPILES_NEON_SPATIAL_CONV
    return ((kernel == 2u && stride == 1u) || (kernel == 3u && stride == 2u)) &&
           input[0] == 1 && output[0] == 1 && input[1] >= 8 &&
           input[1] <= (int32_t)LW_NEON_SPATIAL_MAX_CHANNELS && input[2] >= 6 &&
           input[3] >= 64 && output[1] >= 8 && output[1] % 4 == 0 &&
           output[2] == (stride == 1u ? input[2] : input[2] / 2 + input[2] % 2) &&
           output[3] == (stride == 1u ? input[3] : input[3] / 2 + input[3] % 2);
#else
    (void)input;
    (void)output;
    (void)kernel;
    (void)stride;
    return 0;
#endif
}

#if LW_COMPILES_NEON_SPATIAL_CONV
typedef struct lw_neon_spatial_conv {
    const float* input;
    const float* weights;
    const float* bias;
    float* output;
    size_t input_plane;
    size_t output_plane;
    uint32_t input_width;
    uint32_t input_height;
    uint32_t input_channels;
    uint32_t output_width;
    uint32_t output_channels;
    uint32_t kernel;
    uint32_t stride;
    uint32_t pad;
} lw_neon_spatial_conv;

static void pack_row(float* tile, const float* source,
                     uint32_t kernel, uint32_t stride, uint32_t pixels) {
    uint32_t x = 0u;
    for (; x + 4u <= pixels; x += 4u) {
        const float* src = source + (size_t)x * stride;
        if (stride == 2u) {
            const float32x4x2_t pair = vld2q_f32(src);
            const float32x4_t right = vextq_f32(pair.val[0], vdupq_n_f32(src[8]), 1);
            vst1q_f32(tile + x, pair.val[0]);
            vst1q_f32(tile + LW_NEON_SPATIAL_PIXELS + x, pair.val[1]);
            vst1q_f32(tile + 2u * LW_NEON_SPATIAL_PIXELS + x, right);
        } else {
            vst1q_f32(tile + x, vld1q_f32(src));
            vst1q_f32(tile + LW_NEON_SPATIAL_PIXELS + x, vld1q_f32(src + 1u));
        }
    }
    for (uint32_t tap = 0u; tap < kernel; ++tap) {
        float* packed = tile + (size_t)tap * LW_NEON_SPATIAL_PIXELS;
        uint32_t lane = x;
        for (; lane < pixels; ++lane) {
            packed[lane] = source[(size_t)lane * stride + tap];
        }
        for (; lane < LW_NEON_SPATIAL_PIXELS; ++lane) {
            packed[lane] = 0.0f;
        }
    }
}

static void store_tile(float* dst, float32x4x4_t sum, uint32_t pixels) {
    if (pixels == LW_NEON_SPATIAL_PIXELS) {
        vst1q_f32_x4(dst, sum);
    } else {
        float values[LW_NEON_SPATIAL_PIXELS];
        vst1q_f32_x4(values, sum);
        memcpy(dst, values, (size_t)pixels * sizeof(*dst));
    }
}

static void compute_tile(const lw_neon_spatial_conv* conv, const float* tile,
                         uint32_t first_y, uint32_t end_y, uint32_t pixels, size_t offset) {
    const uint32_t taps = (end_y - first_y) * conv->kernel;
    const uint32_t kernel_area = conv->kernel * conv->kernel;
    const size_t block_weights = (size_t)conv->input_channels * kernel_area * 4u;
    for (uint32_t oc = 0u; oc < conv->output_channels; oc += 4u) {
        const float* coefficients = conv->weights + (size_t)(oc / 4u) * block_weights +
                                    (size_t)first_y * conv->kernel * 4u;
        const float32x4_t b0 = vdupq_n_f32(conv->bias == NULL ? 0.0f : conv->bias[oc]);
        const float32x4_t b1 = vdupq_n_f32(conv->bias == NULL ? 0.0f : conv->bias[oc + 1u]);
        const float32x4_t b2 = vdupq_n_f32(conv->bias == NULL ? 0.0f : conv->bias[oc + 2u]);
        const float32x4_t b3 = vdupq_n_f32(conv->bias == NULL ? 0.0f : conv->bias[oc + 3u]);
        float32x4x4_t sum0 = {{b0, b0, b0, b0}};
        float32x4x4_t sum1 = {{b1, b1, b1, b1}};
        float32x4x4_t sum2 = {{b2, b2, b2, b2}};
        float32x4x4_t sum3 = {{b3, b3, b3, b3}};
        const float* data = tile;
        for (uint32_t ic = 0u; ic < conv->input_channels; ++ic) {
            const float* weights = coefficients + (size_t)ic * kernel_area * 4u;
            for (uint32_t tap = 0u; tap < taps; ++tap) {
                const float32x4x4_t values = vld1q_f32_x4(data);
                const float32x4_t w = vld1q_f32(weights + (size_t)tap * 4u);
                for (uint32_t vector = 0u; vector < 4u; ++vector) {
                    const float32x4_t v = values.val[vector];
                    sum0.val[vector] = vaddq_f32(sum0.val[vector], vmulq_laneq_f32(v, w, 0));
                    sum1.val[vector] = vaddq_f32(sum1.val[vector], vmulq_laneq_f32(v, w, 1));
                    sum2.val[vector] = vaddq_f32(sum2.val[vector], vmulq_laneq_f32(v, w, 2));
                    sum3.val[vector] = vaddq_f32(sum3.val[vector], vmulq_laneq_f32(v, w, 3));
                }
                data += LW_NEON_SPATIAL_PIXELS;
            }
        }
        float* dst = conv->output + (size_t)oc * conv->output_plane + offset;
        store_tile(dst, sum0, pixels);
        store_tile(dst + conv->output_plane, sum1, pixels);
        store_tile(dst + 2u * conv->output_plane, sum2, pixels);
        store_tile(dst + 3u * conv->output_plane, sum3, pixels);
    }
}

/* Border taps outside the source are skipped, including for non-finite weights. */
static void compute_edge(const lw_neon_spatial_conv* conv, uint32_t x, uint32_t y,
                         uint32_t first_y, uint32_t end_y) {
    const int64_t x0 = (int64_t)x * conv->stride - conv->pad;
    const int64_t y0 = (int64_t)y * conv->stride - conv->pad;
    const uint32_t kernel_area = conv->kernel * conv->kernel;
    const size_t block_weights = (size_t)conv->input_channels * kernel_area * 4u;
    const size_t offset = (size_t)y * conv->output_width + x;
    for (uint32_t oc = 0u; oc < conv->output_channels; oc += 4u) {
        const float* coefficients = conv->weights + (size_t)(oc / 4u) * block_weights;
        float32x4_t sum = conv->bias == NULL ? vdupq_n_f32(0.0f) : vld1q_f32(conv->bias + oc);
        for (uint32_t ic = 0u; ic < conv->input_channels; ++ic) {
            const float* weights = coefficients + (size_t)ic * kernel_area * 4u;
            const float* plane = conv->input + (size_t)ic * conv->input_plane;
            for (uint32_t ky = first_y; ky < end_y; ++ky) {
                const float* row = plane + (size_t)(y0 + ky) * conv->input_width;
                for (uint32_t kx = 0u; kx < conv->kernel; ++kx) {
                    const int64_t ix = x0 + kx;
                    if (ix >= 0 && ix < conv->input_width) {
                        const float32x4_t w = vld1q_f32(weights + (ky * conv->kernel + kx) * 4u);
                        sum = vaddq_f32(sum, vmulq_f32(vdupq_n_f32(row[(size_t)ix]), w));
                    }
                }
            }
        }
        float* dst = conv->output + (size_t)oc * conv->output_plane + offset;
        vst1_lane_f32(dst, vget_low_f32(sum), 0);
        vst1_lane_f32(dst + conv->output_plane, vget_low_f32(sum), 1);
        vst1_lane_f32(dst + 2u * conv->output_plane, vget_high_f32(sum), 0);
        vst1_lane_f32(dst + 3u * conv->output_plane, vget_high_f32(sum), 1);
    }
}
#endif

void lw_neon_spatial_conv_rows_f32(const float* input, const float* weights,
                                   const float* bias, float* output,
                                   const int32_t input_dims[4], const int32_t output_dims[4],
                                   uint32_t kernel, uint32_t stride,
                                   uint32_t row_begin, uint32_t row_end) {
#if LW_COMPILES_NEON_SPATIAL_CONV
    const uint32_t pad = stride == 2u ? 1u : 0u;
    const lw_neon_spatial_conv conv = {
        input, weights, bias, output,
        (size_t)input_dims[2] * (size_t)input_dims[3],
        (size_t)output_dims[2] * (size_t)output_dims[3],
        (uint32_t)input_dims[3], (uint32_t)input_dims[2], (uint32_t)input_dims[1],
        (uint32_t)output_dims[3], (uint32_t)output_dims[1], kernel, stride, pad,
    };
    const uint32_t interior_end = (conv.input_width + pad - kernel) / stride + 1u;
    /* One input window tile per worker. For each output row, clipping the
     * vertical tap range keeps padding out of both the input tile and the sum. */
    float tile[LW_NEON_SPATIAL_MAX_CHANNELS * 9u * LW_NEON_SPATIAL_PIXELS];
    for (uint32_t y = row_begin; y < row_end; ++y) {
        const int64_t y0 = (int64_t)y * stride - pad;
        const uint32_t first_y = y0 < 0 ? 1u : 0u;
        const uint32_t end_y = y0 + kernel > conv.input_height ? kernel - 1u : kernel;
        uint32_t x = 0u;
        for (; x < pad; ++x) {
            compute_edge(&conv, x, y, first_y, end_y);
        }
        while (x < interior_end) {
            const uint32_t remaining = interior_end - x;
            const uint32_t pixels = remaining < LW_NEON_SPATIAL_PIXELS
                                        ? remaining
                                        : LW_NEON_SPATIAL_PIXELS;
            const size_t column = (size_t)x * stride - pad;
            size_t packed_rows = 0u;
            for (uint32_t ic = 0u; ic < conv.input_channels; ++ic) {
                const float* plane = input + (size_t)ic * conv.input_plane;
                for (uint32_t ky = first_y; ky < end_y; ++ky) {
                    const float* src = plane + (size_t)(y0 + ky) * conv.input_width + column;
                    pack_row(tile + packed_rows * LW_NEON_SPATIAL_PIXELS, src, kernel, stride, pixels);
                    packed_rows += kernel;
                }
            }
            compute_tile(&conv, tile, first_y, end_y, pixels,
                         (size_t)y * conv.output_width + x);
            x += pixels;
        }
        for (; x < conv.output_width; ++x) {
            compute_edge(&conv, x, y, first_y, end_y);
        }
    }
#else
    (void)input;
    (void)weights;
    (void)bias;
    (void)output;
    (void)input_dims;
    (void)output_dims;
    (void)kernel;
    (void)stride;
    (void)row_begin;
    (void)row_end;
#endif
}
