#include "nhwc_internal.h"

#include <stddef.h>
#include <stdint.h>

#if defined(__EMSCRIPTEN__) && defined(LW_WASM_REC_SIMD_KERNELS)
#include <wasm_simd128.h>
#define LW_DW_WASM128 1
#else
#define LW_DW_WASM128 0
#endif

#if !defined(LW_NHWC_FORCE_PORTABLE) && \
    (defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__))
#include <immintrin.h>
#define LW_DW_X86 1
#else
#define LW_DW_X86 0
#endif

#if LW_DW_X86 && (defined(__GNUC__) || defined(__clang__))
#define LW_DW_TARGET __attribute__((target("avx2,fma")))
#else
#define LW_DW_TARGET
#endif

#if LW_DW_X86
LW_DW_TARGET
static void depthwise_block32(const float* input, const float* weights, const float* bias,
                              float* output, const lw_nhwc_depthwise_desc* desc,
                              uint32_t output_y, uint32_t output_x, uint32_t channel_base,
                              uint32_t vector_count, uint32_t output_pixel_base) {
    __m256 sums[4];
    int32_t input_y0 = (int32_t)((uint64_t)(output_y + desc->output_row_offset) * desc->stride_h) -
        (int32_t)desc->pad_top;
    int32_t input_x0 = (int32_t)((uint64_t)output_x * desc->stride_w) - (int32_t)desc->pad_left;
    for (uint32_t lane = 0u; lane < vector_count; ++lane) {
        sums[lane] = bias == NULL ? _mm256_setzero_ps() :
            _mm256_loadu_ps(bias + channel_base + lane * 8u);
    }
    uint32_t tap = 0u;
    for (uint32_t ky = 0u; ky < desc->kernel_h; ++ky) {
        int32_t iy = input_y0 + (int32_t)ky;
        for (uint32_t kx = 0u; kx < desc->kernel_w; ++kx, ++tap) {
            int32_t ix = input_x0 + (int32_t)kx;
            if (iy < 0 || iy >= (int32_t)desc->input_height || ix < 0 ||
                ix >= (int32_t)desc->input_width) continue;
            const float* source = input + (((size_t)iy * desc->input_width + (uint32_t)ix) *
                                           desc->channels + channel_base);
            const float* weight = weights + (size_t)tap * LW_NHWC_DEPTHWISE_BLOCK;
            for (uint32_t lane = 0u; lane < vector_count; ++lane) {
                sums[lane] = _mm256_fmadd_ps(_mm256_loadu_ps(source + lane * 8u),
                                             _mm256_loadu_ps(weight + lane * 8u), sums[lane]);
            }
        }
    }
    float* destination = output + (((size_t)output_y * desc->output_width + output_x - output_pixel_base) *
                                   desc->channels + channel_base);
    if (desc->post_bias != NULL) {
        for (uint32_t lane = 0u; lane < vector_count; ++lane) {
            sums[lane] = _mm256_add_ps(sums[lane],
                _mm256_loadu_ps(desc->post_bias + channel_base + lane * 8u));
        }
    }
    for (uint32_t lane = 0u; lane < vector_count; ++lane) _mm256_storeu_ps(destination + lane * 8u, sums[lane]);
}
#endif


#if LW_DW_X86
LW_DW_TARGET
static int depthwise_pair_is_interior(const lw_nhwc_depthwise_desc* desc,
                                      uint32_t output_y, uint32_t output_x) {
    int32_t input_y0 = (int32_t)((uint64_t)(output_y + desc->output_row_offset) * desc->stride_h) -
        (int32_t)desc->pad_top;
    int32_t input_x0 = (int32_t)((uint64_t)output_x * desc->stride_w) - (int32_t)desc->pad_left;
    return input_y0 >= 0 &&
           input_y0 + (int32_t)desc->kernel_h <= (int32_t)desc->input_height &&
           input_x0 >= 0 &&
           input_x0 + (int32_t)desc->stride_w + (int32_t)desc->kernel_w <=
               (int32_t)desc->input_width;
}

LW_DW_TARGET
static void depthwise_block32_x2(const float* input, const float* weights, const float* bias,
                                  float* output, const lw_nhwc_depthwise_desc* desc,
                                  uint32_t output_y, uint32_t output_x, uint32_t channel_base,
                                  uint32_t output_pixel_base) {
    __m256 sum0[4];
    __m256 sum1[4];
    int32_t input_y0 = (int32_t)((uint64_t)(output_y + desc->output_row_offset) * desc->stride_h) -
        (int32_t)desc->pad_top;
    int32_t input_x0 = (int32_t)((uint64_t)output_x * desc->stride_w) - (int32_t)desc->pad_left;
    for (uint32_t lane = 0u; lane < 4u; ++lane) {
        __m256 initial = bias == NULL ? _mm256_setzero_ps() :
            _mm256_loadu_ps(bias + channel_base + lane * 8u);
        sum0[lane] = initial;
        sum1[lane] = initial;
    }
    uint32_t tap = 0u;
    for (uint32_t ky = 0u; ky < desc->kernel_h; ++ky) {
        for (uint32_t kx = 0u; kx < desc->kernel_w; ++kx, ++tap) {
            const float* source0 = input + (((size_t)(input_y0 + (int32_t)ky) * desc->input_width +
                (uint32_t)(input_x0 + (int32_t)kx)) * desc->channels + channel_base);
            const float* source1 = input + (((size_t)(input_y0 + (int32_t)ky) * desc->input_width +
                (uint32_t)(input_x0 + (int32_t)desc->stride_w + (int32_t)kx)) *
                desc->channels + channel_base);
            const float* weight = weights + (size_t)tap * LW_NHWC_DEPTHWISE_BLOCK;
            for (uint32_t lane = 0u; lane < 4u; ++lane) {
                __m256 w = _mm256_loadu_ps(weight + lane * 8u);
                sum0[lane] = _mm256_fmadd_ps(_mm256_loadu_ps(source0 + lane * 8u), w, sum0[lane]);
                sum1[lane] = _mm256_fmadd_ps(_mm256_loadu_ps(source1 + lane * 8u), w, sum1[lane]);
            }
        }
    }
    float* destination0 = output + (((size_t)output_y * desc->output_width + output_x - output_pixel_base) *
        desc->channels + channel_base);
    float* destination1 = destination0 + desc->channels;
    if (desc->post_bias != NULL) {
        for (uint32_t lane = 0u; lane < 4u; ++lane) {
            __m256 post = _mm256_loadu_ps(desc->post_bias + channel_base + lane * 8u);
            sum0[lane] = _mm256_add_ps(sum0[lane], post);
            sum1[lane] = _mm256_add_ps(sum1[lane], post);
        }
    }
    for (uint32_t lane = 0u; lane < 4u; ++lane) {
        _mm256_storeu_ps(destination0 + lane * 8u, sum0[lane]);
        _mm256_storeu_ps(destination1 + lane * 8u, sum1[lane]);
    }
}
#endif
lw_status lw_avx2_fma_nhwc_depthwise_range_f32(const float* input,
    const float* packed_weights, const float* bias, float* output,
    const lw_nhwc_depthwise_desc* desc, uint32_t begin, uint32_t count) {
    uint64_t pixels;
    if (input == NULL || packed_weights == NULL || output == NULL || desc == NULL ||
        desc->batch != 1u || desc->channels == 0u || (desc->channels & 7u) != 0u ||
        desc->input_height == 0u || desc->input_width == 0u ||
        desc->output_height == 0u || desc->output_width == 0u ||
        desc->kernel_h == 0u || desc->kernel_w == 0u ||
        desc->stride_h == 0u || desc->stride_w == 0u ||
        desc->kernel_h > UINT32_MAX / desc->kernel_w ||
        desc->output_row_offset != 0u || count == 0u) return LW_STATUS_INVALID_ARGUMENT;
    pixels = (uint64_t)desc->output_height * desc->output_width;
    if (pixels > UINT32_MAX || begin >= pixels || count > pixels - begin ||
        (uint64_t)desc->input_height * desc->input_width >
            SIZE_MAX / ((uint64_t)desc->channels * sizeof(float)) ||
        pixels > SIZE_MAX / ((uint64_t)desc->channels * sizeof(float)) ||
        (uint64_t)desc->input_height + desc->pad_top + desc->pad_bottom < desc->kernel_h ||
        (uint64_t)desc->input_width + desc->pad_left + desc->pad_right < desc->kernel_w ||
        ((uint64_t)desc->input_height + desc->pad_top + desc->pad_bottom - desc->kernel_h) /
            desc->stride_h + 1u != desc->output_height ||
        ((uint64_t)desc->input_width + desc->pad_left + desc->pad_right - desc->kernel_w) /
            desc->stride_w + 1u != desc->output_width) return LW_STATUS_INVALID_SHAPE;
#if LW_DW_X86
    {
        uint32_t taps = desc->kernel_h * desc->kernel_w;
        uint32_t blocks = (desc->channels - 1u) / 32u + 1u;
        uint32_t pixel = begin, end = begin + count;
        while (pixel < end) {
            uint32_t y = pixel / desc->output_width;
            uint32_t x = pixel % desc->output_width;
            uint32_t run = end - pixel;
            if (run > desc->output_width - x) run = desc->output_width - x;
            for (uint32_t block = 0u; block < blocks; ++block) {
                uint32_t channels = desc->channels - block * 32u;
                uint32_t ox = x, limit = x + run;
                const float* weights = packed_weights + (size_t)block * taps * 32u;
                if (channels > 32u) channels = 32u;
                while (ox < limit) {
                    if (ox + 1u < limit && channels == 32u &&
                        depthwise_pair_is_interior(desc, y, ox)) {
                        depthwise_block32_x2(input, weights, bias, output, desc,
                                             y, ox, block * 32u, begin);
                        ox += 2u;
                    } else {
                        depthwise_block32(input, weights, bias, output, desc,
                            y, ox, block * 32u, channels / 8u, begin);
                        ++ox;
                    }
                }
            }
            pixel += run;
        }
    }
    return LW_STATUS_OK;
#else
    (void)bias;
    return LW_STATUS_UNSUPPORTED;
#endif
}

static lw_status depthwise_execute(const float* input, const float* packed_weights,
                                   const float* bias, float* output,
                                   const lw_nhwc_depthwise_desc* desc,
                                   lw_nhwc_depthwise_stats* stats) {
    if (input == NULL || packed_weights == NULL || output == NULL || desc == NULL ||
        desc->batch == 0u || desc->channels == 0u ||
        (desc->channels & 7u) != 0u ||
        desc->input_height == 0u || desc->input_width == 0u || desc->output_height == 0u ||
        desc->output_width == 0u || desc->kernel_h == 0u || desc->kernel_w == 0u ||
        desc->stride_h == 0u || desc->stride_w == 0u) return LW_STATUS_INVALID_ARGUMENT;
    if ((uint64_t)desc->input_height + desc->pad_top + desc->pad_bottom < desc->kernel_h ||
        (uint64_t)desc->input_width + desc->pad_left + desc->pad_right < desc->kernel_w ||
        (uint64_t)desc->output_height + desc->output_row_offset >
            (desc->input_height + (uint64_t)desc->pad_top + desc->pad_bottom - desc->kernel_h) /
                desc->stride_h + 1u ||
        ((uint64_t)desc->input_width + desc->pad_left + desc->pad_right - desc->kernel_w) /
            desc->stride_w + 1u != desc->output_width) return LW_STATUS_INVALID_SHAPE;
#if LW_DW_X86
    uint32_t taps = desc->kernel_h * desc->kernel_w;
    uint32_t blocks = (desc->channels + LW_NHWC_DEPTHWISE_BLOCK - 1u) / LW_NHWC_DEPTHWISE_BLOCK;
    for (uint32_t batch = 0u; batch < desc->batch; ++batch) {
        const float* input_batch = input + (size_t)batch * desc->input_height * desc->input_width * desc->channels;
        float* output_batch = output + (size_t)batch * desc->output_height * desc->output_width * desc->channels;
        for (uint32_t output_y = 0u; output_y < desc->output_height; ++output_y) {
            for (uint32_t block = 0u; block < blocks; ++block) {
                uint32_t channel_base = block * LW_NHWC_DEPTHWISE_BLOCK;
                uint32_t remaining = desc->channels - channel_base;
                uint32_t current_channels = remaining > LW_NHWC_DEPTHWISE_BLOCK ? LW_NHWC_DEPTHWISE_BLOCK : remaining;
                uint32_t vector_count = current_channels / 8u;
                const float* block_weights = packed_weights +
                    (size_t)block * taps * LW_NHWC_DEPTHWISE_BLOCK;
                uint32_t output_x = 0u;
                while (output_x < desc->output_width) {
                    if (output_x + 1u < desc->output_width &&
                        vector_count == 4u &&
                        depthwise_pair_is_interior(desc, output_y, output_x)) {
                        depthwise_block32_x2(input_batch, block_weights, bias, output_batch, desc,
                                             output_y, output_x, channel_base, 0u);
                        if (stats != NULL && stats->x2_invocations != UINT64_MAX) ++stats->x2_invocations;
                        output_x += 2u;
                    } else {
                        depthwise_block32(input_batch, block_weights, bias, output_batch, desc,
                                          output_y, output_x, channel_base, vector_count, 0u);
                        if (stats != NULL && stats->x1_invocations != UINT64_MAX) ++stats->x1_invocations;
                        ++output_x;
                    }
                }
            }
        }
    }
    return LW_STATUS_OK;
#elif LW_DW_WASM128
    /* Preserve the OC32 pack and ascending tap order, but compute four
     * adjacent channels per SIMD128 lane group. This path handles borders,
     * row shards and channel tails with the same descriptor as the scalar
     * implementation. */
    uint32_t taps = desc->kernel_h * desc->kernel_w;
    uint32_t blocks = (desc->channels + LW_NHWC_DEPTHWISE_BLOCK - 1u) /
                      LW_NHWC_DEPTHWISE_BLOCK;
    for (uint32_t batch = 0u; batch < desc->batch; ++batch) {
        const float* input_batch = input + (size_t)batch * desc->input_height *
            desc->input_width * desc->channels;
        float* output_batch = output + (size_t)batch * desc->output_height *
            desc->output_width * desc->channels;
        for (uint32_t oy = 0u; oy < desc->output_height; ++oy) {
            int32_t iy0 = (int32_t)((uint64_t)(oy + desc->output_row_offset) *
                desc->stride_h) - (int32_t)desc->pad_top;
            for (uint32_t ox = 0u; ox < desc->output_width; ++ox) {
                int32_t ix0 = (int32_t)((uint64_t)ox * desc->stride_w) -
                    (int32_t)desc->pad_left;
                float* destination = output_batch +
                    ((size_t)oy * desc->output_width + ox) * desc->channels;
                for (uint32_t block = 0u; block < blocks; ++block) {
                    uint32_t channel_base = block * LW_NHWC_DEPTHWISE_BLOCK;
                    uint32_t lanes = desc->channels - channel_base;
                    const float* block_weights = packed_weights +
                        (size_t)block * taps * LW_NHWC_DEPTHWISE_BLOCK;
                    v128_t sums[LW_NHWC_DEPTHWISE_BLOCK / 4u];
                    if (lanes > LW_NHWC_DEPTHWISE_BLOCK) lanes = LW_NHWC_DEPTHWISE_BLOCK;
                    for (uint32_t lane = 0u; lane < lanes; lane += 4u) {
                        sums[lane / 4u] = bias == NULL ? wasm_f32x4_splat(0.0f) :
                            wasm_v128_load(bias + channel_base + lane);
                    }
                    for (uint32_t ky = 0u; ky < desc->kernel_h; ++ky) {
                        int32_t iy = iy0 + (int32_t)ky;
                        if (iy < 0 || iy >= (int32_t)desc->input_height) continue;
                        for (uint32_t kx = 0u; kx < desc->kernel_w; ++kx) {
                            int32_t ix = ix0 + (int32_t)kx;
                            if (ix < 0 || ix >= (int32_t)desc->input_width) continue;
                            const float* source = input_batch +
                                ((size_t)(uint32_t)iy * desc->input_width + (uint32_t)ix) *
                                desc->channels + channel_base;
                            const float* weight = block_weights +
                                ((size_t)ky * desc->kernel_w + kx) *
                                LW_NHWC_DEPTHWISE_BLOCK;
                            for (uint32_t lane = 0u; lane < lanes; lane += 4u) {
                                sums[lane / 4u] = wasm_f32x4_add(sums[lane / 4u],
                                    wasm_f32x4_mul(wasm_v128_load(source + lane),
                                        wasm_v128_load(weight + lane)));
                            }
                        }
                    }
                    for (uint32_t lane = 0u; lane < lanes; lane += 4u) {
                        v128_t sum = sums[lane / 4u];
                        if (desc->post_bias != NULL) {
                            sum = wasm_f32x4_add(sum,
                                wasm_v128_load(desc->post_bias + channel_base + lane));
                        }
                        wasm_v128_store(destination + channel_base + lane, sum);
                    }
                }
                if (stats != NULL && stats->x1_invocations != UINT64_MAX) {
                    ++stats->x1_invocations;
                }
            }
        }
    }
    return LW_STATUS_OK;
#else
    /* Physical NHWC fallback for non-x86 compiled backends. Preserve the
     * packed 32-channel layout and the per-channel tap accumulation order. */
    for (uint32_t batch = 0u; batch < desc->batch; ++batch) {
        const float* input_batch = input + (size_t)batch * desc->input_height *
            desc->input_width * desc->channels;
        float* output_batch = output + (size_t)batch * desc->output_height *
            desc->output_width * desc->channels;
        for (uint32_t oy = 0u; oy < desc->output_height; ++oy) {
            int32_t iy0 = (int32_t)((uint64_t)(oy + desc->output_row_offset) *
                desc->stride_h) - (int32_t)desc->pad_top;
            for (uint32_t ox = 0u; ox < desc->output_width; ++ox) {
                int32_t ix0 = (int32_t)((uint64_t)ox * desc->stride_w) -
                    (int32_t)desc->pad_left;
                for (uint32_t channel = 0u; channel < desc->channels; ++channel) {
                    uint32_t block = channel / LW_NHWC_DEPTHWISE_BLOCK;
                    uint32_t lane = channel % LW_NHWC_DEPTHWISE_BLOCK;
                    float sum = bias == NULL ? 0.0f : bias[channel];
                    for (uint32_t ky = 0u; ky < desc->kernel_h; ++ky) {
                        int32_t iy = iy0 + (int32_t)ky;
                        if (iy < 0 || iy >= (int32_t)desc->input_height) continue;
                        for (uint32_t kx = 0u; kx < desc->kernel_w; ++kx) {
                            int32_t ix = ix0 + (int32_t)kx;
                            if (ix < 0 || ix >= (int32_t)desc->input_width) continue;
                            size_t input_index = ((size_t)(uint32_t)iy * desc->input_width +
                                (uint32_t)ix) * desc->channels + channel;
                            size_t weight_index = ((size_t)block * desc->kernel_h *
                                desc->kernel_w + (size_t)ky * desc->kernel_w + kx) *
                                LW_NHWC_DEPTHWISE_BLOCK + lane;
                            sum += input_batch[input_index] * packed_weights[weight_index];
                        }
                    }
                    if (desc->post_bias != NULL) sum += desc->post_bias[channel];
                    output_batch[((size_t)oy * desc->output_width + ox) *
                        desc->channels + channel] = sum;
                }
                if (stats != NULL && stats->x1_invocations != UINT64_MAX) {
                    ++stats->x1_invocations;
                }
            }
        }
    }
    return LW_STATUS_OK;
#endif
}

lw_status lw_avx2_fma_nhwc_depthwise_f32(const float* input, const float* packed_weights,
                                         const float* bias, float* output,
                                         const lw_nhwc_depthwise_desc* desc) {
    return depthwise_execute(input, packed_weights, bias, output, desc, NULL);
}

lw_status lw_avx2_fma_nhwc_depthwise_profiled_f32(const float* input, const float* packed_weights,
                                                   const float* bias, float* output,
                                                   const lw_nhwc_depthwise_desc* desc,
                                                   lw_nhwc_depthwise_stats* stats) {
    return depthwise_execute(input, packed_weights, bias, output, desc, stats);
}
