#include "nhwc_internal.h"

#include <stddef.h>

/* Preserve the existing accumulation and exact GELU implementations. A tile
 * is completely expanded before projection; only its hidden rows survive.
 * Kernel selection is performed once by the caller, outside the tile loop. */
void lw_nhwc_ffn_tile_f32(const float* input, const float* expand_weights,
    const lw_nhwc_epilogue* expand_epilogue, const float* project_weights,
    const lw_nhwc_epilogue* project_epilogue, float* output, float* hidden,
    uint32_t pixels, uint32_t input_channels, uint32_t hidden_channels,
    uint32_t output_channels, uint32_t tile_pixels,
    lw_nhwc_pointwise_fn expand, lw_nhwc_pointwise_fn project) {
    uint32_t begin = 0u;
    while (begin < pixels) {
        uint32_t count = pixels - begin;
        lw_nhwc_epilogue ep = *project_epilogue;
        if (count > tile_pixels) count = tile_pixels;
        if (ep.residual != NULL) ep.residual += (size_t)begin * output_channels;
        expand(input + (size_t)begin * input_channels, expand_weights,
               expand_epilogue, hidden, count, input_channels, hidden_channels);
        project(hidden, project_weights, &ep,
                output + (size_t)begin * output_channels,
                count, hidden_channels, output_channels);
        begin += count;
    }
}

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define LW_NHWC_X86 1
#include <immintrin.h>
#endif

static float lw_nhwc_apply_activation(float value, uint16_t activation,
                                      float alpha, float beta) {
    (void)alpha;
    (void)beta;
    if (activation == LW_NHWC_ACT_RELU) {
        return value > 0.0f ? value : 0.0f;
    }
    if (activation == LW_NHWC_ACT_HARDSWISH) {
        float gate = value + 3.0f;
        if (gate < 0.0f) {
            gate = 0.0f;
        } else if (gate > 6.0f) {
            gate = 6.0f;
        }
        return value * gate / 6.0f;
    }
    if (activation == LW_NHWC_ACT_GELU) {
        return lw_nhwc_gelu_scalar_exact_f32(value);
    }
    return value;
}

static void lw_nhwc_pointwise_scalar(const float* input,
                                     const float* packed_weights,
                                     const lw_nhwc_epilogue* epilogue,
                                     float* output,
                                     uint32_t pixels,
                                     uint32_t input_channels,
                                     uint32_t output_channels) {
    const float* bias = epilogue == NULL ? NULL : epilogue->bias;
    const float* residual = epilogue == NULL ? NULL : epilogue->residual;
    const float* post_bias = epilogue == NULL ? NULL : epilogue->post_bias;
    uint16_t activation = epilogue == NULL ? LW_NHWC_ACT_NONE : epilogue->activation;
    float alpha = epilogue == NULL ? 0.0f : epilogue->alpha;
    float beta = epilogue == NULL ? 0.0f : epilogue->beta;
    uint32_t pixel;

    for (pixel = 0u; pixel < pixels; ++pixel) {
        uint32_t output_channel;
        for (output_channel = 0u; output_channel < output_channels;
             ++output_channel) {
            uint32_t block = output_channel / LW_NHWC_OC_BLOCK;
            uint32_t lane = output_channel % LW_NHWC_OC_BLOCK;
            const float* packed = packed_weights +
                                  (size_t)block * input_channels * LW_NHWC_OC_BLOCK;
            float sum = bias == NULL ? 0.0f : bias[output_channel];
            uint32_t input_channel;
            for (input_channel = 0u; input_channel < input_channels;
                 ++input_channel) {
                sum += input[(size_t)pixel * input_channels + input_channel] *
                       packed[(size_t)input_channel * LW_NHWC_OC_BLOCK + lane];
            }
            if (post_bias != NULL) {
                sum += post_bias[output_channel];
            }
            if (residual != NULL) {
                sum += residual[(size_t)pixel * output_channels + output_channel];
            }
            output[(size_t)pixel * output_channels + output_channel] =
                lw_nhwc_apply_activation(sum, activation, alpha, beta);
        }
    }
}

#if defined(LW_NHWC_X86)
#if defined(__GNUC__) || defined(__clang__)
#define LW_NHWC_AVX2_FMA __attribute__((target("avx2,fma")))
#else
#define LW_NHWC_AVX2_FMA
#endif

LW_NHWC_AVX2_FMA
static void lw_nhwc_store_row16(__m256 lo, __m256 hi,
                                float* output,
                                const float* residual,
                                const float* post_bias,
                                uint16_t activation,
                                uint32_t channels) {
    if (channels < 16u) {
        float values[16];
        uint32_t i;
        _mm256_storeu_ps(values, lo);
        _mm256_storeu_ps(values + 8u, hi);
        for (i = 0u; i < channels; ++i) {
            float value = values[i];
            if (post_bias != NULL) value += post_bias[i];
            if (residual != NULL) value += residual[i];
            values[i] = value;
        }
        if (activation == LW_NHWC_ACT_GELU) {
            /* Match the standalone pass layout exactly: full 8-lane groups go
             * through the vector polynomial, the sub-8 tail through the
             * scalar erff helper. channels < 16 here, so at most one full
             * vector group exists. */
            uint32_t vector_end = channels & ~7u;
            if (vector_end >= 8u) {
                _mm256_storeu_ps(values,
                    lw_nhwc_avx2_gelu_vector_exact_f32(_mm256_loadu_ps(values)));
            }
            for (i = vector_end; i < channels; ++i) {
                values[i] = lw_nhwc_gelu_scalar_exact_f32(values[i]);
            }
            for (i = 0u; i < channels; ++i) output[i] = values[i];
            return;
        }
        for (i = 0u; i < channels; ++i) {
            float value = values[i];
            if (activation == LW_NHWC_ACT_RELU) {
                value = value < 0.0f ? 0.0f : value;
            } else if (activation == LW_NHWC_ACT_HARDSWISH) {
                float gate = value + 3.0f;
                if (gate < 0.0f) gate = 0.0f;
                if (gate > 6.0f) gate = 6.0f;
                value = value * gate * (1.0f / 6.0f);
            }
            output[i] = value;
        }
        return;
    }
    if (post_bias != NULL) {
        lo = _mm256_add_ps(lo, _mm256_loadu_ps(post_bias));
        hi = _mm256_add_ps(hi, _mm256_loadu_ps(post_bias + 8u));
    }
    if (residual != NULL) {
        lo = _mm256_add_ps(lo, _mm256_loadu_ps(residual));
        hi = _mm256_add_ps(hi, _mm256_loadu_ps(residual + 8u));
    }
    if (activation == LW_NHWC_ACT_RELU) {
        __m256 zero = _mm256_setzero_ps();
        lo = _mm256_max_ps(lo, zero);
        hi = _mm256_max_ps(hi, zero);
    } else if (activation == LW_NHWC_ACT_HARDSWISH) {
        __m256 three = _mm256_set1_ps(3.0f);
        __m256 six = _mm256_set1_ps(6.0f);
        __m256 inv_six = _mm256_set1_ps(1.0f / 6.0f);
        __m256 zero = _mm256_setzero_ps();
        __m256 gate_lo = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(lo, three), zero), six);
        __m256 gate_hi = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(hi, three), zero), six);
        lo = _mm256_mul_ps(_mm256_mul_ps(lo, gate_lo), inv_six);
        hi = _mm256_mul_ps(_mm256_mul_ps(hi, gate_hi), inv_six);
    } else if (activation == LW_NHWC_ACT_GELU) {
        lo = lw_nhwc_avx2_gelu_vector_exact_f32(lo);
        hi = lw_nhwc_avx2_gelu_vector_exact_f32(hi);
    }
    _mm256_storeu_ps(output, lo);
    _mm256_storeu_ps(output + 8u, hi);
}

LW_NHWC_AVX2_FMA
static void lw_nhwc_tile_rows16(const float* input,
                                const float* packed_weights,
                                const float* bias,
                                const float* residual,
                                const float* post_bias,
                                float* output,
                                uint32_t rows,
                                uint32_t input_channels,
                                uint32_t output_stride,
                                uint16_t activation,
                                uint32_t output_channels) {
    __m256 lo[6];
    __m256 hi[6];
    uint32_t row;
    uint32_t input_channel;
    __m256 zero = _mm256_setzero_ps();

    for (row = 0u; row < 6u; ++row) {
        if (bias != NULL && output_channels < LW_NHWC_OC_BLOCK) {
            float bias_values[LW_NHWC_OC_BLOCK] = {0.0f};
            for (uint32_t lane = 0u; lane < output_channels; ++lane) {
                bias_values[lane] = bias[lane];
            }
            lo[row] = _mm256_loadu_ps(bias_values);
            hi[row] = _mm256_loadu_ps(bias_values + 8u);
        } else if (bias != NULL) {
            lo[row] = _mm256_loadu_ps(bias);
            hi[row] = _mm256_loadu_ps(bias + 8u);
        } else {
            lo[row] = zero;
            hi[row] = zero;
        }
    }
    for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
        const float* weights = packed_weights + (size_t)input_channel * 16u;
        __m256 w0 = _mm256_loadu_ps(weights);
        __m256 w1 = _mm256_loadu_ps(weights + 8u);
        for (row = 0u; row < 6u; ++row) {
            const float* row_input = input + (size_t)(row < rows ? row : 0u) * input_channels;
            __m256 value = _mm256_set1_ps(row_input[input_channel]);
            lo[row] = _mm256_fmadd_ps(value, w0, lo[row]);
            hi[row] = _mm256_fmadd_ps(value, w1, hi[row]);
        }
    }
    for (row = 0u; row < rows; ++row) {
        float* row_output = output + (size_t)row * output_stride;
        const float* row_residual = residual == NULL ? NULL :
            residual + (size_t)row * output_stride;
        lw_nhwc_store_row16(lo[row], hi[row], row_output, row_residual,
                            post_bias, activation, output_channels);
    }
}

/* Grouped-path worker: one noinline call covers a whole pixel range of a
 * single full 16-channel output block, paying the call and stack-frame
 * prologue once per (pixel group, channel block) instead of once per 6-row
 * tile. Bit-identical to repeated lw_nhwc_tile_rows16 calls: same bias
 * initialisation, same ascending input-channel FMA order, same store_row16
 * epilogue. Tail tiles (rows < 6) delegate to lw_nhwc_tile_rows16. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#elif defined(_MSC_VER)
__declspec(noinline)
#endif
LW_NHWC_AVX2_FMA
static void lw_nhwc_ocblock16_tile_range(const float* input,
                                         const float* packed_weights,
                                         const float* bias,
                                         const float* residual,
                                         const float* post_bias,
                                         float* output,
                                         uint32_t pixel_begin,
                                         uint32_t pixel_end,
                                         uint32_t input_channels,
                                         uint32_t output_stride,
                                         uint16_t activation) {
    uint32_t pixel;
    for (pixel = pixel_begin; pixel < pixel_end; pixel += LW_NHWC_PIXEL_TILE) {
        uint32_t rows = pixel_end - pixel;
        const float* tile_input = input + (size_t)pixel * input_channels;
        float* tile_output = output + (size_t)pixel * output_stride;
        const float* tile_residual = residual == NULL ? NULL :
            residual + (size_t)pixel * output_stride;
        __m256 lo0, lo1, lo2, lo3, lo4, lo5;
        __m256 hi0, hi1, hi2, hi3, hi4, hi5;
        uint32_t input_channel;
        if (rows > LW_NHWC_PIXEL_TILE) {
            rows = LW_NHWC_PIXEL_TILE;
        }
        if (rows < LW_NHWC_PIXEL_TILE) {
            lw_nhwc_tile_rows16(tile_input, packed_weights, bias, tile_residual,
                                post_bias, tile_output, rows, input_channels,
                                output_stride, activation, LW_NHWC_OC_BLOCK);
            continue;
        }
        if (bias != NULL) {
            __m256 bias_lo = _mm256_loadu_ps(bias);
            __m256 bias_hi = _mm256_loadu_ps(bias + 8u);
            lo0 = bias_lo; lo1 = bias_lo; lo2 = bias_lo;
            lo3 = bias_lo; lo4 = bias_lo; lo5 = bias_lo;
            hi0 = bias_hi; hi1 = bias_hi; hi2 = bias_hi;
            hi3 = bias_hi; hi4 = bias_hi; hi5 = bias_hi;
        } else {
            __m256 zero = _mm256_setzero_ps();
            lo0 = zero; lo1 = zero; lo2 = zero;
            lo3 = zero; lo4 = zero; lo5 = zero;
            hi0 = zero; hi1 = zero; hi2 = zero;
            hi3 = zero; hi4 = zero; hi5 = zero;
        }
        for (input_channel = 0u; input_channel < input_channels;
             ++input_channel) {
            const float* weights = packed_weights + (size_t)input_channel * 16u;
            __m256 w0 = _mm256_loadu_ps(weights);
            __m256 w1 = _mm256_loadu_ps(weights + 8u);
            __m256 value;
            value = _mm256_set1_ps(tile_input[input_channel]);
            lo0 = _mm256_fmadd_ps(value, w0, lo0);
            hi0 = _mm256_fmadd_ps(value, w1, hi0);
            value = _mm256_set1_ps(
                tile_input[(size_t)input_channels + input_channel]);
            lo1 = _mm256_fmadd_ps(value, w0, lo1);
            hi1 = _mm256_fmadd_ps(value, w1, hi1);
            value = _mm256_set1_ps(
                tile_input[(size_t)2u * input_channels + input_channel]);
            lo2 = _mm256_fmadd_ps(value, w0, lo2);
            hi2 = _mm256_fmadd_ps(value, w1, hi2);
            value = _mm256_set1_ps(
                tile_input[(size_t)3u * input_channels + input_channel]);
            lo3 = _mm256_fmadd_ps(value, w0, lo3);
            hi3 = _mm256_fmadd_ps(value, w1, hi3);
            value = _mm256_set1_ps(
                tile_input[(size_t)4u * input_channels + input_channel]);
            lo4 = _mm256_fmadd_ps(value, w0, lo4);
            hi4 = _mm256_fmadd_ps(value, w1, hi4);
            value = _mm256_set1_ps(
                tile_input[(size_t)5u * input_channels + input_channel]);
            lo5 = _mm256_fmadd_ps(value, w0, lo5);
            hi5 = _mm256_fmadd_ps(value, w1, hi5);
        }
        lw_nhwc_store_row16(lo0, hi0, tile_output, tile_residual, post_bias,
                            activation, LW_NHWC_OC_BLOCK);
        lw_nhwc_store_row16(lo1, hi1, tile_output + output_stride,
                            tile_residual == NULL ? NULL :
                                tile_residual + output_stride,
                            post_bias, activation, LW_NHWC_OC_BLOCK);
        lw_nhwc_store_row16(lo2, hi2, tile_output + (size_t)2u * output_stride,
                            tile_residual == NULL ? NULL :
                                tile_residual + (size_t)2u * output_stride,
                            post_bias, activation, LW_NHWC_OC_BLOCK);
        lw_nhwc_store_row16(lo3, hi3, tile_output + (size_t)3u * output_stride,
                            tile_residual == NULL ? NULL :
                                tile_residual + (size_t)3u * output_stride,
                            post_bias, activation, LW_NHWC_OC_BLOCK);
        lw_nhwc_store_row16(lo4, hi4, tile_output + (size_t)4u * output_stride,
                            tile_residual == NULL ? NULL :
                                tile_residual + (size_t)4u * output_stride,
                            post_bias, activation, LW_NHWC_OC_BLOCK);
        lw_nhwc_store_row16(lo5, hi5, tile_output + (size_t)5u * output_stride,
                            tile_residual == NULL ? NULL :
                                tile_residual + (size_t)5u * output_stride,
                            post_bias, activation, LW_NHWC_OC_BLOCK);
    }
}

/* Whole-tensor fast path: one call per conv, tile-outer/oc-inner loops with
 * the unrolled 6x16 accumulate inline. Per-tile function-call overhead of the
 * grouped path disappears. Bit-identical to lw_nhwc_tile_rows16: same bias
 * initialisation, same ascending input-channel FMA order, same store_row16
 * epilogue. Tail tiles (rows < 6) delegate to lw_nhwc_tile_rows16. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#elif defined(_MSC_VER)
__declspec(noinline)
#endif
LW_NHWC_AVX2_FMA
static void lw_nhwc_pointwise_unrolled_all(const float* input,
                                           const float* packed_weights,
                                           const lw_nhwc_epilogue* epilogue,
                                           float* output, uint32_t pixels,
                                           uint32_t input_channels,
                                           uint32_t output_channels) {
    const float* bias = epilogue->bias;
    const float* residual = epilogue->residual;
    const float* post_bias = epilogue->post_bias;
    uint16_t activation = epilogue->activation;
    uint32_t tile;
    for (tile = 0u; tile < pixels; tile += LW_NHWC_PIXEL_TILE) {
        uint32_t rows = pixels - tile;
        uint32_t output_channel;
        if (rows > LW_NHWC_PIXEL_TILE) rows = LW_NHWC_PIXEL_TILE;
        for (output_channel = 0u; output_channel < output_channels;
             output_channel += LW_NHWC_OC_BLOCK) {
            const float* packed = packed_weights +
                (size_t)(output_channel / LW_NHWC_OC_BLOCK) * input_channels *
                LW_NHWC_OC_BLOCK;
            const float* tile_input = input + (size_t)tile * input_channels;
            float* tile_output = output + (size_t)tile * output_channels +
                                 output_channel;
            const float* tile_residual = residual == NULL ? NULL :
                residual + (size_t)tile * output_channels + output_channel;
            const float* block_bias = bias == NULL ? NULL : bias + output_channel;
            const float* block_post_bias =
                post_bias == NULL ? NULL : post_bias + output_channel;
            if (rows < LW_NHWC_PIXEL_TILE) {
                lw_nhwc_tile_rows16(tile_input, packed, block_bias, tile_residual,
                                    block_post_bias, tile_output, rows,
                                    input_channels, output_channels, activation,
                                    LW_NHWC_OC_BLOCK);
                continue;
            }
            {
                __m256 lo0, lo1, lo2, lo3, lo4, lo5;
                __m256 hi0, hi1, hi2, hi3, hi4, hi5;
                uint32_t input_channel;
                if (block_bias != NULL) {
                    __m256 bias_lo = _mm256_loadu_ps(block_bias);
                    __m256 bias_hi = _mm256_loadu_ps(block_bias + 8u);
                    lo0 = bias_lo; lo1 = bias_lo; lo2 = bias_lo;
                    lo3 = bias_lo; lo4 = bias_lo; lo5 = bias_lo;
                    hi0 = bias_hi; hi1 = bias_hi; hi2 = bias_hi;
                    hi3 = bias_hi; hi4 = bias_hi; hi5 = bias_hi;
                } else {
                    __m256 zero = _mm256_setzero_ps();
                    lo0 = zero; lo1 = zero; lo2 = zero;
                    lo3 = zero; lo4 = zero; lo5 = zero;
                    hi0 = zero; hi1 = zero; hi2 = zero;
                    hi3 = zero; hi4 = zero; hi5 = zero;
                }
                for (input_channel = 0u; input_channel < input_channels;
                     ++input_channel) {
                    const float* weights =
                        packed + (size_t)input_channel * 16u;
                    __m256 w0 = _mm256_loadu_ps(weights);
                    __m256 w1 = _mm256_loadu_ps(weights + 8u);
                    __m256 value;
                    value = _mm256_set1_ps(tile_input[input_channel]);
                    lo0 = _mm256_fmadd_ps(value, w0, lo0);
                    hi0 = _mm256_fmadd_ps(value, w1, hi0);
                    value = _mm256_set1_ps(
                        tile_input[(size_t)input_channels + input_channel]);
                    lo1 = _mm256_fmadd_ps(value, w0, lo1);
                    hi1 = _mm256_fmadd_ps(value, w1, hi1);
                    value = _mm256_set1_ps(
                        tile_input[(size_t)2u * input_channels + input_channel]);
                    lo2 = _mm256_fmadd_ps(value, w0, lo2);
                    hi2 = _mm256_fmadd_ps(value, w1, hi2);
                    value = _mm256_set1_ps(
                        tile_input[(size_t)3u * input_channels + input_channel]);
                    lo3 = _mm256_fmadd_ps(value, w0, lo3);
                    hi3 = _mm256_fmadd_ps(value, w1, hi3);
                    value = _mm256_set1_ps(
                        tile_input[(size_t)4u * input_channels + input_channel]);
                    lo4 = _mm256_fmadd_ps(value, w0, lo4);
                    hi4 = _mm256_fmadd_ps(value, w1, hi4);
                    value = _mm256_set1_ps(
                        tile_input[(size_t)5u * input_channels + input_channel]);
                    lo5 = _mm256_fmadd_ps(value, w0, lo5);
                    hi5 = _mm256_fmadd_ps(value, w1, hi5);
                }
                lw_nhwc_store_row16(lo0, hi0, tile_output, tile_residual,
                                    block_post_bias, activation, LW_NHWC_OC_BLOCK);
                lw_nhwc_store_row16(
                    lo1, hi1, tile_output + output_channels,
                    tile_residual == NULL ? NULL : tile_residual + output_channels,
                    block_post_bias, activation, LW_NHWC_OC_BLOCK);
                lw_nhwc_store_row16(
                    lo2, hi2, tile_output + (size_t)2u * output_channels,
                    tile_residual == NULL ? NULL :
                        tile_residual + (size_t)2u * output_channels,
                    block_post_bias, activation, LW_NHWC_OC_BLOCK);
                lw_nhwc_store_row16(
                    lo3, hi3, tile_output + (size_t)3u * output_channels,
                    tile_residual == NULL ? NULL :
                        tile_residual + (size_t)3u * output_channels,
                    block_post_bias, activation, LW_NHWC_OC_BLOCK);
                lw_nhwc_store_row16(
                    lo4, hi4, tile_output + (size_t)4u * output_channels,
                    tile_residual == NULL ? NULL :
                        tile_residual + (size_t)4u * output_channels,
                    block_post_bias, activation, LW_NHWC_OC_BLOCK);
                lw_nhwc_store_row16(
                    lo5, hi5, tile_output + (size_t)5u * output_channels,
                    tile_residual == NULL ? NULL :
                        tile_residual + (size_t)5u * output_channels,
                    block_post_bias, activation, LW_NHWC_OC_BLOCK);
            }
        }
    }
}
#endif

void lw_avx2_fma_nhwc_pointwise_grouped_f32(const float* input,
                                             const float* packed_weights,
                                             const lw_nhwc_epilogue* epilogue,
                                             float* output,
                                             uint32_t pixels,
                                             uint32_t input_channels,
                                             uint32_t output_channels,
                                             uint32_t group_tiles) {
#if defined(LW_NHWC_X86)
    const float* bias = epilogue == NULL ? NULL : epilogue->bias;
    const float* residual = epilogue == NULL ? NULL : epilogue->residual;
    const float* post_bias = epilogue == NULL ? NULL : epilogue->post_bias;
#endif
    uint16_t activation = epilogue == NULL ? LW_NHWC_ACT_NONE : epilogue->activation;

    if (input == NULL || packed_weights == NULL || output == NULL ||
        input_channels == 0u || output_channels == 0u || pixels == 0u) {
        return;
    }
    if (activation != LW_NHWC_ACT_NONE && activation != LW_NHWC_ACT_RELU &&
        activation != LW_NHWC_ACT_HARDSWISH &&
        activation != LW_NHWC_ACT_GELU) {
        lw_nhwc_pointwise_scalar(input, packed_weights, epilogue, output, pixels,
                                 input_channels, output_channels);
        return;
    }

#if defined(LW_NHWC_X86)
    {
        uint32_t tile_count = (pixels + LW_NHWC_PIXEL_TILE - 1u) /
                              LW_NHWC_PIXEL_TILE;
        uint32_t tile_group = group_tiles == 0u ? tile_count : group_tiles;
        uint32_t group_begin;
        if (tile_group == 0u) {
            tile_group = 1u;
        }
        for (group_begin = 0u; group_begin < tile_count;
             group_begin += tile_group) {
            uint32_t group_end = group_begin + tile_group;
            uint32_t output_channel;
            if (group_end > tile_count) {
                group_end = tile_count;
            }
            for (output_channel = 0u; output_channel < output_channels;
                 output_channel += LW_NHWC_OC_BLOCK) {
                const float* packed = packed_weights +
                    (size_t)(output_channel / LW_NHWC_OC_BLOCK) * input_channels *
                    LW_NHWC_OC_BLOCK;
                uint32_t block_channels = output_channels - output_channel;
                if (block_channels > LW_NHWC_OC_BLOCK) {
                    block_channels = LW_NHWC_OC_BLOCK;
                }
                if (block_channels == LW_NHWC_OC_BLOCK) {
                    /* Full channel block: one worker call covers the whole
                     * pixel group, paying the stack-frame prologue once. */
                    uint32_t pixel_begin = group_begin * LW_NHWC_PIXEL_TILE;
                    uint32_t pixel_end = group_end * LW_NHWC_PIXEL_TILE;
                    if (pixel_end > pixels) {
                        pixel_end = pixels;
                    }
                    lw_nhwc_ocblock16_tile_range(
                        input, packed,
                        bias == NULL ? NULL : bias + output_channel,
                        residual == NULL ? NULL : residual + output_channel,
                        post_bias == NULL ? NULL : post_bias + output_channel,
                        output + output_channel,
                        pixel_begin, pixel_end,
                        input_channels, output_channels, activation);
                    continue;
                }
                for (uint32_t tile = group_begin; tile < group_end; ++tile) {
                    uint32_t pixel = tile * LW_NHWC_PIXEL_TILE;
                    uint32_t rows = pixels - pixel;
                    const float* tile_input = input + (size_t)pixel * input_channels;
                    float* tile_output = output + (size_t)pixel * output_channels +
                                         output_channel;
                    const float* tile_residual = residual == NULL ? NULL :
                        residual + (size_t)pixel * output_channels + output_channel;
                    if (rows > LW_NHWC_PIXEL_TILE) {
                        rows = LW_NHWC_PIXEL_TILE;
                    }
                    lw_nhwc_tile_rows16(tile_input, packed,
                                        bias == NULL ? NULL : bias + output_channel,
                                        tile_residual,
                                        post_bias == NULL ? NULL :
                                            post_bias + output_channel,
                                        tile_output, rows,
                                        input_channels, output_channels, activation,
                                        block_channels);
                }
            }
        }
    }
#else
    (void)group_tiles;
    lw_nhwc_pointwise_scalar(input, packed_weights, epilogue, output, pixels,
                             input_channels, output_channels);
#endif
}


#if defined(LW_NHWC_X86)
LW_NHWC_AVX2_FMA
static void lw_nhwc_tile_rows16_4(const float* input, const float* packed_weights,
                                   const float* bias, const float* residual,
                                   const float* post_bias, float* output,
                                   uint32_t rows, uint32_t input_channels,
                                   uint32_t output_stride, uint16_t activation) {
    __m256 lo[4], hi[4];
    __m256 zero = _mm256_setzero_ps();
    uint32_t row;
    for (row = 0u; row < 4u; ++row) {
        if (bias != NULL) {
            lo[row] = _mm256_loadu_ps(bias);
            hi[row] = _mm256_loadu_ps(bias + 8u);
        } else {
            lo[row] = zero;
            hi[row] = zero;
        }
    }
    for (uint32_t input_channel = 0u; input_channel < input_channels; ++input_channel) {
        const float* weights = packed_weights + (size_t)input_channel * 16u;
        __m256 w0 = _mm256_loadu_ps(weights);
        __m256 w1 = _mm256_loadu_ps(weights + 8u);
        for (row = 0u; row < rows; ++row) {
            __m256 value = _mm256_set1_ps(input[(size_t)row * input_channels + input_channel]);
            lo[row] = _mm256_fmadd_ps(value, w0, lo[row]);
            hi[row] = _mm256_fmadd_ps(value, w1, hi[row]);
        }
    }
    for (row = 0u; row < rows; ++row) {
        float* row_output = output + (size_t)row * output_stride;
        const float* row_residual = residual == NULL ? NULL : residual + (size_t)row * output_stride;
        lw_nhwc_store_row16(lo[row], hi[row], row_output, row_residual,
                            post_bias, activation, LW_NHWC_OC_BLOCK);
    }
}

LW_NHWC_AVX2_FMA
static __m256 lw_nhwc_apply_vec(__m256 value, uint16_t activation) {
    if (activation == LW_NHWC_ACT_RELU) {
        return _mm256_max_ps(value, _mm256_setzero_ps());
    }
    if (activation == LW_NHWC_ACT_HARDSWISH) {
        __m256 gate = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(value, _mm256_set1_ps(3.0f)),
                                                  _mm256_setzero_ps()),
                                    _mm256_set1_ps(6.0f));
        return _mm256_mul_ps(_mm256_mul_ps(value, gate), _mm256_set1_ps(1.0f / 6.0f));
    }
    if (activation == LW_NHWC_ACT_GELU) {
        return lw_nhwc_avx2_gelu_vector_exact_f32(value);
    }
    return value;
}

LW_NHWC_AVX2_FMA
static void lw_nhwc_store_row32(__m256* accumulators, float* output,
                                const float* residual, const float* post_bias,
                                uint16_t activation) {
    for (uint32_t vector = 0u; vector < 4u; ++vector) {
        __m256 value = accumulators[vector];
        if (post_bias != NULL) value = _mm256_add_ps(value, _mm256_loadu_ps(post_bias + vector * 8u));
        if (residual != NULL) value = _mm256_add_ps(value, _mm256_loadu_ps(residual + vector * 8u));
        value = lw_nhwc_apply_vec(value, activation);
        _mm256_storeu_ps(output + vector * 8u, value);
    }
}

LW_NHWC_AVX2_FMA
static void lw_nhwc_tile_rows32(const float* input, const float* packed_weights,
                                const float* bias, const float* residual,
                                const float* post_bias, float* output,
                                uint32_t rows, uint32_t input_channels,
                                uint32_t output_stride, uint16_t activation,
                                uint32_t row_tile) {
    __m256 accumulators[3][4];
    __m256 zero = _mm256_setzero_ps();
    uint32_t row;
    for (row = 0u; row < row_tile; ++row) {
        for (uint32_t vector = 0u; vector < 4u; ++vector) {
            uint32_t lane_offset = vector * 8u;
            accumulators[row][vector] = bias == NULL ? zero : _mm256_loadu_ps(bias + lane_offset);
        }
    }
    for (uint32_t input_channel = 0u; input_channel < input_channels; ++input_channel) {
        const float* weights0 = packed_weights + (size_t)input_channel * 16u;
        const float* weights1 = weights0 + (size_t)input_channels * 16u;
        __m256 weights[4] = {
            _mm256_loadu_ps(weights0), _mm256_loadu_ps(weights0 + 8u),
            _mm256_loadu_ps(weights1), _mm256_loadu_ps(weights1 + 8u)
        };
        for (row = 0u; row < rows; ++row) {
            __m256 value = _mm256_set1_ps(input[(size_t)row * input_channels + input_channel]);
            for (uint32_t vector = 0u; vector < 4u; ++vector) {
                accumulators[row][vector] =
                    _mm256_fmadd_ps(value, weights[vector], accumulators[row][vector]);
            }
        }
    }
    for (row = 0u; row < rows; ++row) {
        float* row_output = output + (size_t)row * output_stride;
        const float* row_residual = residual == NULL ? NULL : residual + (size_t)row * output_stride;
        lw_nhwc_store_row32(accumulators[row], row_output, row_residual,
                            post_bias, activation);
    }
}
#endif

void lw_avx2_fma_nhwc_pointwise_4x16_f32(const float* input,
                                           const float* packed_weights,
                                           const lw_nhwc_epilogue* epilogue,
                                           float* output,
                                           uint32_t pixels,
                                           uint32_t input_channels,
                                           uint32_t output_channels) {
    if (output_channels % LW_NHWC_OC_BLOCK != 0u ||
        epilogue == NULL ||
        (epilogue->activation != LW_NHWC_ACT_NONE &&
         epilogue->activation != LW_NHWC_ACT_RELU &&
         epilogue->activation != LW_NHWC_ACT_HARDSWISH &&
         epilogue->activation != LW_NHWC_ACT_GELU)) {
        lw_avx2_fma_nhwc_pointwise_f32(input, packed_weights, epilogue, output,
                                       pixels, input_channels, output_channels);
        return;
    }
#if defined(LW_NHWC_X86)
    for (uint32_t tile = 0u; tile < pixels; tile += 4u) {
        uint32_t rows = pixels - tile;
        if (rows > 4u) rows = 4u;
        for (uint32_t output_channel = 0u; output_channel < output_channels;
             output_channel += LW_NHWC_OC_BLOCK) {
            const float* packed = packed_weights +
                (size_t)(output_channel / LW_NHWC_OC_BLOCK) * input_channels * LW_NHWC_OC_BLOCK;
            lw_nhwc_tile_rows16_4(input + (size_t)tile * input_channels, packed,
                                   epilogue->bias == NULL ? NULL : epilogue->bias + output_channel,
                                   epilogue->residual == NULL ? NULL :
                                       epilogue->residual + (size_t)tile * output_channels + output_channel,
                                   epilogue->post_bias == NULL ? NULL :
                                       epilogue->post_bias + output_channel,
                                   output + (size_t)tile * output_channels + output_channel,
                                   rows, input_channels, output_channels, epilogue->activation);
        }
    }
#else
    lw_avx2_fma_nhwc_pointwise_f32(input, packed_weights, epilogue, output,
                                   pixels, input_channels, output_channels);
#endif
}

static void lw_avx2_fma_nhwc_pointwise_32x_f32(const float* input,
                                                const float* packed_weights,
                                                const lw_nhwc_epilogue* epilogue,
                                                float* output, uint32_t pixels,
                                                uint32_t input_channels,
                                                uint32_t output_channels,
                                                uint32_t row_tile) {
    if (output_channels % 32u != 0u ||
        epilogue == NULL ||
        (epilogue->activation != LW_NHWC_ACT_NONE &&
         epilogue->activation != LW_NHWC_ACT_RELU &&
         epilogue->activation != LW_NHWC_ACT_HARDSWISH &&
         epilogue->activation != LW_NHWC_ACT_GELU)) {
        lw_avx2_fma_nhwc_pointwise_f32(input, packed_weights, epilogue, output,
                                       pixels, input_channels, output_channels);
        return;
    }
#if defined(LW_NHWC_X86)
    for (uint32_t tile = 0u; tile < pixels; tile += row_tile) {
        uint32_t rows = pixels - tile;
        if (rows > row_tile) rows = row_tile;
        for (uint32_t output_channel = 0u; output_channel < output_channels;
             output_channel += 32u) {
            const float* packed = packed_weights +
                (size_t)(output_channel / LW_NHWC_OC_BLOCK) * input_channels * LW_NHWC_OC_BLOCK;
            lw_nhwc_tile_rows32(input + (size_t)tile * input_channels, packed,
                                epilogue->bias == NULL ? NULL : epilogue->bias + output_channel,
                                epilogue->residual == NULL ? NULL :
                                    epilogue->residual + (size_t)tile * output_channels + output_channel,
                                epilogue->post_bias == NULL ? NULL :
                                    epilogue->post_bias + output_channel,
                                output + (size_t)tile * output_channels + output_channel,
                                rows, input_channels, output_channels, epilogue->activation, row_tile);
        }
    }
#else
    (void)row_tile;
    lw_avx2_fma_nhwc_pointwise_f32(input, packed_weights, epilogue, output,
                                   pixels, input_channels, output_channels);
#endif
}

void lw_avx2_fma_nhwc_pointwise_3x32_f32(const float* input,
                                           const float* packed_weights,
                                           const lw_nhwc_epilogue* epilogue,
                                           float* output,
                                           uint32_t pixels,
                                           uint32_t input_channels,
                                           uint32_t output_channels) {
    lw_avx2_fma_nhwc_pointwise_32x_f32(input, packed_weights, epilogue, output,
                                       pixels, input_channels, output_channels, 3u);
}

void lw_avx2_fma_nhwc_pointwise_2x32_f32(const float* input,
                                           const float* packed_weights,
                                           const lw_nhwc_epilogue* epilogue,
                                           float* output,
                                           uint32_t pixels,
                                           uint32_t input_channels,
                                           uint32_t output_channels) {
    lw_avx2_fma_nhwc_pointwise_32x_f32(input, packed_weights, epilogue, output,
                                       pixels, input_channels, output_channels, 2u);
}

void lw_avx2_fma_nhwc_pointwise_f32(const float* input,
                                    const float* packed_weights,
                                    const lw_nhwc_epilogue* epilogue,
                                    float* output,
                                    uint32_t pixels,
                                    uint32_t input_channels,
                                    uint32_t output_channels) {
#if defined(LW_NHWC_X86)
    /* Whole-tensor fast path: tile loop + unrolled 6x16 accumulate live in one
     * function, so the per-tile call/prologue overhead of the grouped path
     * disappears. Only for weights that fit L2 comfortably (tile-outer order
     * re-streams the weights once per 6-row tile) and full 16-channel blocks.
     * Everything else keeps the grouped order, which is weight-cache friendly
     * for large channel counts. */
    if (output_channels % LW_NHWC_OC_BLOCK == 0u &&
        (uint64_t)input_channels * output_channels * sizeof(float) <=
            UINT64_C(262144) &&
        epilogue != NULL &&
        (epilogue->activation == LW_NHWC_ACT_NONE ||
         epilogue->activation == LW_NHWC_ACT_RELU ||
         epilogue->activation == LW_NHWC_ACT_HARDSWISH ||
         epilogue->activation == LW_NHWC_ACT_GELU)) {
        lw_nhwc_pointwise_unrolled_all(input, packed_weights, epilogue, output,
                                       pixels, input_channels, output_channels);
        return;
    }
#endif
    {
        /* Adaptive pixel grouping: make each pixel group as large as possible
         * while its input slice stays L2-resident (~448 KB budget, leaving
         * room for the output slice and weight block), so the per-oc-block
         * input re-reads hit L2 and the packed weights stream from DRAM as
         * few times as possible.  Small inputs collapse to a single group and
         * the weights stream exactly once; huge detection feature maps still
         * get bounded input slices.  Bit-identical: grouping only reorders
         * independent (tile, oc block) computations. */
        uint32_t tile_count =
            (pixels + LW_NHWC_PIXEL_TILE - 1u) / LW_NHWC_PIXEL_TILE;
        uint64_t row_bytes = (uint64_t)input_channels * sizeof(float);
        uint32_t budget_pixels =
            (uint32_t)(UINT64_C(458752) / (row_bytes == 0u ? 1u : row_bytes));
        uint32_t group_tiles =
            (budget_pixels + LW_NHWC_PIXEL_TILE - 1u) / LW_NHWC_PIXEL_TILE;
        if (group_tiles < LW_NHWC_POINTWISE_GROUP_TILES) {
            group_tiles = LW_NHWC_POINTWISE_GROUP_TILES;
        }
        if (group_tiles > tile_count) {
            group_tiles = tile_count;
        }
        lw_avx2_fma_nhwc_pointwise_grouped_f32(
            input, packed_weights, epilogue, output, pixels, input_channels,
            output_channels, group_tiles);
    }
}
