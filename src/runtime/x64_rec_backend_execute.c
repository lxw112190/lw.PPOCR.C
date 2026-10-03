#include "x64_rec_backend_internal.h"
#include "ctc_projection_internal.h"
#include "rec_backend_kernels_internal.h"
#include "operator_internal.h"
#include "../ppocr/profile_internal.h"
#include "../simd/simd_kernels.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#if defined(_MSC_VER)
#include <malloc.h>
static void* rec_aligned_alloc(size_t alignment, size_t size) { return _aligned_malloc(size, alignment); }
static void rec_aligned_free(void* p) { _aligned_free(p); }
#else
static void* rec_aligned_alloc(size_t alignment, size_t size) {
    void* p = NULL;
    return posix_memalign(&p, alignment, size) == 0 ? p : NULL;
}
static void rec_aligned_free(void* p) { free(p); }
#endif

static float* offset_ptr(lw_x64_rec_instance* instance, uint64_t offset) {
    if (instance == NULL || instance->arena == NULL || offset > SIZE_MAX) return NULL;
    return (float*)(void*)(instance->arena + (size_t)offset);
}

static lw_scalar_binary_op binary_operation(uint16_t operation) {
    switch (operation) {
    case LW_OP_ADD: return LW_SCALAR_BINARY_ADD;
    case LW_OP_MUL: return LW_SCALAR_BINARY_MUL;
    case LW_OP_DIV: return LW_SCALAR_BINARY_DIV;
    case LW_OP_SUB: return LW_SCALAR_BINARY_SUB;
    default: return LW_SCALAR_BINARY_POW;
    }
}

static void scalar_nhwc_conv(const lw_x64_rec_conv_op* conv, const float* input,
                             float* output) {
    uint32_t oy, ox, oc, ky, kx, ic;
    for (oy = 0u; oy < conv->output_height; ++oy) {
        for (ox = 0u; ox < conv->output_width; ++ox) {
            for (oc = 0u; oc < conv->output_channels; ++oc) {
                float sum = conv->bias == NULL ? 0.0f : conv->bias[oc];
                for (ky = 0u; ky < conv->kernel_h; ++ky) {
                    int32_t iy = (int32_t)(oy * conv->stride_h + ky) - (int32_t)conv->pad_top;
                    if (iy < 0 || iy >= (int32_t)conv->input_height) continue;
                    for (kx = 0u; kx < conv->kernel_w; ++kx) {
                        int32_t ix = (int32_t)(ox * conv->stride_w + kx) - (int32_t)conv->pad_left;
                        if (ix < 0 || ix >= (int32_t)conv->input_width) continue;
                        for (ic = 0u; ic < conv->input_channels; ++ic) {
                            size_t input_index = ((size_t)iy * conv->input_width + (size_t)ix) * conv->input_channels + ic;
                            size_t weight_index;
                            if (conv->groups == conv->input_channels) {
                                if (ic != oc) continue;
                                weight_index = (((size_t)oc * conv->kernel_h + ky) * conv->kernel_w) + kx;
                            } else {
                                weight_index = ((((size_t)oc * conv->input_channels + ic) * conv->kernel_h + ky) * conv->kernel_w) + kx;
                            }
                            sum += input[input_index] * conv->original_weights[weight_index];                        }
                    }
                }
                output[((size_t)oy * conv->output_width + ox) * conv->output_channels + oc] = sum;
            }
        }
    }
}
static void scalar_nhwc_batch_norm(const lw_x64_rec_affine_op* affine, const float* input, float* output) {
    uint32_t pixel;
    for (pixel = 0u; pixel < affine->pixels; ++pixel) {
        uint32_t channel;
        for (channel = 0u; channel < affine->channels; ++channel) {
            float factor = affine->scale[channel] / sqrtf(affine->variance[channel] + affine->epsilon);
            size_t index = affine->channel_major
                              ? (size_t)channel * affine->pixels + pixel
                              : (size_t)pixel * affine->channels + channel;
            float value = input[index];
            output[index] = (value - affine->mean[channel]) * factor + affine->bias[channel];
        }
    }
}

/* Intra-op sharding for 1x1 pointwise convs: pixels are independent and each
 * output pixel accumulates over the input channels in the kernel's fixed
 * order, so any contiguous pixel split is bit-identical to the serial run. */
typedef struct lw_rec_pointwise_shard {
    const float* input;
    const float* packed_weights;
    lw_nhwc_epilogue epilogue;
    float* output;
    uint32_t pixels;
    uint32_t input_channels;
    uint32_t output_channels;
    uint16_t kernel;
} lw_rec_pointwise_shard;

static void pointwise_run_range(const lw_rec_pointwise_shard* shard,
                                uint32_t pixel_begin, uint32_t pixel_count) {
    lw_nhwc_epilogue ep = shard->epilogue;
    const float* input = shard->input + (size_t)pixel_begin * shard->input_channels;
    float* output = shard->output + (size_t)pixel_begin * shard->output_channels;
    if (ep.residual != NULL) {
        ep.residual += (size_t)pixel_begin * shard->output_channels;
    }
#if defined(__EMSCRIPTEN__)
    lw_rec_backend_kernels_current()->pointwise(
        input, shard->packed_weights, &ep, output, pixel_count,
        shard->input_channels, shard->output_channels);
#else
    switch (shard->kernel) {
    case LW_X64_REC_PW_4X16:
        lw_avx2_fma_nhwc_pointwise_4x16_f32(input, shard->packed_weights, &ep, output,
                                            pixel_count, shard->input_channels,
                                            shard->output_channels);
        break;
    case LW_X64_REC_PW_3X32:
        lw_avx2_fma_nhwc_pointwise_3x32_f32(input, shard->packed_weights, &ep, output,
                                            pixel_count, shard->input_channels,
                                            shard->output_channels);
        break;
    case LW_X64_REC_PW_2X32:
        lw_avx2_fma_nhwc_pointwise_2x32_f32(input, shard->packed_weights, &ep, output,
                                            pixel_count, shard->input_channels,
                                            shard->output_channels);
        break;
    default:
        lw_avx2_fma_nhwc_pointwise_f32(input, shard->packed_weights, &ep, output,
                                       pixel_count, shard->input_channels,
                                       shard->output_channels);
        break;
    }
#endif
}

/* Shard begins use 12-pixel blocks: the kernel tile widths (2, 3, 4, 6) all
 * divide 12, so every worker starts on a clean tile boundary.  The final
 * shard takes the arbitrary tail. */
#define LW_REC_POINTWISE_SHARD_GRAIN 12u
static uint32_t pointwise_shard_begin(uint32_t pixels, uint32_t worker_count,
                                      uint32_t worker_index) {
    uint32_t blocks = (pixels + LW_REC_POINTWISE_SHARD_GRAIN - 1u) /
                      LW_REC_POINTWISE_SHARD_GRAIN;
    uint32_t base = blocks / worker_count;
    uint32_t remainder = blocks % worker_count;
    uint32_t block = worker_index * base + (worker_index < remainder ? worker_index : remainder);
    uint32_t begin = block * LW_REC_POINTWISE_SHARD_GRAIN;
    return begin > pixels ? pixels : begin;
}

static void pointwise_shard_entry(void* context_void, uint32_t worker_index,
                                  uint32_t worker_count) {
    const lw_rec_pointwise_shard* shard = (const lw_rec_pointwise_shard*)context_void;
    uint32_t begin = pointwise_shard_begin(shard->pixels, worker_count, worker_index);
    uint32_t end = pointwise_shard_begin(shard->pixels, worker_count, worker_index + 1u);
    if (end > begin) {
        pointwise_run_range(shard, begin, end - begin);
    }
}

/* Minimum MACs before worker fan-out pays for synchronization. */
#define LW_REC_POINTWISE_SHARD_MIN_WORK (4ull * 1000ull * 1000ull)
static uint32_t pointwise_shard_workers(const lw_x64_rec_instance* instance,
                                        uint32_t pixels, uint32_t input_channels,
                                        uint32_t output_channels) {
    uint64_t work;
    uint32_t workers;
    if (instance->thread_pool == NULL || instance->intra_op_workers <= 1u ||
        pixels < 2u * LW_REC_POINTWISE_SHARD_GRAIN) {
        return 1u;
    }
    work = (uint64_t)pixels * input_channels * output_channels;
    if (work < LW_REC_POINTWISE_SHARD_MIN_WORK) {
        return 1u;
    }
    workers = instance->intra_op_workers;
    if (workers > pixels / LW_REC_POINTWISE_SHARD_GRAIN) {
        workers = pixels / LW_REC_POINTWISE_SHARD_GRAIN;
    }
    return workers <= 1u ? 1u : workers;
}

/* Cache-local FFN scheduling. A tile boundary must divide every supported
 * Pointwise row tile (2/3/4/6); channel accumulation and epilogues stay in the
 * existing kernels. Reuse the planned intermediate slot: no new allocation. */
#ifndef LW_REC_FFN_TILE_PIXELS
#define LW_REC_FFN_TILE_PIXELS 96u
#endif
typedef struct lw_rec_ffn_shard {
    lw_rec_pointwise_shard expansion;
    lw_rec_pointwise_shard projection;
} lw_rec_ffn_shard;

static void ffn_shard_entry(void* context_void, uint32_t worker_index,
                             uint32_t worker_count) {
    const lw_rec_ffn_shard* pair = (const lw_rec_ffn_shard*)context_void;
    uint32_t begin = pointwise_shard_begin(pair->expansion.pixels, worker_count, worker_index);
    uint32_t end = pointwise_shard_begin(pair->expansion.pixels, worker_count, worker_index + 1u);
    while (begin < end) {
        uint32_t count = end - begin;
        if (count > LW_REC_FFN_TILE_PIXELS) count = LW_REC_FFN_TILE_PIXELS;
        pointwise_run_range(&pair->expansion, begin, count);
        pointwise_run_range(&pair->projection, begin, count);
        begin += count;
    }
}

static int ffn_span_valid(uint64_t offset, uint64_t bytes, uint64_t capacity) {
    return offset <= capacity && bytes <= capacity - offset;
}

static int ffn_disjoint(uint64_t a, uint64_t a_bytes, uint64_t b, uint64_t b_bytes) {
    return a <= b ? a_bytes <= b - a : b_bytes <= a - b;
}

static void ffn_init_shard(lw_x64_rec_instance* instance,
                            const lw_x64_rec_conv_op* conv, uint32_t pixels,
                            lw_rec_pointwise_shard* shard) {
    memset(shard, 0, sizeof(*shard));
    shard->input = offset_ptr(instance, conv->input_offset);
    shard->output = offset_ptr(instance, conv->output_offset);
    shard->packed_weights = conv->packed_weights;
    shard->pixels = pixels;
    shard->input_channels = conv->input_channels;
    shard->output_channels = conv->output_channels;
    shard->kernel = conv->pointwise_kernel;
    shard->epilogue.bias = conv->bias;
    shard->epilogue.post_bias = conv->post_bias;
    shard->epilogue.activation = conv->activation;
    if (conv->has_residual)
        shard->epilogue.residual = offset_ptr(instance, conv->residual_offset);
}

lw_status lw_x64_rec_instance_run_ffn_pair(lw_x64_rec_instance* instance,
                                           uint32_t op_index, lw_error* error) {
    const lw_x64_rec_program* program;
    const lw_x64_rec_op* first;
    const lw_x64_rec_op* second;
    const lw_x64_rec_conv_op *a, *b;
    uint64_t pixels64, input_bytes, middle_bytes, output_bytes;
    uint32_t pixels, workers;
    lw_rec_ffn_shard pair;
    if (instance == NULL || instance->program == NULL || instance->arena == NULL ||
        op_index >= instance->program->op_count ||
        instance->program->op_count - op_index < 2u) goto invalid;
    program = instance->program;
    if (program->arena_bytes > SIZE_MAX) goto invalid;
    first = &program->ops[op_index];
    second = &program->ops[op_index + 1u];
    a = &first->data.conv;
    b = &second->data.conv;
    if (!(first->flags & LW_X64_REC_OP_FFN_BEGIN) ||
        first->kind != LW_X64_REC_OP_POINTWISE || second->kind != LW_X64_REC_OP_POINTWISE ||
        a->scalar_fallback || b->scalar_fallback || a->has_residual ||
        a->kernel_h != 1u || a->kernel_w != 1u || b->kernel_h != 1u || b->kernel_w != 1u ||
        a->stride_h != 1u || a->stride_w != 1u || b->stride_h != 1u || b->stride_w != 1u ||
        a->groups != 1u || b->groups != 1u ||
        a->activation != LW_NHWC_ACT_GELU || b->activation != LW_NHWC_ACT_NONE ||
        a->packed_weights == NULL || b->packed_weights == NULL ||
        a->output_offset != b->input_offset || a->output_channels != b->input_channels ||
        a->input_channels == 0u || a->output_channels == 0u || b->output_channels == 0u ||
        a->input_height != b->input_height || a->input_width != b->input_width ||
        a->input_height != a->output_height || a->input_width != a->output_width ||
        b->input_height != b->output_height || b->input_width != b->output_width) goto invalid;
    pixels64 = (uint64_t)a->input_height * a->input_width;
    if (pixels64 == 0u || pixels64 > UINT32_MAX) goto invalid;
    pixels = (uint32_t)pixels64;
    if (pixels64 > UINT64_MAX / ((uint64_t)a->input_channels * sizeof(float)) ||
        pixels64 > UINT64_MAX / ((uint64_t)a->output_channels * sizeof(float)) ||
        pixels64 > UINT64_MAX / ((uint64_t)b->output_channels * sizeof(float))) goto invalid;
    input_bytes = pixels64 * a->input_channels * sizeof(float);
    middle_bytes = pixels64 * a->output_channels * sizeof(float);
    output_bytes = pixels64 * b->output_channels * sizeof(float);
    if (!ffn_span_valid(a->input_offset, input_bytes, program->arena_bytes) ||
        !ffn_span_valid(a->output_offset, middle_bytes, program->arena_bytes) ||
        !ffn_span_valid(b->output_offset, output_bytes, program->arena_bytes) ||
        !ffn_disjoint(a->input_offset, input_bytes, a->output_offset, middle_bytes) ||
        !ffn_disjoint(a->input_offset, input_bytes, b->output_offset, output_bytes) ||
        !ffn_disjoint(a->output_offset, middle_bytes, b->output_offset, output_bytes)) goto invalid;
    if (b->has_residual &&
        (!ffn_span_valid(b->residual_offset, output_bytes, program->arena_bytes) ||
         !ffn_disjoint(b->residual_offset, output_bytes, a->output_offset, middle_bytes) ||
         !ffn_disjoint(b->residual_offset, output_bytes, b->output_offset, output_bytes))) goto invalid;
    ffn_init_shard(instance, a, pixels, &pair.expansion);
    ffn_init_shard(instance, b, pixels, &pair.projection);
    workers = pointwise_shard_workers(instance, pixels, a->input_channels, a->output_channels);
    if (workers > 1u) {
        lw_thread_pool_run(instance->thread_pool, workers, ffn_shard_entry, &pair);
    } else {
        ffn_shard_entry(&pair, 0u, 1u);
    }
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
invalid:
    lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "REC FFN pair/lifetime is invalid");
    return LW_STATUS_INVALID_ARGUMENT;
}

/* Intra-op sharding for dense convs along output rows.  The kernel accepts an
 * output_row_offset and computes every output pixel with the same tap order
 * as the serial run, so row splits are bit-identical.  Each worker gets its
 * own scratch slice: the per-op scratch geometry is row-count independent. */
typedef struct lw_rec_dense_shard {
    const float* input;
    const float* packed_weights;
    lw_nhwc_epilogue epilogue;
    float* output;
    lw_nhwc_dense_desc desc;
    uint8_t* scratch_base;
    uint64_t scratch_stride;
    uint64_t scratch_bytes;
    lw_status status[LW_PARALLEL_MAX_WORKERS];
} lw_rec_dense_shard;

static void dense_shard_entry(void* context_void, uint32_t worker_index,
                              uint32_t worker_count) {
    lw_rec_dense_shard* shard = (lw_rec_dense_shard*)context_void;
    const uint32_t full_rows = shard->desc.output_height +
                               shard->desc.output_row_offset;
    uint32_t base = full_rows / worker_count;
    uint32_t remainder = full_rows % worker_count;
    uint32_t row_begin = worker_index * base +
                         (worker_index < remainder ? worker_index : remainder);
    uint32_t row_end = row_begin + base + (worker_index < remainder ? 1u : 0u);
    lw_nhwc_dense_desc desc;
    lw_nhwc_epilogue ep;
    float* output;
    if (row_end <= row_begin) {
        shard->status[worker_index] = LW_STATUS_OK;
        return;
    }
    desc = shard->desc;
    desc.output_height = row_end - row_begin;
    desc.output_row_offset = row_begin;
    ep = shard->epilogue;
    output = shard->output +
             (size_t)row_begin * desc.output_width * desc.output_channels;
    if (ep.residual != NULL) {
        ep.residual += (size_t)row_begin * desc.output_width * desc.output_channels;
    }
    shard->status[worker_index] = lw_avx2_fma_nhwc_dense_f32(
        shard->input, shard->packed_weights, &ep, output, &desc,
        shard->scratch_base + (size_t)worker_index * shard->scratch_stride,
        shard->scratch_bytes);
}

static uint32_t dense_shard_workers(const lw_x64_rec_instance* instance,
                                    const lw_x64_rec_conv_op* conv) {
    uint64_t work;
    uint32_t workers;
    uint32_t output_rows;
    if (instance->thread_pool == NULL || instance->intra_op_workers <= 1u) {
        return 1u;
    }
    output_rows = conv->output_height;
    if (output_rows < 2u) {
        return 1u;
    }
    work = (uint64_t)output_rows * conv->output_width * conv->output_channels *
           conv->input_channels * conv->kernel_h * conv->kernel_w;
    if (work < LW_REC_POINTWISE_SHARD_MIN_WORK) {
        return 1u;
    }
    workers = instance->intra_op_workers;
    if (workers > output_rows) {
        workers = output_rows;
    }
    if (workers > LW_PARALLEL_MAX_WORKERS) {
        workers = LW_PARALLEL_MAX_WORKERS;
    }
    return workers <= 1u ? 1u : workers;
}

static uint8_t* dense_shard_scratch(lw_x64_rec_instance* instance, uint32_t workers,
                                    uint64_t scratch_bytes) {
    uint64_t needed;
    if (scratch_bytes == 0u || workers <= 1u) {
        return NULL;
    }
    needed = (uint64_t)workers * scratch_bytes;
    if (instance->dense_shard_scratch_bytes < needed) {
        uint8_t* grown;
        rec_aligned_free(instance->dense_shard_scratch);
        instance->dense_shard_scratch = NULL;
        instance->dense_shard_scratch_bytes = 0u;
        grown = (uint8_t*)rec_aligned_alloc(64u, (size_t)needed);
        if (grown == NULL) {
            return NULL;
        }
        instance->dense_shard_scratch = grown;
        instance->dense_shard_scratch_bytes = needed;
    }
    return instance->dense_shard_scratch;
}

/* Depthwise convs shard along output rows like dense ones; the kernel applies
 * output_row_offset to the input window and needs no scratch.  Every output
 * pixel keeps the serial tap order, so splits are bit-identical. */
typedef struct lw_rec_depthwise_shard {
    const float* input;
    const float* packed_weights;
    const float* bias;
    float* output;
    lw_nhwc_depthwise_desc desc;
    lw_status status[LW_PARALLEL_MAX_WORKERS];
} lw_rec_depthwise_shard;

static void depthwise_shard_entry(void* context_void, uint32_t worker_index,
                                  uint32_t worker_count) {
    lw_rec_depthwise_shard* shard = (lw_rec_depthwise_shard*)context_void;
    const uint32_t full_rows = shard->desc.output_height +
                               shard->desc.output_row_offset;
    uint32_t base = full_rows / worker_count;
    uint32_t remainder = full_rows % worker_count;
    uint32_t row_begin = worker_index * base +
                         (worker_index < remainder ? worker_index : remainder);
    uint32_t row_end = row_begin + base + (worker_index < remainder ? 1u : 0u);
    lw_nhwc_depthwise_desc desc;
    float* output;
    if (row_end <= row_begin) {
        shard->status[worker_index] = LW_STATUS_OK;
        return;
    }
    desc = shard->desc;
    desc.output_height = row_end - row_begin;
    desc.output_row_offset = row_begin;
    output = shard->output +
             (size_t)row_begin * desc.output_width * desc.channels;
    shard->status[worker_index] = lw_avx2_fma_nhwc_depthwise_f32(
        shard->input, shard->packed_weights, shard->bias, output, &desc);
}

static uint32_t depthwise_shard_workers(const lw_x64_rec_instance* instance,
                                        const lw_x64_rec_conv_op* conv) {
    uint64_t work;
    uint32_t workers;
    uint32_t output_rows;
    if (instance->thread_pool == NULL || instance->intra_op_workers <= 1u) {
        return 1u;
    }
    output_rows = conv->output_height;
    if (output_rows < 2u) {
        return 1u;
    }
    work = (uint64_t)output_rows * conv->output_width * conv->input_channels *
           conv->kernel_h * conv->kernel_w;
    if (work < LW_REC_POINTWISE_SHARD_MIN_WORK) {
        return 1u;
    }
    workers = instance->intra_op_workers;
    if (workers > output_rows) {
        workers = output_rows;
    }
    if (workers > LW_PARALLEL_MAX_WORKERS) {
        workers = LW_PARALLEL_MAX_WORKERS;
    }
    return workers <= 1u ? 1u : workers;
}

/* Matmul sharding.  Two bit-identical modes:
 * - mode 1: rank-2 shared-weight matmul, split over rows (4-row grain keeps
 *   the AVX2 row blocking; the kernel tail handles the remainder rows);
 * - mode 2: attention-style rank-4 [1,B,M,K] x [1,B,K,N] (no broadcasting),
 *   split over B, each batch dispatched as a rank-2 matmul.
 * Every output element accumulates over the inner axis in the serial order. */
typedef struct lw_rec_matmul_shard {
    const float* input;
    const float* right;
    float* output;
    uint32_t batch;
    uint32_t rows;
    uint32_t inner;
    uint32_t columns;
    uint32_t mode;
    lw_status status[LW_PARALLEL_MAX_WORKERS];
} lw_rec_matmul_shard;

static uint32_t matmul_shard_row_begin(uint32_t rows, uint32_t worker_count,
                                       uint32_t worker_index) {
    uint32_t blocks = (rows + 3u) / 4u;
    uint32_t base = blocks / worker_count;
    uint32_t remainder = blocks % worker_count;
    uint32_t block = worker_index * base + (worker_index < remainder ? worker_index : remainder);
    uint32_t begin = block * 4u;
    return begin > rows ? rows : begin;
}

static void matmul_shard_entry(void* context_void, uint32_t worker_index,
                               uint32_t worker_count) {
    lw_rec_matmul_shard* shard = (lw_rec_matmul_shard*)context_void;
    shard->status[worker_index] = LW_STATUS_OK;
    if (shard->mode == 1u) {
        uint32_t row_begin = matmul_shard_row_begin(shard->rows, worker_count, worker_index);
        uint32_t row_end = matmul_shard_row_begin(shard->rows, worker_count, worker_index + 1u);
        if (row_end <= row_begin) return;
        shard->status[worker_index] = lw_matmul_shared_f32(
            shard->input + (size_t)row_begin * shard->inner, shard->right,
            shard->output + (size_t)row_begin * shard->columns, 1u, row_end - row_begin,
            shard->inner, shard->columns);
    } else {
        /* Per-batch rank-2 view through the same scalar kernel the unsplit
         * op would use, so every element keeps the exact accumulation order. */
        int32_t in_dims[2];
        int32_t w_dims[2];
        int32_t out_dims[2];
        uint32_t base = shard->batch / worker_count;
        uint32_t remainder = shard->batch % worker_count;
        uint32_t batch_begin = worker_index * base +
                               (worker_index < remainder ? worker_index : remainder);
        uint32_t batch_end = batch_begin + base + (worker_index < remainder ? 1u : 0u);
        uint32_t b;
        in_dims[0] = (int32_t)shard->rows; in_dims[1] = (int32_t)shard->inner;
        w_dims[0] = (int32_t)shard->inner; w_dims[1] = (int32_t)shard->columns;
        out_dims[0] = (int32_t)shard->rows; out_dims[1] = (int32_t)shard->columns;
        for (b = batch_begin; b < batch_end; ++b) {
            lw_status status = lw_scalar_matmul_f32(
                shard->input + (size_t)b * shard->rows * shard->inner,
                shard->right + (size_t)b * shard->inner * shard->columns,
                shard->output + (size_t)b * shard->rows * shard->columns,
                2u, in_dims, 2u, w_dims, 2u, out_dims);
            if (status != LW_STATUS_OK) {
                shard->status[worker_index] = status;
                return;
            }
        }
    }
}

static uint32_t matmul_shard_workers(const lw_x64_rec_instance* instance,
                                     const lw_x64_rec_matmul_op* matmul,
                                     uint32_t* out_mode) {
    uint64_t work = (uint64_t)matmul->batch * matmul->rows * matmul->inner *
                    matmul->columns;
    uint32_t workers;
    *out_mode = 0u;
    if (instance->thread_pool == NULL || instance->intra_op_workers <= 1u ||
        work < LW_REC_POINTWISE_SHARD_MIN_WORK) {
        return 1u;
    }
    workers = instance->intra_op_workers;
    if (workers > LW_PARALLEL_MAX_WORKERS) {
        workers = LW_PARALLEL_MAX_WORKERS;
    }
    if (matmul->weights_rank == 2u && matmul->batch == 1u && matmul->rows >= 8u) {
        /* Rank-2: row split. */
        if (workers > matmul->rows / 4u) workers = matmul->rows / 4u;
        *out_mode = 1u;
    } else if (matmul->input_rank == 4u && matmul->weights_rank == 4u &&
               matmul->output_rank == 4u && matmul->batch > 1u &&
               matmul->input_dimensions[0] == 1 && matmul->weights_dimensions[0] == 1 &&
               matmul->output_dimensions[0] == 1 &&
               matmul->input_dimensions[1] == (int32_t)matmul->batch &&
               matmul->weights_dimensions[1] == (int32_t)matmul->batch &&
               matmul->output_dimensions[1] == (int32_t)matmul->batch) {
        /* Attention-style batched matmul without broadcasting: batch split. */
        if (workers > matmul->batch) workers = matmul->batch;
        *out_mode = 2u;
    } else {
        return 1u;
    }
    return workers <= 1u ? 1u : workers;
}

static lw_status execute_op(lw_x64_rec_instance* instance, const lw_x64_rec_op* op,
                            lw_error* error) {
    (void)error;
    float* input;
    float* output;
    if (op == NULL) return LW_STATUS_INVALID_ARGUMENT;
    switch (op->kind) {
#if !defined(__EMSCRIPTEN__)
    case LW_X64_REC_OP_POINTWISE_NCHW:
    case LW_X64_REC_OP_STEM_NCHW:
    case LW_X64_REC_OP_DEPTHWISE_NCHW: {
        int32_t input_dimensions[4] = {
            1, (int32_t)op->data.conv.input_channels,
            (int32_t)op->data.conv.input_height,
            (int32_t)op->data.conv.input_width
        };
        int32_t output_dimensions[4] = {
            1, (int32_t)op->data.conv.output_channels,
            (int32_t)op->data.conv.output_height,
            (int32_t)op->data.conv.output_width
        };
        input = offset_ptr(instance, op->data.conv.input_offset);
        output = offset_ptr(instance, op->data.conv.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        if (op->kind == LW_X64_REC_OP_POINTWISE_NCHW) {
            if (op->data.conv.nchw_pointwise_kernel == LW_X64_REC_NCHW_PW_FMA8) {
                lw_avx2_fma_packed_conv1x1_8x8_f32(
                    input, op->data.conv.packed_weights, op->data.conv.bias,
                    output, input_dimensions, output_dimensions);
            } else {
                lw_avx2_fma_packed_conv1x1_f32(
                    input, op->data.conv.packed_weights, op->data.conv.bias,
                    output, input_dimensions, output_dimensions);
            }
        } else if (op->kind == LW_X64_REC_OP_STEM_NCHW) {
            lw_avx2_fma_packed_conv3x3_stride2_pad1_f32(
                input, op->data.conv.packed_weights, op->data.conv.bias,
                output, input_dimensions, output_dimensions);
        } else if (op->data.conv.kernel_h == 1u && op->data.conv.kernel_w == 5u) {
            int32_t weight_dimensions[4] = {
                (int32_t)op->data.conv.output_channels, 1, (int32_t)op->data.conv.weight_h, (int32_t)op->data.conv.weight_w
            };
            int32_t kernel[2] = { 1, 5 };
            int32_t strides[2] = { (int32_t)op->data.conv.stride_h, (int32_t)op->data.conv.stride_w };
            int32_t dilations[2] = { 1, 1 };
            int32_t pads[4] = { (int32_t)op->data.conv.pad_top, (int32_t)op->data.conv.pad_left,
                                      (int32_t)op->data.conv.pad_bottom, (int32_t)op->data.conv.pad_right };
            {
                lw_status fallback_status = lw_scalar_conv2d_f32(input, op->data.conv.original_weights, op->data.conv.bias,
                                                                  (op->data.conv.bias != NULL ? op->data.conv.output_channels : 0u), output, input_dimensions,
                                                                  weight_dimensions, output_dimensions, kernel, strides,
                                                                  dilations, pads, op->data.conv.groups);
                if (fallback_status != LW_STATUS_OK) { lw_set_error(error, fallback_status, "NCHW depthwise fallback failed"); return fallback_status; }
            }
        } else if (op->data.conv.stride_h == 2u && op->data.conv.stride_w == 1u) {
            lw_avx2_depthwise_conv3x3_stride2x1_pad1_f32(
                input, op->data.conv.original_weights, op->data.conv.bias,
                output, input_dimensions, output_dimensions);
        } else {
            lw_avx2_depthwise_conv3x3_unit_pad1_f32(
                input, op->data.conv.original_weights, op->data.conv.bias,
                output, input_dimensions);
        }
        return LW_STATUS_OK;
    }
#endif
    case LW_X64_REC_OP_POINTWISE: {
        lw_nhwc_epilogue ep = { NULL, NULL, op->data.conv.activation, 0u, 0.0f, 0.0f, NULL };
        input = offset_ptr(instance, op->data.conv.input_offset);
        output = offset_ptr(instance, op->data.conv.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        ep.bias = op->data.conv.bias;
        ep.post_bias = op->data.conv.post_bias;
        if (op->data.conv.has_residual != 0u) {
            ep.residual = offset_ptr(instance, op->data.conv.residual_offset);
            if (ep.residual == NULL) return LW_STATUS_INVALID_ARGUMENT;
        }
        {
            uint32_t pixels = op->data.conv.input_height * op->data.conv.input_width;
            uint32_t shard_workers = pointwise_shard_workers(instance, pixels,
                op->data.conv.input_channels, op->data.conv.output_channels);
            if (shard_workers > 1u) {
                lw_rec_pointwise_shard shard;
                shard.input = input;
                shard.packed_weights = op->data.conv.packed_weights;
                shard.epilogue = ep;
                shard.output = output;
                shard.pixels = pixels;
                shard.input_channels = op->data.conv.input_channels;
                shard.output_channels = op->data.conv.output_channels;
                shard.kernel = op->data.conv.pointwise_kernel;
                lw_thread_pool_run(instance->thread_pool, shard_workers,
                                   pointwise_shard_entry, &shard);
                return LW_STATUS_OK;
            }
#if defined(__EMSCRIPTEN__)
            lw_rec_backend_kernels_current()->pointwise(
                input, op->data.conv.packed_weights, &ep, output, pixels,
                op->data.conv.input_channels, op->data.conv.output_channels);
#else
            switch (op->data.conv.pointwise_kernel) {
            case LW_X64_REC_PW_4X16:
                lw_avx2_fma_nhwc_pointwise_4x16_f32(
                    input, op->data.conv.packed_weights, &ep, output, pixels,
                    op->data.conv.input_channels, op->data.conv.output_channels);
                break;
            case LW_X64_REC_PW_3X32:
                lw_avx2_fma_nhwc_pointwise_3x32_f32(
                    input, op->data.conv.packed_weights, &ep, output, pixels,
                    op->data.conv.input_channels, op->data.conv.output_channels);
                break;
            case LW_X64_REC_PW_2X32:
                lw_avx2_fma_nhwc_pointwise_2x32_f32(
                    input, op->data.conv.packed_weights, &ep, output, pixels,
                    op->data.conv.input_channels, op->data.conv.output_channels);
                break;
            default:
                lw_avx2_fma_nhwc_pointwise_f32(
                    input, op->data.conv.packed_weights, &ep, output, pixels,
                    op->data.conv.input_channels, op->data.conv.output_channels);
                break;
            }
#endif
        }
        return LW_STATUS_OK;
    }
    case LW_X64_REC_OP_DENSE: {
        lw_nhwc_dense_desc desc;
        lw_nhwc_epilogue ep;
        lw_status status;
        memset(&desc, 0, sizeof(desc));
        desc.batch = 1u;
        desc.input_channels = op->data.conv.input_channels;
        desc.input_height = op->data.conv.input_height;
        desc.input_width = op->data.conv.input_width;
        desc.output_channels = op->data.conv.output_channels;
        desc.output_height = op->data.conv.output_height;
        desc.output_width = op->data.conv.output_width;
        desc.kernel_h = op->data.conv.kernel_h;
        desc.kernel_w = op->data.conv.kernel_w;
        desc.stride_h = op->data.conv.stride_h;
        desc.stride_w = op->data.conv.stride_w;
        desc.pad_top = op->data.conv.pad_top;
        desc.pad_left = op->data.conv.pad_left;
        desc.pad_bottom = op->data.conv.pad_bottom;
        desc.pad_right = op->data.conv.pad_right;
        desc.dense_kc = op->data.conv.dense_kc;
        memset(&ep, 0, sizeof(ep));
        ep.bias = op->data.conv.bias;
        ep.activation = op->data.conv.activation;
        ep.post_bias = op->data.conv.post_bias;
        input = offset_ptr(instance, op->data.conv.input_offset);
        output = offset_ptr(instance, op->data.conv.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        if (op->data.conv.has_residual != 0u) {
            ep.residual = offset_ptr(instance, op->data.conv.residual_offset);
            if (ep.residual == NULL) return LW_STATUS_INVALID_ARGUMENT;
        }
        {
            uint32_t shard_workers = dense_shard_workers(instance, &op->data.conv);
            if (shard_workers > 1u) {
                uint8_t* shard_scratch = dense_shard_scratch(instance, shard_workers,
                                                             op->data.conv.scratch_bytes);
                if (shard_scratch != NULL) {
                    lw_rec_dense_shard shard;
                    uint32_t wi;
                    shard.input = input;
                    shard.packed_weights = op->data.conv.packed_weights;
                    shard.epilogue = ep;
                    shard.output = output;
                    shard.desc = desc;
                    shard.scratch_base = shard_scratch;
                    shard.scratch_stride = op->data.conv.scratch_bytes;
                    shard.scratch_bytes = op->data.conv.scratch_bytes;
                    lw_thread_pool_run(instance->thread_pool, shard_workers,
                                       dense_shard_entry, &shard);
                    status = LW_STATUS_OK;
                    for (wi = 0u; wi < shard_workers; ++wi) {
                        if (shard.status[wi] != LW_STATUS_OK) {
                            status = shard.status[wi];
                            break;
                        }
                    }
                    if (status == LW_STATUS_OK) return LW_STATUS_OK;
                    if (ep.post_bias != NULL || ep.residual != NULL ||
                        ep.activation != LW_NHWC_ACT_NONE) {
                        /* The scalar fallback cannot reproduce the fused epilogue. */
                        return status;
                    }
                    scalar_nhwc_conv(&op->data.conv, input, output);
                    return LW_STATUS_OK;
                }
            }
        }
#if defined(__EMSCRIPTEN__)
        status = lw_rec_backend_kernels_current()->dense(
#else
        status = lw_avx2_fma_nhwc_dense_f32(
#endif
            input, op->data.conv.packed_weights, &ep, output, &desc,
            instance->scratch, op->data.conv.scratch_bytes);
        if (status == LW_STATUS_OK) return LW_STATUS_OK;
        if (ep.post_bias != NULL || ep.residual != NULL ||
            ep.activation != LW_NHWC_ACT_NONE) {
            /* The scalar fallback cannot reproduce the fused epilogue. */
            return status;
        }
        scalar_nhwc_conv(&op->data.conv, input, output);
        return LW_STATUS_OK;
    }
    case LW_X64_REC_OP_DEPTHWISE: {
        lw_nhwc_depthwise_desc desc;
        lw_status status;
        memset(&desc, 0, sizeof(desc));
        desc.batch = 1u;
        desc.channels = op->data.conv.input_channels;
        desc.input_height = op->data.conv.input_height;
        desc.input_width = op->data.conv.input_width;
        desc.output_height = op->data.conv.output_height;
        desc.output_width = op->data.conv.output_width;
        desc.kernel_h = op->data.conv.kernel_h;
        desc.kernel_w = op->data.conv.kernel_w;
        desc.stride_h = op->data.conv.stride_h;
        desc.stride_w = op->data.conv.stride_w;
        desc.pad_top = op->data.conv.pad_top;
        desc.pad_left = op->data.conv.pad_left;
        desc.pad_bottom = op->data.conv.pad_bottom;
        desc.pad_right = op->data.conv.pad_right;
        desc.post_bias = op->data.conv.post_bias;
        input = offset_ptr(instance, op->data.conv.input_offset);
        output = offset_ptr(instance, op->data.conv.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        {
            uint32_t shard_workers = depthwise_shard_workers(instance, &op->data.conv);
            if (shard_workers > 1u) {
                lw_rec_depthwise_shard shard;
                uint32_t wi;
                shard.input = input;
                shard.packed_weights = op->data.conv.packed_weights;
                shard.bias = op->data.conv.bias;
                shard.output = output;
                shard.desc = desc;
                lw_thread_pool_run(instance->thread_pool, shard_workers,
                                   depthwise_shard_entry, &shard);
                status = LW_STATUS_OK;
                for (wi = 0u; wi < shard_workers; ++wi) {
                    if (shard.status[wi] != LW_STATUS_OK) {
                        status = shard.status[wi];
                        break;
                    }
                }
                if (status == LW_STATUS_OK) return LW_STATUS_OK;
                if (desc.post_bias != NULL) {
                    /* The scalar fallback cannot reproduce the fused epilogue. */
                    return status;
                }
                scalar_nhwc_conv(&op->data.conv, input, output);
                return LW_STATUS_OK;
            }
        }
#if defined(__EMSCRIPTEN__)
        status = lw_rec_backend_kernels_current()->depthwise(input, op->data.conv.packed_weights,
#else
        status = lw_avx2_fma_nhwc_depthwise_f32(input, op->data.conv.packed_weights,
#endif
                                                 op->data.conv.bias, output, &desc);
        if (status == LW_STATUS_OK) return LW_STATUS_OK;
        if (desc.post_bias != NULL) {
            /* The scalar fallback cannot reproduce the fused epilogue. */
            return status;
        }
        scalar_nhwc_conv(&op->data.conv, input, output);
        return LW_STATUS_OK;
    }
    case LW_X64_REC_OP_AFFINE:
        input = offset_ptr(instance, op->data.affine.input_offset); output = offset_ptr(instance, op->data.affine.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        if (op->data.affine.channel_major && instance->program->backend_layout == LW_X64_REC_BACKEND_NCHW) {
            lw_avx2_nchw_affine_f32(input, op->data.affine.mul, op->data.affine.add,
                                    output, op->data.affine.channels, op->data.affine.pixels);
        } else if (op->data.affine.channel_major) {
            scalar_nhwc_batch_norm(&op->data.affine, input, output);
        } else {
#if defined(__EMSCRIPTEN__) && defined(LW_WASM_REC_SIMD_KERNELS)
            lw_wasm128_affine_nhwc_f32(input, op->data.affine.mul, op->data.affine.add,
                                       output, op->data.affine.pixels, op->data.affine.channels);
#else
            lw_avx2_nhwc_affine_f32(input, op->data.affine.mul, op->data.affine.add,
                                    output, op->data.affine.pixels, op->data.affine.channels);
#endif
        }
        return LW_STATUS_OK;
    case LW_X64_REC_OP_ADD: case LW_X64_REC_OP_MUL: case LW_X64_REC_OP_DIV:
    case LW_X64_REC_OP_SUB: case LW_X64_REC_OP_POW:
        input = op->data.binary.left_constant != NULL ? (float*)(uintptr_t)op->data.binary.left_constant : offset_ptr(instance, op->data.binary.left_offset);
        output = offset_ptr(instance, op->data.binary.output_offset);
        if (op->data.binary.channel_major_nchw != 0u &&
            instance->program->backend_layout == LW_X64_REC_BACKEND_NCHW &&
            (op->data.binary.broadcast_kind == LW_X64_REC_BROADCAST_LEFT_CHANNEL ||
             op->data.binary.broadcast_kind == LW_X64_REC_BROADCAST_RIGHT_CHANNEL)) {
            float* full = op->data.binary.broadcast_kind == LW_X64_REC_BROADCAST_LEFT_CHANNEL
                ? offset_ptr(instance, op->data.binary.right_offset) : input;
            float* channel = op->data.binary.broadcast_kind == LW_X64_REC_BROADCAST_LEFT_CHANNEL
                ? (op->data.binary.left_constant != NULL
                    ? (float*)(uintptr_t)op->data.binary.left_constant
                    : offset_ptr(instance, op->data.binary.left_offset))
                : (op->data.binary.right_constant != NULL
                    ? (float*)(uintptr_t)op->data.binary.right_constant
                    : offset_ptr(instance, op->data.binary.right_offset));
            if (full == NULL || channel == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
            lw_avx2_binary_channel_nchw_f32(
                binary_operation(op->data.binary.operation), full, channel, output,
                op->data.binary.pixels, op->data.binary.channels,
                op->data.binary.broadcast_kind == LW_X64_REC_BROADCAST_LEFT_CHANNEL);
            return LW_STATUS_OK;
        }
        if (op->data.binary.broadcast_kind == LW_X64_REC_BROADCAST_LEFT_CHANNEL) {
            float* full = offset_ptr(instance, op->data.binary.right_offset);
            float* channel = op->data.binary.left_constant != NULL ? (float*)(uintptr_t)op->data.binary.left_constant : offset_ptr(instance, op->data.binary.left_offset);
            if (full == NULL || channel == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
#if defined(__EMSCRIPTEN__) && defined(LW_WASM_REC_SIMD_KERNELS)
            lw_wasm128_binary_channel_nhwc_f32(binary_operation(op->data.binary.operation),
                full, channel, output, op->data.binary.pixels, op->data.binary.channels, 1);
#else
            lw_avx2_binary_channel_f32(binary_operation(op->data.binary.operation), full, channel, output, op->data.binary.pixels, op->data.binary.channels, 1);
#endif
            return LW_STATUS_OK;
        }
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        if (op->data.binary.broadcast_kind == LW_X64_REC_BROADCAST_RIGHT_SCALAR && op->data.binary.right_constant != NULL) {
#if defined(__EMSCRIPTEN__) && defined(LW_WASM_REC_SIMD_KERNELS)
            lw_wasm128_binary_scalar_f32(binary_operation(op->data.binary.operation), input,
                op->data.binary.right_constant[0], output, op->data.binary.element_count, 0);
#else
            lw_avx2_binary_right_scalar_f32(binary_operation(op->data.binary.operation), input,
                                             op->data.binary.right_constant[0], output, op->data.binary.element_count);
#endif
            return LW_STATUS_OK;
        }
        if (op->data.binary.broadcast_kind == LW_X64_REC_BROADCAST_SAME) {
            float* right = op->data.binary.right_constant != NULL ? (float*)(uintptr_t)op->data.binary.right_constant : offset_ptr(instance, op->data.binary.right_offset);
            if (right == NULL) return LW_STATUS_INVALID_ARGUMENT;
            if (op->data.binary.mixed_nhwc != 0u) {
                /* Cross-layout SAME: one side is canonical [N,C,H,W] order,
                 * the other physical NHWC.  Gather the NHWC side through the
                 * index map so both operands align semantically. */
                uint32_t dn = (uint32_t)op->data.binary.dimensions[0];
                uint32_t dc = (uint32_t)op->data.binary.dimensions[1];
                uint32_t dh = (uint32_t)op->data.binary.dimensions[2];
                uint32_t dw = (uint32_t)op->data.binary.dimensions[3];
                uint32_t n, c, y, x;
                for (n = 0u; n < dn; ++n) {
                    for (c = 0u; c < dc; ++c) {
                        for (y = 0u; y < dh; ++y) {
                            for (x = 0u; x < dw; ++x) {
                                uint64_t canonical_flat = ((uint64_t)(n * dc + c) * dh + y) * dw + x;
                                uint64_t nhwc_flat = ((uint64_t)(n * dh + y) * dw + x) * dc + c;
                                float lhs = op->data.binary.mixed_nhwc == 2u ? input[nhwc_flat] : input[canonical_flat];
                                float rhs = op->data.binary.mixed_nhwc == 1u ? right[nhwc_flat] : right[canonical_flat];
                                switch (op->data.binary.operation) {
                                case LW_OP_ADD: output[canonical_flat] = lhs + rhs; break;
                                case LW_OP_MUL: output[canonical_flat] = lhs * rhs; break;
                                case LW_OP_DIV: output[canonical_flat] = lhs / rhs; break;
                                case LW_OP_SUB: output[canonical_flat] = lhs - rhs; break;
                                default: output[canonical_flat] = powf(lhs, rhs); break;
                                }
                            }
                        }
                    }
                }
                return LW_STATUS_OK;
            }
#if defined(__EMSCRIPTEN__) && defined(LW_WASM_REC_SIMD_KERNELS)
            lw_wasm128_binary_contiguous_f32(binary_operation(op->data.binary.operation),
                input, right, output, op->data.binary.element_count);
#else
            lw_avx2_binary_contiguous_f32(binary_operation(op->data.binary.operation), input, right, output, op->data.binary.element_count);
#endif
            return LW_STATUS_OK;
        }
        if (op->data.binary.broadcast_kind == LW_X64_REC_BROADCAST_RIGHT_CHANNEL) {
            float* channel = op->data.binary.right_constant != NULL ? (float*)(uintptr_t)op->data.binary.right_constant : offset_ptr(instance, op->data.binary.right_offset);
            if (channel == NULL) return LW_STATUS_INVALID_ARGUMENT;
#if defined(__EMSCRIPTEN__) && defined(LW_WASM_REC_SIMD_KERNELS)
            lw_wasm128_binary_channel_nhwc_f32(binary_operation(op->data.binary.operation),
                input, channel, output, op->data.binary.pixels, op->data.binary.channels, 0);
#else
            lw_avx2_binary_channel_f32(binary_operation(op->data.binary.operation), input, channel, output, op->data.binary.pixels, op->data.binary.channels, 0);
#endif
            return LW_STATUS_OK;
        }
        if (op->data.binary.broadcast_kind == LW_X64_REC_BROADCAST_RIGHT_PIXEL_SCALAR) {
            const float* pixel_scalars = op->data.binary.right_constant != NULL
                ? op->data.binary.right_constant
                : offset_ptr(instance, op->data.binary.right_offset);
            uint32_t pixel;
            if (input == NULL || output == NULL || pixel_scalars == NULL) return LW_STATUS_INVALID_ARGUMENT;
            for (pixel = 0u; pixel < op->data.binary.pixels; ++pixel) {
#if defined(__EMSCRIPTEN__) && defined(LW_WASM_REC_SIMD_KERNELS)
                lw_wasm128_binary_scalar_f32(binary_operation(op->data.binary.operation),
                    input + (size_t)pixel * op->data.binary.channels, pixel_scalars[pixel],
                    output + (size_t)pixel * op->data.binary.channels,
                    op->data.binary.channels, 0);
#else
                lw_avx2_binary_right_scalar_f32(binary_operation(op->data.binary.operation),
                    input + (size_t)pixel * op->data.binary.channels, pixel_scalars[pixel],
                    output + (size_t)pixel * op->data.binary.channels, op->data.binary.channels);
#endif
            }
            return LW_STATUS_OK;
        }
        if (op->data.binary.right_constant != NULL && op->data.binary.channels > 1u) {
            uint64_t index;
            const float* right = op->data.binary.right_constant;
            for (index = 0u; index < op->data.binary.element_count; ++index) {
                float lhs = input[index];
                float rhs = right[index % op->data.binary.channels];
                switch (op->data.binary.operation) {
                case LW_OP_ADD: output[index] = lhs + rhs; break;
                case LW_OP_MUL: output[index] = lhs * rhs; break;
                case LW_OP_DIV: output[index] = lhs / rhs; break;
                case LW_OP_SUB: output[index] = lhs - rhs; break;
                default: output[index] = powf(lhs, rhs); break;
                }
            }
            return LW_STATUS_OK;
        }
        return LW_STATUS_UNSUPPORTED;
    case LW_X64_REC_OP_RELU: case LW_X64_REC_OP_ERF: case LW_X64_REC_OP_GELU: case LW_X64_REC_OP_HARD_SIGMOID:
        input = offset_ptr(instance, op->data.unary.input_offset); output = offset_ptr(instance, op->data.unary.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        if (op->kind == LW_X64_REC_OP_RELU) {
#if defined(__EMSCRIPTEN__) && defined(LW_WASM_REC_SIMD_KERNELS)
            lw_wasm128_relu_f32(input, output, op->data.unary.element_count);
#else
            lw_avx2_relu_contiguous_f32(input, output, op->data.unary.element_count);
#endif
        }
        else if (op->kind == LW_X64_REC_OP_ERF) {
#if defined(__EMSCRIPTEN__)
            lw_wasm128_erf_f32(input, output, op->data.unary.element_count);
#else
            lw_avx2_erf_f32(input, output, op->data.unary.element_count);
#endif
        } else if (op->kind == LW_X64_REC_OP_GELU) {
#if defined(__EMSCRIPTEN__)
            lw_wasm128_gelu_f32(input, output, op->data.unary.element_count);
#else
            lw_avx2_gelu_f32(input, output, op->data.unary.element_count);
#endif
        } else {
            lw_avx2_hard_sigmoid_contiguous_f32(input, output,
                op->data.unary.element_count, op->data.unary.alpha, op->data.unary.beta);
        }
        return LW_STATUS_OK;
    case LW_X64_REC_OP_SIGMOID:
        /* Canonical scalar sigmoid: identical expf evaluation order, so
         * results match the interpreter bit for bit. */
        input = offset_ptr(instance, op->data.unary.input_offset); output = offset_ptr(instance, op->data.unary.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        return lw_scalar_sigmoid_f32(input, output, op->data.unary.element_count);
    case LW_X64_REC_OP_SQRT:
        input = offset_ptr(instance, op->data.unary.input_offset); output = offset_ptr(instance, op->data.unary.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        return lw_scalar_sqrt_f32(input, output, op->data.unary.element_count);
    case LW_X64_REC_OP_SLICE:
        input = offset_ptr(instance, op->data.slice.input_offset); output = offset_ptr(instance, op->data.slice.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        return lw_scalar_slice_f32(input, output, op->data.slice.rank,
            op->data.slice.input_dimensions, op->data.slice.output_dimensions,
            op->data.slice.slice_count, op->data.slice.starts, op->data.slice.ends,
            op->data.slice.axes, op->data.slice.steps);
    case LW_X64_REC_OP_CONCAT: {
        /* Channel-axis concat over NHWC rank-4 maps: per-pixel
         * concatenated channel copies (pure data movement). */
        const float* concat_inputs[LWM_V0_MAX_NODE_INPUTS];
        uint32_t concat_slot;
        uint64_t concat_pixel;
        output = offset_ptr(instance, op->data.concat.output_offset);
        if (output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        for (concat_slot = 0u; concat_slot < op->data.concat.input_count; ++concat_slot) {
            concat_inputs[concat_slot] = offset_ptr(instance, op->data.concat.input_offsets[concat_slot]);
            if (concat_inputs[concat_slot] == NULL) return LW_STATUS_INVALID_ARGUMENT;
        }
        for (concat_pixel = 0u; concat_pixel < op->data.concat.pixels; ++concat_pixel) {
            float* destination = output + (size_t)concat_pixel * op->data.concat.output_channels;
            for (concat_slot = 0u; concat_slot < op->data.concat.input_count; ++concat_slot) {
                uint32_t concat_channels = op->data.concat.input_channels[concat_slot];
                memcpy(destination, concat_inputs[concat_slot] + (size_t)concat_pixel * concat_channels,
                       (size_t)concat_channels * sizeof(float));
                destination += concat_channels;
            }
        }
        return LW_STATUS_OK;
    }
    case LW_X64_REC_OP_SOFTMAX:
        input = offset_ptr(instance, op->data.softmax.input_offset); output = offset_ptr(instance, op->data.softmax.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        return lw_scalar_softmax_f32(input, output, op->data.softmax.rank,
                                     op->data.softmax.dimensions, op->data.softmax.axis);
    case LW_X64_REC_OP_REDUCE_MEAN:
        input = offset_ptr(instance, op->data.reduce.input_offset); output = offset_ptr(instance, op->data.reduce.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        if (op->data.reduce.general != 0u) {
            /* Not the global-pool pattern: run the canonical scalar
             * reduce with the original axes, bit-identical to the
             * interpreter (e.g. Small's rank-3 last-axis mean). */
            return lw_scalar_reduce_mean_f32(input, output, op->data.reduce.input_rank,
                op->data.reduce.input_dimensions, op->data.reduce.axes_count,
                op->data.reduce.axes, op->data.reduce.keep_dimensions,
                op->data.reduce.no_op_with_empty_axes, op->data.reduce.output_rank,
                op->data.reduce.output_dimensions);
        }
        if (instance->program->backend_layout == LW_X64_REC_BACKEND_NCHW) {
            for (uint32_t channel = 0u; channel < op->data.reduce.channels; ++channel) {
                const float* source = input + (size_t)channel * op->data.reduce.height *
                                             op->data.reduce.width;
                float sum = 0.0f;
                for (uint32_t spatial = 0u; spatial < op->data.reduce.height *
                                                   op->data.reduce.width; ++spatial) {
                    sum += source[spatial];
                }
                output[channel] = sum / (float)(op->data.reduce.height * op->data.reduce.width);
            }
        } else {
#if defined(__EMSCRIPTEN__) && defined(LW_WASM_REC_SIMD_KERNELS)
            lw_wasm128_reduce_mean_hw_f32(input, output, op->data.reduce.batch,
                op->data.reduce.height, op->data.reduce.width, op->data.reduce.channels);
#else
            lw_avx2_nhwc_reduce_mean_hw_f32(input, output, op->data.reduce.batch,
                                             op->data.reduce.height, op->data.reduce.width,
                                             op->data.reduce.channels);
#endif
        }
        return LW_STATUS_OK;
    case LW_X64_REC_OP_AVG_POOL: case LW_X64_REC_OP_MAX_POOL:
        input = offset_ptr(instance, op->data.pool.input_offset); output = offset_ptr(instance, op->data.pool.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        if (instance->program->backend_layout == LW_X64_REC_BACKEND_NCHW) {
            return op->kind == LW_X64_REC_OP_MAX_POOL
                ? lw_scalar_max_pool2d_f32(input, output, op->data.pool.input_dimensions,
                                           op->data.pool.output_dimensions, op->data.pool.kernel,
                                           op->data.pool.strides, op->data.pool.pads, 0u)
                : lw_scalar_average_pool2d_f32(input, output, op->data.pool.input_dimensions,
                                               op->data.pool.output_dimensions, op->data.pool.kernel,
                                               op->data.pool.strides, op->data.pool.pads, 0u,
                                               op->data.pool.count_include_pad);
        }
#if defined(__EMSCRIPTEN__) && defined(LW_WASM_REC_SIMD_KERNELS)
        lw_wasm128_pool_nhwc_f32(input, output,
#else
        lw_avx2_nhwc_pool_f32(input, output,
#endif
            (uint32_t)op->data.pool.input_dimensions[0],
            (uint32_t)op->data.pool.input_dimensions[1],
            (uint32_t)op->data.pool.input_dimensions[2],
            (uint32_t)op->data.pool.output_dimensions[1],
            (uint32_t)op->data.pool.output_dimensions[2],
            (uint32_t)op->data.pool.input_dimensions[3],
            (uint32_t)op->data.pool.kernel[0], (uint32_t)op->data.pool.kernel[1],
            (uint32_t)op->data.pool.strides[0], (uint32_t)op->data.pool.strides[1],
            (uint32_t)op->data.pool.pads[0], (uint32_t)op->data.pool.pads[1],
            op->data.pool.count_include_pad, op->data.pool.is_max);
        return LW_STATUS_OK;
    case LW_X64_REC_OP_TRANSPOSE: {
        uint32_t rank = op->data.transpose.rank;
        uint32_t axis;
        uint32_t nontrivial[2] = { 0u, 0u };
        uint32_t nontrivial_count = 0u;
        uint64_t strides[4];
        uint64_t stride = 1u;
        int handled = 0;
        input = offset_ptr(instance, op->data.transpose.input_offset); output = offset_ptr(instance, op->data.transpose.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        /* Effective-2D detection: when at most two output axes have extent > 1
         * the transpose degenerates to a matrix transpose (or a plain copy),
         * which the blocked AVX2 kernel handles far faster than the generic
         * per-element scalar walk. Pure data movement, bit-identical. */
        if (rank <= 4u) {
            for (axis = rank; axis > 0u; --axis) {
                strides[axis - 1u] = stride;
                stride *= (uint64_t)(uint32_t)op->data.transpose.input_dimensions[axis - 1u];
            }
            for (axis = 0u; axis < rank; ++axis) {
                int32_t source_axis = op->data.transpose.permutation[axis];
                if (source_axis < 0 || (uint32_t)source_axis >= rank ||
                    op->data.transpose.output_dimensions[axis] !=
                        op->data.transpose.input_dimensions[source_axis]) {
                    nontrivial_count = UINT32_MAX;
                    break;
                }
                if (op->data.transpose.output_dimensions[axis] > 1) {
                    if (nontrivial_count >= 2u) {
                        nontrivial_count = UINT32_MAX;
                        break;
                    }
                    nontrivial[nontrivial_count++] = axis;
                }
            }
            if (nontrivial_count == 2u) {
                uint32_t a = nontrivial[0];
                uint32_t b = nontrivial[1];
                uint64_t rows = (uint64_t)(uint32_t)op->data.transpose.output_dimensions[a];
                uint64_t cols = (uint64_t)(uint32_t)op->data.transpose.output_dimensions[b];
                uint64_t stride_a = strides[(uint32_t)op->data.transpose.permutation[a]];
                uint64_t stride_b = strides[(uint32_t)op->data.transpose.permutation[b]];
                if (stride_a == 1u && stride_b == rows && rows <= UINT32_MAX &&
                    cols <= UINT32_MAX) {
                    /* out[i][j] = in[j][i]: true matrix transpose. */
                    lw_avx2_transpose_2d_f32(input, output, (uint32_t)cols,
                                             (uint32_t)rows);
                    handled = 1;
                } else if (stride_a == cols && stride_b == 1u) {
                    memcpy(output, input, (size_t)(rows * cols) * sizeof(float));
                    handled = 1;
                }
            } else if (nontrivial_count <= 1u) {
                uint64_t elements = stride;
                memcpy(output, input, (size_t)elements * sizeof(float));
                handled = 1;
            }
        }
        if (handled != 0) return LW_STATUS_OK;
        return lw_scalar_transpose_f32(input, output, op->data.transpose.rank, op->data.transpose.input_dimensions, op->data.transpose.rank, op->data.transpose.permutation, op->data.transpose.output_dimensions);
    }
    case LW_X64_REC_OP_MATMUL: {
        const float* right;
        input = offset_ptr(instance, op->data.matmul.input_offset); output = offset_ptr(instance, op->data.matmul.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        if (op->data.matmul.packed_weights != NULL) {
#if defined(__EMSCRIPTEN__) && defined(LW_WASM_REC_SIMD_KERNELS)
            lw_wasm128_packed_matmul_shared_f32(input, op->data.matmul.packed_weights,
                output, op->data.matmul.batch, op->data.matmul.rows,
                op->data.matmul.inner, op->data.matmul.columns);
#else
            lw_avx2_packed_matmul_shared_f32(input, op->data.matmul.packed_weights, output, op->data.matmul.batch, op->data.matmul.rows, op->data.matmul.inner, op->data.matmul.columns);
#endif
            return LW_STATUS_OK;
        }
        right = op->data.matmul.weights != NULL ? op->data.matmul.weights : (const float*)offset_ptr(instance, op->data.matmul.weights_offset);
        if (right == NULL) return LW_STATUS_INVALID_ARGUMENT;
        {
            uint32_t shard_mode = 0u;
            uint32_t shard_workers = matmul_shard_workers(instance, &op->data.matmul,
                                                          &shard_mode);
            if (shard_workers > 1u) {
                lw_rec_matmul_shard shard;
                uint32_t wi;
                lw_status status = LW_STATUS_OK;
                shard.input = input;
                shard.right = right;
                shard.output = output;
                shard.batch = op->data.matmul.batch;
                shard.rows = op->data.matmul.rows;
                shard.inner = op->data.matmul.inner;
                shard.columns = op->data.matmul.columns;
                shard.mode = shard_mode;
                lw_thread_pool_run(instance->thread_pool, shard_workers,
                                   matmul_shard_entry, &shard);
                for (wi = 0u; wi < shard_workers; ++wi) {
                    if (shard.status[wi] != LW_STATUS_OK) {
                        status = shard.status[wi];
                        break;
                    }
                }
                return status;
            }
        }
        if (op->data.matmul.weights_rank == 2u) {
            return lw_matmul_shared_f32(input, right, output, op->data.matmul.batch, op->data.matmul.rows, op->data.matmul.inner, op->data.matmul.columns);
        }
        return lw_scalar_matmul_f32(input, right, output, op->data.matmul.input_rank, op->data.matmul.input_dimensions, op->data.matmul.weights_rank, op->data.matmul.weights_dimensions, op->data.matmul.output_rank, op->data.matmul.output_dimensions); }
    default: return LW_STATUS_UNSUPPORTED;
    }
}

void lw_x64_rec_instance_set_thread_pool(lw_x64_rec_instance* instance,
                                         lw_thread_pool* pool, uint32_t workers) {
    if (instance == NULL) return;
    if (pool == NULL || workers <= 1u) { pool = NULL; workers = 0u; }
    instance->thread_pool = pool;
    instance->intra_op_workers = workers;
}

lw_status lw_x64_rec_instance_create(const lw_x64_rec_program* program, lw_x64_rec_instance** out, lw_error* error) {
    lw_x64_rec_instance* instance;
    if (out == NULL || program == NULL) { lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "REC instance requires a compiled program"); return LW_STATUS_INVALID_ARGUMENT; }
    *out = NULL; instance = (lw_x64_rec_instance*)calloc(1u, sizeof(*instance));
    if (instance == NULL) { lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "REC instance allocation failed"); return LW_STATUS_OUT_OF_MEMORY; }
    instance->program = program;
    instance->owns_workspace = 1u;
    if (program->arena_bytes > 0u) instance->arena = (uint8_t*)rec_aligned_alloc(64u, (size_t)program->arena_bytes);
    if (program->scratch_bytes > 0u) instance->scratch = (uint8_t*)rec_aligned_alloc(64u, (size_t)program->scratch_bytes);
    if ((program->arena_bytes > 0u && instance->arena == NULL) || (program->scratch_bytes > 0u && instance->scratch == NULL)) { lw_x64_rec_instance_free(instance); lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "REC instance workspace allocation failed"); return LW_STATUS_OUT_OF_MEMORY; }
    if (program->ctc.enabled) { uint64_t rows = program->ctc.rows; if (rows == 0u || rows > SIZE_MAX / sizeof(float)) { lw_x64_rec_instance_free(instance); return LW_STATUS_INVALID_SHAPE; } instance->ctc_scores = (float*)rec_aligned_alloc(64u, (size_t)(rows * sizeof(float))); instance->best_indices = (uint32_t*)rec_aligned_alloc(64u, (size_t)(rows * sizeof(uint32_t))); instance->best_probabilities = (float*)rec_aligned_alloc(64u, (size_t)(rows * sizeof(float))); if (instance->ctc_scores == NULL || instance->best_indices == NULL || instance->best_probabilities == NULL) { lw_x64_rec_instance_free(instance); lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "REC CTC workspace allocation failed"); return LW_STATUS_OUT_OF_MEMORY; } }
    *out = instance; lw_set_error(error, LW_STATUS_OK, ""); return LW_STATUS_OK;
}

lw_status lw_x64_rec_instance_create_borrowed(
    const lw_x64_rec_program* program,
    uint8_t* arena, uint64_t arena_bytes,
    uint8_t* scratch, uint64_t scratch_bytes,
    float* ctc_scores, uint32_t* best_indices, float* best_probabilities,
    uint64_t ctc_rows, lw_x64_rec_instance** out, lw_error* error) {
    lw_x64_rec_instance* instance;
    if (out == NULL || program == NULL) { lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "REC instance requires a compiled program"); return LW_STATUS_INVALID_ARGUMENT; }
    if ((program->arena_bytes > 0u && (arena == NULL || arena_bytes < program->arena_bytes)) ||
        (program->scratch_bytes > 0u && (scratch == NULL || scratch_bytes < program->scratch_bytes)) ||
        (program->ctc.enabled && (ctc_rows < program->ctc.rows || ctc_scores == NULL ||
                                  best_indices == NULL || best_probabilities == NULL))) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "REC borrowed workspace is smaller than the program requires");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    *out = NULL; instance = (lw_x64_rec_instance*)calloc(1u, sizeof(*instance));
    if (instance == NULL) { lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "REC instance allocation failed"); return LW_STATUS_OUT_OF_MEMORY; }
    instance->program = program;
    instance->arena = arena;
    instance->scratch = scratch;
    instance->ctc_scores = ctc_scores;
    instance->best_indices = best_indices;
    instance->best_probabilities = best_probabilities;
    instance->owns_workspace = 0u;
    *out = instance; lw_set_error(error, LW_STATUS_OK, ""); return LW_STATUS_OK;
}

void lw_x64_rec_instance_free(lw_x64_rec_instance* instance) { if (instance == NULL) return; rec_aligned_free(instance->dense_shard_scratch); if (instance->owns_workspace != 0u) { rec_aligned_free(instance->ctc_scores); rec_aligned_free(instance->best_indices); rec_aligned_free(instance->best_probabilities); rec_aligned_free(instance->scratch); rec_aligned_free(instance->arena); } free(instance); }

float* lw_x64_rec_instance_input(lw_x64_rec_instance* instance, uint64_t* element_count) { const lw_x64_rec_value* value; if (element_count != NULL) *element_count = 0u; if (instance == NULL || instance->program == NULL || instance->program->input_value >= instance->program->value_count) return NULL; value=&instance->program->values[instance->program->input_value]; if (element_count != NULL) *element_count=value->bytes/sizeof(float); return offset_ptr(instance,value->offset); }

/* Env-gated (LW_X64_REC_PROFILE=1) per-kind op timing for the compiled
 * backend.  Zero overhead when disabled: one cached getenv per process. */
#if defined(_MSC_VER)
#pragma warning(disable : 4996)
#endif
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
static uint64_t rec_profile_now_ns(void) {
    static LARGE_INTEGER freq;
    LARGE_INTEGER now;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    return (uint64_t)((now.QuadPart * 1000000000ull) / (uint64_t)freq.QuadPart);
}
#else
#include <time.h>
static uint64_t rec_profile_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
#endif

static int rec_profile_enabled(void) {
    static int cached = -1;
    if (cached < 0) cached = lw_profile_env_present("LW_X64_REC_PROFILE") ? 1 : 0;
    return cached;
}

#if defined(__EMSCRIPTEN__)
static int rec_op_profile_enabled(void) {
    static int cached = -1;
    if (cached < 0) cached = lw_profile_env_is_one("LW_WASM_REC_OP_PROFILE") ? 1 : 0;
    return cached;
}

static const char* rec_op_kind_name(uint16_t kind) {
    switch (kind) {
    case LW_X64_REC_OP_POINTWISE: return "pointwise";
    case LW_X64_REC_OP_DENSE: return "dense";
    case LW_X64_REC_OP_DEPTHWISE: return "depthwise";
    case LW_X64_REC_OP_AFFINE: return "affine";
    case LW_X64_REC_OP_ADD: return "add";
    case LW_X64_REC_OP_MUL: return "mul";
    case LW_X64_REC_OP_DIV: return "div";
    case LW_X64_REC_OP_SUB: return "sub";
    case LW_X64_REC_OP_POW: return "pow";
    case LW_X64_REC_OP_RELU: return "relu";
    case LW_X64_REC_OP_ERF: return "erf";
    case LW_X64_REC_OP_GELU: return "gelu";
    case LW_X64_REC_OP_HARD_SIGMOID: return "hard_sigmoid";
    case LW_X64_REC_OP_SIGMOID: return "sigmoid";
    case LW_X64_REC_OP_SQRT: return "sqrt";
    case LW_X64_REC_OP_REDUCE_MEAN: return "reduce_mean";
    case LW_X64_REC_OP_AVG_POOL: return "avg_pool";
    case LW_X64_REC_OP_MAX_POOL: return "max_pool";
    case LW_X64_REC_OP_TRANSPOSE: return "transpose";
    case LW_X64_REC_OP_MATMUL: return "matmul";
    case LW_X64_REC_OP_CONCAT: return "concat";
    case LW_X64_REC_OP_RESIZE: return "resize";
    case LW_X64_REC_OP_SOFTMAX: return "softmax";
    case LW_X64_REC_OP_SLICE: return "slice";
    default: return "other";
    }
}

static void rec_op_profile_record(const lw_x64_rec_program* program, uint32_t index,
                                    uint64_t invocation, uint64_t elapsed,
                                    lw_status status) {
    const lw_x64_rec_op* op = &program->ops[index];
    fprintf(stderr, "WASM_REC_OP width=%u invocation=%llu index=%u semantic=%u "
        "span=%u kind=%s elapsed=%.6f status=%d",
        program->target_width, (unsigned long long)invocation, index,
        op->semantic_begin, (unsigned)op->semantic_count, rec_op_kind_name(op->kind),
        elapsed / 1e6, (int)status);
    if (op->kind == LW_X64_REC_OP_POINTWISE || op->kind == LW_X64_REC_OP_DENSE ||
        op->kind == LW_X64_REC_OP_DEPTHWISE) {
        const lw_x64_rec_conv_op* conv = &op->data.conv;
        fprintf(stderr, " ic=%u oc=%u ih=%u iw=%u oh=%u ow=%u kh=%u kw=%u "
            "sh=%u sw=%u groups=%u act=%u residual=%u post_bias=%u input=%llu output=%llu",
            conv->input_channels, conv->output_channels, conv->input_height, conv->input_width,
            conv->output_height, conv->output_width, conv->kernel_h, conv->kernel_w,
            conv->stride_h, conv->stride_w, conv->groups, (unsigned)conv->activation,
            (unsigned)conv->has_residual, conv->post_bias != NULL ? 1u : 0u,
            (unsigned long long)conv->input_offset, (unsigned long long)conv->output_offset);
    }
    fputc('\n', stderr);
}
#endif

static uint64_t* rec_profile_slot(lw_x64_rec_profile* profile, uint16_t kind) {
    switch (kind) {
    case LW_X64_REC_OP_POINTWISE: return &profile->pointwise_ns;
    case LW_X64_REC_OP_DENSE: return &profile->dense_ns;
    case LW_X64_REC_OP_DEPTHWISE: return &profile->depthwise_ns;
    case LW_X64_REC_OP_ADD: case LW_X64_REC_OP_MUL: case LW_X64_REC_OP_DIV:
    case LW_X64_REC_OP_SUB: case LW_X64_REC_OP_POW: return &profile->binary_ns;
    case LW_X64_REC_OP_RELU: case LW_X64_REC_OP_ERF: case LW_X64_REC_OP_GELU:
    case LW_X64_REC_OP_HARD_SIGMOID: case LW_X64_REC_OP_SIGMOID:
    case LW_X64_REC_OP_SQRT: return &profile->unary_ns;
    case LW_X64_REC_OP_REDUCE_MEAN: return &profile->reduce_ns;
    case LW_X64_REC_OP_AVG_POOL: case LW_X64_REC_OP_MAX_POOL: return &profile->pool_ns;
    case LW_X64_REC_OP_TRANSPOSE: return &profile->transpose_ns;
    case LW_X64_REC_OP_MATMUL: return &profile->matmul_ns;
    default: return NULL;
    }
}

enum rec_pw_epi_kind {
    REC_PW_PLAIN = 0,
    REC_PW_BIAS,
    REC_PW_HARDSWISH,
    REC_PW_RESIDUAL,
    REC_PW_RESIDUAL_GELU,
    REC_PW_POSTBIAS_GELU,
    REC_PW_OTHER,
    REC_PW_EPI_COUNT
};

static uint32_t classify_rec_pointwise(const lw_x64_rec_conv_op* conv) {
    if (conv->post_bias == NULL && conv->has_residual == 0u) {
        if (conv->activation == LW_NHWC_ACT_NONE)
            return conv->bias == NULL ? REC_PW_PLAIN : REC_PW_BIAS;
        if (conv->activation == LW_NHWC_ACT_HARDSWISH) return REC_PW_HARDSWISH;
    }
    if (conv->post_bias == NULL && conv->has_residual != 0u &&
        conv->activation == LW_NHWC_ACT_NONE) return REC_PW_RESIDUAL;
    if (conv->has_residual != 0u && conv->activation == LW_NHWC_ACT_GELU)
        return REC_PW_RESIDUAL_GELU;
    if (conv->post_bias != NULL && conv->activation == LW_NHWC_ACT_GELU)
        return REC_PW_POSTBIAS_GELU;
    return REC_PW_OTHER;
}

static void record_rec_pointwise_epilogue(lw_x64_rec_profile* profile,
                                           const lw_x64_rec_op* op,
                                           uint64_t elapsed) {
    const lw_x64_rec_conv_op* conv = &op->data.conv;
    uint32_t kind = classify_rec_pointwise(conv);
    uint64_t pixels = (uint64_t)conv->input_height * conv->input_width;
    profile->pw_epi_ns[kind] += elapsed;
    ++profile->pw_epi_calls[kind];
    profile->pw_epi_macs[kind] += pixels * conv->input_channels * conv->output_channels;
}

static lw_status run_backbone_ops(lw_x64_rec_instance* instance, lw_error* error) {
    uint32_t i;
    lw_status status;
    int profiling;
#if defined(__EMSCRIPTEN__)
    static uint64_t next_invocation = 0u; /* Single-threaded diagnostic runs only. */
    uint64_t invocation = 0u;
    int detailed = 0;
#endif
    if (instance == NULL || instance->program == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "REC instance is invalid");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    profiling = rec_profile_enabled();
#if defined(__EMSCRIPTEN__)
    /* CLS shares this executor but has no CTC tail: never pool it into REC. */
    detailed = rec_op_profile_enabled() && instance->program->ctc.enabled != 0u;
    if (detailed) {
        invocation = ++next_invocation;
        fprintf(stderr, "WASM_REC_OP_BEGIN width=%u invocation=%llu ops=%u\n",
            instance->program->target_width, (unsigned long long)invocation,
            instance->program->op_count);
        profiling = 1;
    }
#endif
    if (profiling) {
        memset(&instance->profile, 0, sizeof(instance->profile));
    }
    for (i = 0u; i < instance->program->op_count; ++i) {
        int paired = 0;
#if defined(LW_EXPERIMENTAL_REC_FFN_TILING)
        paired = (instance->program->ops[i].flags & LW_X64_REC_OP_FFN_BEGIN) != 0u;
#endif
#if defined(__EMSCRIPTEN__)
        /* Detailed one-op traces retain the unfused diagnostic execution. */
        if (detailed) paired = 0;
#endif
        if (profiling) {
            uint64_t started = rec_profile_now_ns();
            uint64_t* slot;
            status = paired ? lw_x64_rec_instance_run_ffn_pair(instance, i, error)
                            : execute_op(instance, &instance->program->ops[i], error);
            slot = rec_profile_slot(&instance->profile, instance->program->ops[i].kind);
            {
                uint64_t finished = rec_profile_now_ns();
                uint64_t elapsed = finished >= started ? finished - started : 0u;
                if (slot != NULL) *slot += elapsed;
                instance->profile.total_ns += elapsed;
                if (paired) {
                    instance->profile.ffn_ns += elapsed;
                    ++instance->profile.ffn_calls;
                } else if (instance->program->ops[i].kind == LW_X64_REC_OP_POINTWISE) {
                    record_rec_pointwise_epilogue(&instance->profile,
                                                   &instance->program->ops[i], elapsed);
                }
#if defined(__EMSCRIPTEN__)
                if (detailed) rec_op_profile_record(instance->program, i, invocation, elapsed, status);
#endif
            }
        } else {
            status = paired ? lw_x64_rec_instance_run_ffn_pair(instance, i, error)
                            : execute_op(instance, &instance->program->ops[i], error);
        }
        if (status != LW_STATUS_OK) {
            if (error != NULL && error->message[0] == '\0') {
                char message[96];
                (void)snprintf(message, sizeof(message), "REC physical op %u kind %u failed", (unsigned)i, (unsigned)instance->program->ops[i].kind);
                lw_set_error(error, status, message);
            }
            return status;
        }
        if (paired) ++i;
    }
#if defined(__EMSCRIPTEN__)
    if (detailed) fprintf(stderr, "WASM_REC_OP_END width=%u invocation=%llu\n",
        instance->program->target_width, (unsigned long long)invocation);
#endif
    return LW_STATUS_OK;
}

lw_status lw_x64_rec_instance_run_op(lw_x64_rec_instance* instance, uint32_t op_index, lw_error* error) {
    if (instance == NULL || instance->program == NULL || op_index >= instance->program->op_count) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "REC physical op index is invalid");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    return execute_op(instance, &instance->program->ops[op_index], error);
}
lw_status lw_x64_rec_instance_run_backbone(lw_x64_rec_instance* instance, lw_error* error) {
    lw_status status = run_backbone_ops(instance, error);
    if (status == LW_STATUS_OK) lw_set_error(error, LW_STATUS_OK, "");
    return status;
}

lw_status lw_x64_rec_instance_run(lw_x64_rec_instance* instance, lw_error* error) {
    lw_status status;
    float* activation;
    int profiling = rec_profile_enabled();
    status = run_backbone_ops(instance, error);
    if (status != LW_STATUS_OK) return status;
    if (instance->program->ctc.enabled) {
        uint64_t ctc_started = profiling ? rec_profile_now_ns() : 0u;
        activation = offset_ptr(instance, instance->program->values[instance->program->ctc.activation_value].offset);
        status = lw_x64_rec_ctc_execute(instance->program, instance, activation, error);
        if (status != LW_STATUS_OK) return status;
        if (profiling) {
            instance->profile.ctc_ns = rec_profile_now_ns() - ctc_started;
        }
    }
    if (profiling) {
        const lw_x64_rec_profile* p = &instance->profile;
        fprintf(stderr,
            "X64REC width=%u total=%.3f pw=%.3f dense=%.3f dw=%.3f bin=%.3f unary=%.3f "
            "reduce=%.3f pool=%.3f transpose=%.3f matmul=%.3f ctc=%.3f\n",
            (unsigned)instance->program->target_width, p->total_ns / 1e6,
            p->pointwise_ns / 1e6, p->dense_ns / 1e6, p->depthwise_ns / 1e6,
            p->binary_ns / 1e6, p->unary_ns / 1e6, p->reduce_ns / 1e6,
            p->pool_ns / 1e6, p->transpose_ns / 1e6, p->matmul_ns / 1e6,
            p->ctc_ns / 1e6);
        if (p->ffn_calls != 0u)
            fprintf(stderr, "REC_FFN width=%u pairs=%u elapsed=%.3f tile=%u\n",
                instance->program->target_width, p->ffn_calls, p->ffn_ns / 1e6,
                LW_REC_FFN_TILE_PIXELS);
#if defined(__EMSCRIPTEN__)
        {
            static const char* const names[REC_PW_EPI_COUNT] = {
                "plain", "bias", "hardswish", "residual", "resgelu", "postgelu", "other"
            };
            uint32_t kind;
            for (kind = 0u; kind < REC_PW_EPI_COUNT; ++kind) {
                if (p->pw_epi_calls[kind] == 0u) continue;
                fprintf(stderr, "WASM_PW_EPI width=%u kind=%s calls=%llu macs=%llu elapsed=%.3f\n",
                        (unsigned)instance->program->target_width, names[kind],
                        (unsigned long long)p->pw_epi_calls[kind],
                        (unsigned long long)p->pw_epi_macs[kind],
                        p->pw_epi_ns[kind] / 1e6);
            }
        }
#endif
    }
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}
