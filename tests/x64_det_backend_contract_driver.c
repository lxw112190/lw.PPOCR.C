/* Contract for the standalone x64 DET backend: both strategies must reproduce
 * the canonical executor's probability map within 1e-3/1e-5 and be
 * bit-deterministic across runs. Also validates the profile mapping the
 * detector hookup relies on. */

#include "x64_det_backend_internal.h"
#include "lwm_read.h"
#include "model_internal.h"
#include "parallel_internal.h"
#include "session_internal.h"
#include "det_internal.h"

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <time.h>
#endif

static uint64_t clock_ns(void) {
#if defined(_WIN32)
    LARGE_INTEGER c, f;
    QueryPerformanceCounter(&c);
    QueryPerformanceFrequency(&f);
    return (uint64_t)((double)c.QuadPart * 1000000000.0 / (double)f.QuadPart);
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
#endif
}

static uint64_t profile_clock(void* context) {
    (void)context;
    return clock_ns();
}

static void fill_input(float* input, uint64_t count) {
    uint32_t state = 0x9e3779b9u;
    uint64_t i;
    for (i = 0u; i < count; ++i) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        input[i] = ((float)(state & 0xffffffu) / (float)0x1000000u) * 2.0f - 1.0f;
    }
}

static int compare_probabilities(const float* reference, const float* actual,
                                 uint64_t count, double* out_max_abs) {
    uint64_t i;
    uint64_t violations = 0u;
    double max_abs = 0.0;
    uint64_t worst_index[4] = {0u, 0u, 0u, 0u};
    double worst_diff[4] = {0.0, 0.0, 0.0, 0.0};
    for (i = 0u; i < count; ++i) {
        double diff = fabs((double)reference[i] - (double)actual[i]);
        double tolerance = 1.0e-3 * fabs((double)reference[i]) + 1.0e-5;
        uint32_t slot;
        if (diff > max_abs) max_abs = diff;
        if (diff > tolerance) ++violations;
        for (slot = 0u; slot < 4u; ++slot) {
            if (diff > worst_diff[slot]) {
                uint32_t tail;
                for (tail = 3u; tail > slot; --tail) {
                    worst_index[tail] = worst_index[tail - 1u];
                    worst_diff[tail] = worst_diff[tail - 1u];
                }
                worst_index[slot] = i;
                worst_diff[slot] = diff;
                break;
            }
        }
    }
    if (out_max_abs != NULL) *out_max_abs = max_abs;
    if (violations != 0u) {
        for (i = 0u; i < 4u; ++i) {
            fprintf(stderr, "worst mismatch at %llu: ref %.9g got %.9g (diff %.9g)\n",
                    (unsigned long long)worst_index[i], reference[worst_index[i]],
                    actual[worst_index[i]], worst_diff[i]);
        }
        fprintf(stderr, "%llu probability violations (max abs diff %.9g)\n",
                (unsigned long long)violations, max_abs);
        return 0;
    }
    return 1;
}

/* Force one shardable kernel to reject its packed weights. Every worker must
 * then recompute its own global output rows through the scalar fallback. */
static int check_sharded_scalar_fallback(lw_x64_det_program* program,
                                         lw_x64_det_instance* instance,
                                         const float* source_input, uint64_t input_count,
                                         uint16_t kind) {
    lw_thread_pool* pool = NULL;
    float* expected = NULL;
    const float* saved_weights = NULL;
    lw_x64_det_op* target = NULL;
    lw_error error;
    uint32_t target_index = 0u;
    uint32_t op_index;
    uint64_t output_count;
    uint64_t input_slot_count = 0u;
    float* input_slot;
    int ok = 0;
    for (op_index = 0u; op_index < program->op_count; ++op_index) {
        lw_x64_det_op* candidate = &program->ops[op_index];
        const lw_x64_det_conv_op* conv = &candidate->data.conv;
        uint64_t macs;
        if (candidate->kind != kind || conv->output_height < 4u) continue;
        macs = (uint64_t)conv->output_channels * conv->input_channels *
               conv->kernel_h * conv->kernel_w *
               conv->output_height * conv->output_width;
        if (kind == LW_X64_DET_OP_DEPTHWISE && conv->input_channels != 0u) {
            macs /= conv->input_channels;
        }
        if (macs >= (kind == LW_X64_DET_OP_DEPTHWISE
                         ? UINT64_C(1000000) : UINT64_C(2000000))) {
            target = candidate;
            target_index = op_index;
            break;
        }
    }
    if (target == NULL) {
        fprintf(stderr, "no shardable DET op for forced scalar fallback kind %u\n",
                (unsigned)kind);
        return 0;
    }
    input_slot = lw_x64_det_instance_input(instance, &input_slot_count);
    if (input_slot == NULL || input_slot_count != input_count) return 0;
    memcpy(input_slot, source_input, (size_t)input_count * sizeof(float));
    lw_error_init(&error);
    for (op_index = 0u; op_index <= target_index; ++op_index) {
        if (lw_x64_det_instance_run_op(instance, op_index, &error) != LW_STATUS_OK) {
            fprintf(stderr, "DET fallback setup failed at op %u: %s\n", op_index, error.message);
            return 0;
        }
    }
    output_count = (uint64_t)target->data.conv.output_channels *
                   target->data.conv.output_height * target->data.conv.output_width;
    if (output_count == 0u || output_count > SIZE_MAX / sizeof(float)) return 0;
    expected = (float*)malloc((size_t)output_count * sizeof(float));
    pool = lw_thread_pool_create(4u);
    if (expected == NULL || pool == NULL) goto cleanup;
    memcpy(expected, instance->arena + (size_t)target->data.conv.output_offset,
           (size_t)output_count * sizeof(float));
    saved_weights = target->data.conv.packed_weights;
    target->data.conv.packed_weights = NULL;
    if (lw_x64_det_instance_run_op_ex(instance, target_index, pool, 4u, &error) !=
        LW_STATUS_OK) {
        fprintf(stderr, "DET forced scalar fallback failed: %s\n", error.message);
        goto cleanup;
    }
    ok = compare_probabilities(expected,
        (const float*)(const void*)(instance->arena + (size_t)target->data.conv.output_offset),
        output_count, NULL);
    if (!ok) fprintf(stderr, "DET forced scalar fallback mismatch at op %u\n", target_index);
cleanup:
    if (saved_weights != NULL) target->data.conv.packed_weights = saved_weights;
    lw_thread_pool_free(pool);
    free(expected);
    return ok;
}

/* Contract check: run both arms op-by-op, snapshotting every produced tensor
 * the moment its producer completes (arena slots are reused later, so
 * end-of-run reads are meaningless), and compare tensor-by-tensor. The NCHW
 * arm shares the walk/convert/arena machinery with the NHWC arm but none of
 * the NHWC kernels, so any divergence localizes to the first differing node. */
static int check_per_tensor_parity(const lw_x64_det_program* nchw_program,
                                   const lw_x64_det_instance* nchw,
                                   const lw_x64_det_program* nhwc_program,
                                   const lw_x64_det_instance* nhwc) {
    int result = 1;
    float** snapshots = (float**)calloc(nchw_program->value_count, sizeof(float*));
    lw_thread_pool* shard_pool = lw_thread_pool_create(4u);
    uint32_t pass;
    if (snapshots == NULL) {
        lw_thread_pool_free(shard_pool);
        return 0;
    }
    for (pass = 0u; pass < 3u; ++pass) {
        const lw_x64_det_program* program = pass == 0u ? nchw_program : nhwc_program;
        const lw_x64_det_instance* instance = pass == 0u ? nchw : nhwc;
        const int sharded = pass == 2u;
        uint32_t op;
        lw_error error;
        for (op = 0u; op < program->op_count; ++op) {
            const lw_x64_det_op* physical = &program->ops[op];
            uint32_t offset;
            lw_error_init(&error);
            if (sharded) {
                if (lw_x64_det_instance_run_op_ex((lw_x64_det_instance*)(uintptr_t)instance,
                                                  op, shard_pool, 16u, &error) != LW_STATUS_OK) {
                    fprintf(stderr, "sharded parity replay failed at op %u: %s\n", op,
                            error.message);
                    result = 0;
                    goto done;
                }
            } else if (lw_x64_det_instance_run_op((lw_x64_det_instance*)(uintptr_t)instance,
                                                  op, &error) != LW_STATUS_OK) {
                fprintf(stderr, "parity replay failed: %s\n", error.message);
                result = 0;
                goto done;
            }
            if (pass != 0u && physical->kind == LW_X64_DET_OP_LAYOUT_CONVERT) {
                const lw_x64_det_convert_op* convert = &physical->data.convert;
                const float* source = (const float*)(const void*)(instance->arena +
                    (size_t)convert->input_offset);
                const float* actual = (const float*)(const void*)(instance->arena +
                    (size_t)convert->output_offset);
                uint64_t count = (uint64_t)convert->batch * convert->channels *
                    convert->height * convert->width;
                float* expected = (float*)malloc((size_t)count * sizeof(float));
                double max_abs = 0.0;
                uint64_t element;
                if (expected == NULL) {
                    result = 0;
                    goto done;
                }
                if (convert->to_nhwc) {
                    lw_x64_fast_nchw_to_nhwc(source, expected, convert->batch,
                                             convert->channels, convert->height,
                                             convert->width);
                } else {
                    lw_x64_fast_nhwc_to_nchw(source, expected, convert->batch,
                                             convert->channels, convert->height,
                                             convert->width);
                }
                for (element = 0u; element < count; ++element) {
                    double diff = fabs((double)expected[element] - (double)actual[element]);
                    if (diff > max_abs) max_abs = diff;
                }
                free(expected);
                if (max_abs > 1.0e-4) {
                    fprintf(stderr, "convert op %u (to_nhwc=%u) produced wrong output: max_abs %.9g\n",
                            op, convert->to_nhwc, max_abs);
                    result = 0;
                    goto done;
                }
            }
            for (offset = 0u; offset < physical->semantic_count; ++offset) {
                uint32_t node_index = physical->semantic_begin + offset;
                const uint8_t* node = nchw_program->model->bytes +
                    (size_t)nchw_program->model->node_offset +
                    (size_t)node_index * LWM_V0_NODE_SIZE;
                uint32_t tensor_index = lwm_read_u32(node + 40u);
                const lw_x64_det_value* value;
                if (tensor_index >= program->value_count) continue;
                value = &program->values[tensor_index];
                if (value->producer < 0 || value->bytes == 0u || value->constant_data != NULL) {
                    continue;
                }
                if (pass == 0u) {
                    snapshots[tensor_index] = (float*)malloc((size_t)value->bytes);
                    if (snapshots[tensor_index] == NULL) {
                        result = 0;
                        goto done;
                    }
                    memcpy(snapshots[tensor_index],
                           instance->arena + (size_t)value->offset, (size_t)value->bytes);
                } else if (snapshots[tensor_index] != NULL) {
                    float* converted = NULL;
                    const float* source =
                        (const float*)(const void*)(instance->arena + (size_t)value->offset);
                    uint64_t elements = value->bytes / sizeof(float);
                    uint64_t element;
                    double max_abs = 0.0;
                    if (value->layout == LW_X64_DET_LAYOUT_NHWC) {
                        converted = (float*)malloc((size_t)value->bytes);
                        if (converted == NULL) {
                            result = 0;
                            goto done;
                        }
                        lw_x64_fast_nhwc_to_nchw(source, converted,
                            (uint32_t)nchw_program->values[tensor_index].dimensions[0],
                            (uint32_t)nchw_program->values[tensor_index].dimensions[1],
                            (uint32_t)nchw_program->values[tensor_index].dimensions[2],
                            (uint32_t)nchw_program->values[tensor_index].dimensions[3]);
                        source = converted;
                    }
                    for (element = 0u; element < elements; ++element) {
                        double diff = fabs((double)snapshots[tensor_index][element] -
                                           (double)source[element]);
                        if (diff > max_abs) max_abs = diff;
                    }
                    free(converted);
                    /* The NHWC arm accumulates in FMA while the NCHW arm uses
                     * non-FMA kernels; 1e-3 tolerates that noise while staying
                     * four orders of magnitude below real divergence (which
                     * showed up at O(1)-O(100)). */
                    if (max_abs > 1.0e-3) {
                        uint64_t mismatch_element;
                        uint64_t printed = 0u;
                        fprintf(stderr, "%sarm divergence: tensor %u (node %u, op %u, kind %u) max_abs %.9g\n",
                                sharded ? "sharded " : "", tensor_index, node_index, op,
                                physical->kind, max_abs);
                        for (mismatch_element = 0u; mismatch_element < elements && printed < 4u;
                             ++mismatch_element) {
                            double diff = fabs((double)snapshots[tensor_index][mismatch_element] -
                                               (double)source[mismatch_element]);
                            if (diff > 1.0e-3) {
                                fprintf(stderr, "  at %llu: nchw %.9g nhwc %.9g\n",
                                        (unsigned long long)mismatch_element,
                                        snapshots[tensor_index][mismatch_element],
                                        source[mismatch_element]);
                                ++printed;
                            }
                        }
                        result = 0;
                        goto done;
                    }
                }
            }
        }
    }
done:
    lw_thread_pool_free(shard_pool);
    {
        uint32_t t;
        for (t = 0u; t < nchw_program->value_count; ++t) free(snapshots[t]);
    }
    free(snapshots);
    return result;
}

/* Contract check: the arena interval placement must never overlap two
 * values with overlapping live ranges (primary or alternate slots). */
static int check_arena_overlaps(const lw_x64_det_program* program) {
    uint32_t a;
    uint32_t b;
    for (a = 0u; a < program->value_count; ++a) {
        const lw_x64_det_value* va = &program->values[a];
        int32_t af;
        int32_t al;
        if (va->alias != 0u || va->constant_data != NULL) continue;
        if ((int32_t)a == (int32_t)program->input_value) {
            af = 0;
            al = (int32_t)program->op_count;
        } else if (va->producer >= 0) {
            af = va->producer;
            al = va->last_use > va->producer ? va->last_use : va->producer;
        } else {
            continue;
        }
        for (b = 0u; b < program->value_count; ++b) {
            const lw_x64_det_value* vb = &program->values[b];
            int32_t bf;
            int32_t bl;
            if (vb->alias != 0u || vb->constant_data != NULL) continue;
            if ((int32_t)b == (int32_t)program->input_value) {
                bf = 0;
                bl = (int32_t)program->op_count;
            } else if (vb->producer >= 0) {
                bf = vb->producer;
                bl = vb->last_use > vb->producer ? vb->last_use : vb->producer;
            } else {
                continue;
            }
            {
                int32_t alt_b;
                int found = 0;
                for (alt_b = 0; alt_b < 2; ++alt_b) {
                    int32_t first = alt_b ? vb->alt_producer : bf;
                    int32_t last = alt_b ? vb->alt_last_use : bl;
                    uint64_t offset = alt_b ? vb->alt_offset : vb->offset;
                    if (alt_b && vb->alt_producer < 0) continue;
                    if (first > af || last < af || (alt_b == 0 && a == b && first == af && last == al && offset == va->offset)) {
                        /* non-overlapping or self */
                    } else if (first <= al && last >= af) {
                        /* overlapping lifetimes: offsets must be disjoint */
                        if (va->offset < offset + vb->bytes && offset < va->offset + va->bytes) {
                            fprintf(stderr, "arena overlap: value %u [%d,%d]@%llu vs value %u%s [%d,%d]@%llu\n",
                                    a, af, al, (unsigned long long)va->offset, b,
                                    alt_b ? "-alt" : "", first, last, (unsigned long long)offset);
                            found = 1;
                            break;
                        }
                    }
                }
                if (found) return 0;
            }
        }
    }
    return 1;
}

static int run_canonical(lw_session* session, const float* input, uint64_t input_count,
                         float* output, uint64_t output_count) {
    lw_error error;
    lw_error_init(&error);
    if (lw_execute_session_f32(session, input, input_count, output, output_count, &error) !=
        LW_STATUS_OK) {
        fprintf(stderr, "canonical run failed: %s\n", error.message);
        return 0;
    }
    return 1;
}

/* Execute the compiled program through the per-op test hook, validating the
 * run_op path and byte-determinism. */
static int run_backend_op_by_op(lw_x64_det_instance* instance, float* output,
                                uint64_t output_count, const lw_x64_det_program* program) {
    lw_error error;
    lw_status status;
    uint32_t i;
    lw_error_init(&error);
    for (i = 0u; i < program->op_count; ++i) {
        const lw_x64_det_op* op = &program->ops[i];
        status = lw_x64_det_instance_run_op(instance, i, &error);
        if (status != LW_STATUS_OK) {
            fprintf(stderr, "run_op %u (kind %u) failed: %s\n", (unsigned)i,
                    (unsigned)op->kind, error.message);
            return 0;
        }
    }
    {
        const lw_x64_det_value* value = &program->values[program->output_value];
        const float* source = value->layout == LW_X64_DET_LAYOUT_NCHW || value->alt_producer < 0
            ? (const float*)(const void*)(instance->arena + (size_t)value->offset)
            : (const float*)(const void*)(instance->arena + (size_t)value->alt_offset);
        if (output_count < value->bytes / sizeof(float)) {
            fprintf(stderr, "output buffer too small\n");
            return 0;
        }
        memcpy(output, source, (size_t)value->bytes);
    }
    return 1;
}

static int check_direct_input(const lw_model* model, uint32_t height, uint32_t width,
                              const float* nchw_input, const float* expected,
                              float* actual, lw_error* error) {
    lw_x64_det_program* program = NULL;
    lw_x64_det_instance* instance = NULL;
    uint64_t input_count;
    float* input;
    int ok = 0;
    const uint64_t plane = (uint64_t)height * width;
    if (lw_x64_det_backend_compile_ex(model, height, width, LW_X64_DET_COMPILE_NHWC_DIRECT,
                                      &program, error) != LW_X64_DET_COMPILE_OK ||
        program->direct_nhwc == 0u ||
        lw_x64_det_instance_create(program, &instance, error) != LW_STATUS_OK) {
        fprintf(stderr, "direct NHWC DET setup failed: %s\n", error->message);
        goto cleanup;
    }
    input = lw_x64_det_instance_input(instance, &input_count);
    if (input == NULL || input_count != plane * 3u) {
        fprintf(stderr, "direct NHWC DET input slot is invalid\n");
        goto cleanup;
    }
    lw_x64_fast_nchw_to_nhwc(nchw_input, input, 1u, 3u, height, width);
    if (lw_x64_det_instance_run_profiled(instance, actual, plane, NULL, error) != LW_STATUS_OK ||
        memcmp(expected, actual, (size_t)plane * sizeof(float)) != 0) {
        fprintf(stderr, "direct NHWC DET differs from the converted input: %s\n",
                error->message);
        goto cleanup;
    }
    ok = 1;
cleanup:
    lw_x64_det_instance_free(instance);
    lw_x64_det_program_free(program);
    return ok;
}

int main(int argc, char** argv) {
    const uint32_t default_height = 32u;
    const uint32_t default_width = 64u;
    uint32_t height = default_height;
    uint32_t width = default_width;
    uint64_t plane;
    uint64_t input_count;
    lw_model* model = NULL;
    lw_x64_det_program* nhwc_program = NULL;
    lw_x64_det_program* nchw_program = NULL;
    lw_x64_det_instance* nhwc = NULL;
    lw_x64_det_instance* nchw = NULL;
    lw_session* canonical = NULL;
    lw_tensor_desc desc;
    lw_error error;
    float* source_input = NULL;
    float* canonical_output = NULL;
    float* backend_output = NULL;
    float* determinism_output = NULL;
    double nhwc_max_abs = 0.0;
    double nchw_max_abs = 0.0;
    int code = 1;
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "usage: %s det.lwm [height] [width]\n", argv[0]);
        return 2;
    }
    if (argc >= 3) {
        height = (uint32_t)strtoul(argv[2], NULL, 10);
        if (height == 0u) return 2;
    }
    if (argc >= 4) {
        width = (uint32_t)strtoul(argv[3], NULL, 10);
        if (width == 0u) return 2;
    }
    plane = (uint64_t)height * width;
    input_count = plane * 3u;
    lw_error_init(&error);
    if (lw_model_load(argv[1], NULL, &model, &error) != LW_STATUS_OK) {
        fprintf(stderr, "model load failed: %s\n", error.message);
        goto cleanup;
    }
    if (lw_x64_det_backend_compile_ex(model, height, width, LW_X64_DET_COMPILE_NHWC,
                                      &nhwc_program, &error) != LW_X64_DET_COMPILE_OK ||
        lw_x64_det_backend_compile_ex(model, height, width, LW_X64_DET_COMPILE_NCHW,
                                      &nchw_program, &error) != LW_X64_DET_COMPILE_OK) {
        fprintf(stderr, "backend compile failed: %s\n", error.message);
        goto cleanup;
    }
    if (lw_x64_det_instance_create(nhwc_program, &nhwc, &error) != LW_STATUS_OK ||
        lw_x64_det_instance_create(nchw_program, &nchw, &error) != LW_STATUS_OK) {
        fprintf(stderr, "instance create failed: %s\n", error.message);
        goto cleanup;
    }
    lw_tensor_desc_init(&desc);
    desc.dtype = LW_DTYPE_F32;
    desc.rank = 4u;
    desc.dimensions[0] = 1;
    desc.dimensions[1] = 3;
    desc.dimensions[2] = (int32_t)height;
    desc.dimensions[3] = (int32_t)width;
    if (lw_session_create(model, &desc, 1u, NULL, &canonical, &error) != LW_STATUS_OK) {
        fprintf(stderr, "canonical create failed: %s\n", error.message);
        goto cleanup;
    }
    source_input = (float*)malloc((size_t)input_count * sizeof(float));
    canonical_output = (float*)malloc((size_t)plane * sizeof(float));
    backend_output = (float*)malloc((size_t)plane * sizeof(float));
    determinism_output = (float*)malloc((size_t)plane * sizeof(float));
    if (source_input == NULL || canonical_output == NULL || backend_output == NULL ||
        determinism_output == NULL) {
        fprintf(stderr, "allocation failed\n");
        goto cleanup;
    }
    fill_input(source_input, input_count);
    if (!run_canonical(canonical, source_input, input_count, canonical_output, plane)) {
        goto cleanup;
    }
    /* Both backends read NCHW graph input (the effective-layout walk converts
     * for the NHWC arm), so the same buffer feeds all three paths. */
    {
        uint64_t count = 0u;
        float* nhwc_input = lw_x64_det_instance_input(nhwc, &count);
        float* nchw_input = lw_x64_det_instance_input(nchw, &count);
        if (nhwc_input == NULL || nchw_input == NULL || count != input_count) {
            fprintf(stderr, "backend input slot is invalid\n");
            goto cleanup;
        }
        memcpy(nhwc_input, source_input, (size_t)input_count * sizeof(float));
        memcpy(nchw_input, source_input, (size_t)input_count * sizeof(float));
    }
    {
        /* NCHW arm first: it shares the walk/convert/arena machinery with the
         * NHWC arm but none of the NHWC kernels, isolating machinery bugs. */
        if (!check_arena_overlaps(nchw_program)) {
            fprintf(stderr, "NCHW arena overlaps\n");
            goto cleanup;
        }
        if (!check_arena_overlaps(nhwc_program)) {
            fprintf(stderr, "NHWC arena overlaps\n");
            goto cleanup;
        }
        if (!run_backend_op_by_op(nchw, backend_output, plane, nchw_program)) {
            goto cleanup;
        }
        if (!compare_probabilities(canonical_output, backend_output, plane, &nchw_max_abs)) {
            fprintf(stderr, "NCHW backend probability map mismatch\n");
            goto cleanup;
        }
        if (!run_backend_op_by_op(nhwc, determinism_output, plane, nhwc_program)) {
            goto cleanup;
        }
        if (!check_per_tensor_parity(nchw_program, nchw, nhwc_program, nhwc)) {
            fprintf(stderr, "NHWC/NCHW arm parity check failed\n");
            goto cleanup;
        }
        if (!run_backend_op_by_op(nchw, determinism_output, plane, nchw_program)) {
            goto cleanup;
        }
        if (memcmp(backend_output, determinism_output, (size_t)plane * sizeof(float)) != 0) {
            fprintf(stderr, "NCHW backend is not bit-deterministic\n");
            goto cleanup;
        }
    }
    {
        lw_execution_profile profile;
        lw_thread_pool* shard_pool = lw_thread_pool_create(4u);
        memset(&profile, 0, sizeof(profile));
        profile.struct_size = (uint32_t)sizeof(profile);
        profile.clock = profile_clock;
        if (lw_x64_det_instance_run_profiled_ex(nhwc, backend_output, plane, shard_pool, 4u,
                                                &profile, &error) != LW_STATUS_OK) {
            lw_thread_pool_free(shard_pool);
            fprintf(stderr, "NHWC profiled run failed: %s\n", error.message);
            goto cleanup;
        }
        lw_thread_pool_free(shard_pool);
        /* Sharding smoke: the thread histograms must record every sharded
         * conv this graph size produced (zero is only valid when the graph
         * is too small to shard anything). */
        {
            uint64_t sharded_convs = 0u;
            uint32_t histogram;
            for (histogram = 1u; histogram < LW_EXECUTION_PROFILE_THREAD_HISTOGRAM_CAPACITY;
                 ++histogram) {
                sharded_convs += profile.conv_thread_histogram[histogram] +
                                 profile.conv_transpose_thread_histogram[histogram];
            }
            if (plane >= 640u * 640u && sharded_convs == 0u) {
                fprintf(stderr, "sharding smoke: no sharded convs at 640x640\n");
                goto cleanup;
            }
        }
        if (!compare_probabilities(canonical_output, backend_output, plane, &nhwc_max_abs)) {
            fprintf(stderr, "NHWC backend probability map mismatch\n");
            goto cleanup;
        }
        if (profile.layout_selected_nodes != nhwc_program->nhwc_effective_nodes ||
            profile.layout_fallback_nodes != nhwc_program->nchw_effective_nodes) {
            fprintf(stderr, "layout telemetry mismatch: selected=%llu expected=%u fallback=%llu expected=%u\n",
                    (unsigned long long)profile.layout_selected_nodes,
                    nhwc_program->nhwc_effective_nodes,
                    (unsigned long long)profile.layout_fallback_nodes,
                    nhwc_program->nchw_effective_nodes);
            goto cleanup;
        }
        if (profile.layout_transform_invocations != nhwc_program->layout_conversions) {
            fprintf(stderr, "layout transform telemetry mismatch: %llu != %u\n",
                    (unsigned long long)profile.layout_transform_invocations,
                    nhwc_program->layout_conversions);
            goto cleanup;
        }
        {
            uint64_t node_invocations = 0u;
            uint64_t expected_invocations = 0u;
            uint32_t node;
            for (node = 0u; node < LW_EXECUTION_PROFILE_NODE_CAPACITY; ++node) {
                node_invocations += profile.node_invocations[node];
            }
            for (node = 0u; node < nhwc_program->op_count; ++node) {
                const lw_x64_det_op* physical = &nhwc_program->ops[node];
                uint32_t offset;
                for (offset = 0u; offset < physical->semantic_count; ++offset) {
                    if (physical->semantic_begin + offset <
                        LW_EXECUTION_PROFILE_NODE_CAPACITY) {
                        ++expected_invocations;
                    }
                }
            }
            if (node_invocations != expected_invocations) {
                fprintf(stderr, "node invocation telemetry mismatch: %llu != %u\n",
                        (unsigned long long)node_invocations,
                        (unsigned)expected_invocations);
                goto cleanup;
            }
        }
        if (!run_backend_op_by_op(nhwc, determinism_output, plane, nhwc_program)) {
            goto cleanup;
        }
        if (memcmp(backend_output, determinism_output, (size_t)plane * sizeof(float)) != 0) {
            fprintf(stderr, "NHWC backend is not bit-deterministic\n");
            goto cleanup;
        }
        if (!check_direct_input(model, height, width, source_input,
                                backend_output, determinism_output, &error)) {
            goto cleanup;
        }
    }
    if (plane >= 640u * 640u &&
        (!check_sharded_scalar_fallback(nhwc_program, nhwc, source_input, input_count,
                                        LW_X64_DET_OP_DENSE) ||
         !check_sharded_scalar_fallback(nhwc_program, nhwc, source_input, input_count,
                                        LW_X64_DET_OP_DEPTHWISE))) {
        goto cleanup;
    }
    printf("{\"schema_version\":1,\"height\":%u,\"width\":%u,"
           "\"nhwc_ops\":%u,\"nchw_ops\":%u,"
           "\"nhwc_effective\":%u,\"nchw_effective\":%u,"
           "\"nhwc_conversions\":%u,\"nchw_conversions\":%u,"
           "\"nhwc_arena_bytes\":%llu,\"nchw_arena_bytes\":%llu,"
           "\"nhwc_max_abs_diff\":%.9g,\"nchw_max_abs_diff\":%.9g}\n",
           height, width,
           nhwc_program->op_count, nchw_program->op_count,
           nhwc_program->nhwc_effective_nodes, nhwc_program->nchw_effective_nodes,
           nhwc_program->layout_conversions, nchw_program->layout_conversions,
           (unsigned long long)nhwc_program->arena_bytes,
           (unsigned long long)nchw_program->arena_bytes,
           nhwc_max_abs, nchw_max_abs);
    /* Detector-level smoke: the enable hook must run the backend path and
     * produce the same boxes as the canonical detector on a deterministic
     * image. */
    {
        lw_detector* reference_detector = NULL;
        lw_detector* backend_detector = NULL;
        lw_detector_options options;
        lw_detection_result reference_result;
        lw_detection_result backend_result;
        lw_detection_box reference_boxes[64];
        lw_detection_box backend_boxes[64];
        uint8_t* image = NULL;
        const uint32_t image_width = 64u;
        const uint32_t image_height = 64u;
        uint32_t pixel;
        lw_detector_options_init(&options);
        lw_detection_result_init(&reference_result);
        lw_detection_result_init(&backend_result);
        image = (uint8_t*)malloc((size_t)image_width * image_height * 3u);
        if (image == NULL) {
            fprintf(stderr, "detector smoke allocation failed\n");
            goto cleanup;
        }
        for (pixel = 0u; pixel < image_width * image_height * 3u; ++pixel) {
            image[pixel] = (uint8_t)((pixel * 37u + (pixel / 192u) * 13u + 17u) & 0xffu);
        }
        if (lw_detector_create(argv[1], &options, &reference_detector, &error) != LW_STATUS_OK ||
            lw_detector_create(argv[1], &options, &backend_detector, &error) != LW_STATUS_OK) {
            fprintf(stderr, "detector create failed: %s\n", error.message);
            free(image);
            goto cleanup;
        }
        lw_detector_test_disable_x64_backend(reference_detector);
        lw_detector_test_enable_x64_backend(backend_detector);
        lw_detector_set_intra_op_thread_count(backend_detector, 4u);
        if (lw_detector_detect_bgr_u8(reference_detector, image,
                                      (uint64_t)image_width * image_height * 3u,
                                      image_width, image_height, image_width * 3u,
                                      reference_boxes, 64u, &reference_result, &error) !=
                LW_STATUS_OK ||
            lw_detector_detect_bgr_u8(backend_detector, image,
                                      (uint64_t)image_width * image_height * 3u,
                                      image_width, image_height, image_width * 3u,
                                      backend_boxes, 64u, &backend_result, &error) !=
                LW_STATUS_OK) {
            fprintf(stderr, "detector smoke run failed: %s\n", error.message);
            lw_detector_free(backend_detector);
            lw_detector_free(reference_detector);
            free(image);
            goto cleanup;
        }
        if (reference_result.box_count != backend_result.box_count) {
            fprintf(stderr, "detector smoke box count mismatch: %u != %u\n",
                    reference_result.box_count, backend_result.box_count);
            lw_detector_free(backend_detector);
            lw_detector_free(reference_detector);
            free(image);
            goto cleanup;
        }
        {
            uint32_t box;
            for (box = 0u; box < reference_result.box_count; ++box) {
                float coordinates[9] = {
                    reference_boxes[box].x1, reference_boxes[box].y1,
                    reference_boxes[box].x2, reference_boxes[box].y2,
                    reference_boxes[box].x3, reference_boxes[box].y3,
                    reference_boxes[box].x4, reference_boxes[box].y4,
                    reference_boxes[box].score
                };
                float actual[9] = {
                    backend_boxes[box].x1, backend_boxes[box].y1,
                    backend_boxes[box].x2, backend_boxes[box].y2,
                    backend_boxes[box].x3, backend_boxes[box].y3,
                    backend_boxes[box].x4, backend_boxes[box].y4,
                    backend_boxes[box].score
                };
                uint32_t field;
                for (field = 0u; field < 9u; ++field) {
                    if (fabsf(coordinates[field] - actual[field]) > 1.0e-2f) {
                        fprintf(stderr, "detector smoke box %u field %u mismatch: %.6f != %.6f\n",
                                box, field, coordinates[field], actual[field]);
                        lw_detector_free(backend_detector);
                        lw_detector_free(reference_detector);
                        free(image);
                        goto cleanup;
                    }
                }
            }
        }
        lw_detector_free(backend_detector);
        lw_detector_free(reference_detector);
        free(image);
    }
    code = 0;
cleanup:
    free(determinism_output);
    free(backend_output);
    free(canonical_output);
    free(source_input);
    lw_session_free(canonical);
    lw_x64_det_instance_free(nchw);
    lw_x64_det_instance_free(nhwc);
    lw_x64_det_program_free(nchw_program);
    lw_x64_det_program_free(nhwc_program);
    lw_model_free(model);
    return code;
}
