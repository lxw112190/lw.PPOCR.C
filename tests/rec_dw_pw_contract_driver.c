#include "x64_rec_backend_internal.h"
#include "ctc_projection_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int dump_program;

static int range_contract(void) {
    const uint32_t starts[] = {0u, 1u, 2u, 9u, 10u, 11u, 12u, 41u, 43u};
    const uint32_t counts[] = {1u, 2u, 3u, 12u, 48u};
    float input[7 * 11 * 40], weights[2 * 9 * 32], bias[40], post[40];
    float full[7 * 11 * 40], tile[16 + 7 * 11 * 40 + 16];
    lw_nhwc_depthwise_desc d = {0};
    uint32_t i, j, k, stride;
    for (i = 0u; i < 7u * 11u * 40u; ++i) input[i] = (float)((int32_t)(i % 53u) - 26) / 31.0f;
    for (i = 0u; i < 2u * 9u * 32u; ++i) weights[i] = (float)((int32_t)(i % 29u) - 14) / 43.0f;
    for (i = 0u; i < 40u; ++i) { bias[i] = (float)i / 57.0f; post[i] = -(float)i / 83.0f; }
    d.batch = 1u; d.channels = 40u; d.input_height = 7u; d.input_width = 11u;
    d.output_width = 11u; d.kernel_h = d.kernel_w = 3u; d.stride_w = 1u;
    d.pad_top = d.pad_left = d.pad_bottom = d.pad_right = 1u;
    for (stride = 1u; stride <= 2u; ++stride) {
        d.stride_h = stride;
        d.output_height = (7u - 1u) / stride + 1u;
        d.post_bias = stride == 1u ? post : NULL;
        if (lw_avx2_fma_nhwc_depthwise_f32(input, weights, bias, full, &d) != LW_STATUS_OK) return 0;
        for (j = 0u; j < sizeof(starts) / sizeof(starts[0]); ++j) {
            for (k = 0u; k < sizeof(counts) / sizeof(counts[0]); ++k) {
                uint32_t begin = starts[j], count = counts[k], pixels = d.output_height * d.output_width;
                if (begin >= pixels) continue;
                if (count > pixels - begin) count = pixels - begin;
                for (i = 0u; i < sizeof(tile) / sizeof(tile[0]); ++i) tile[i] = 12345.0f;
                if (lw_avx2_fma_nhwc_depthwise_range_f32(input, weights, bias, tile + 16u, &d,
                    begin, count) != LW_STATUS_OK ||
                    memcmp(tile + 16u, full + (size_t)begin * 40u,
                           (size_t)count * 40u * sizeof(float)) != 0) return 0;
                for (i = 0u; i < 16u; ++i) if (tile[i] != 12345.0f) return 0;
                for (i = 16u + count * 40u; i < sizeof(tile) / sizeof(tile[0]); ++i)
                    if (tile[i] != 12345.0f) return 0;
            }
        }
        if (lw_avx2_fma_nhwc_depthwise_range_f32(input, weights, bias, tile, &d,
            UINT32_MAX, 2u) == LW_STATUS_OK ||
            lw_avx2_fma_nhwc_depthwise_range_f32(input, weights, bias, tile, &d,
            0u, 0u) == LW_STATUS_OK) return 0;
    }
    return 1;
}

/* The reference independently materializes the full DW plane. It does not
 * use the new range primitive or the candidate's elided intermediate. */
static int compare_pair(lw_x64_rec_instance* reference, lw_x64_rec_instance* fused,
                        uint32_t index, lw_error* error) {
    lw_x64_rec_program p = *reference->program;
    lw_x64_rec_instance r = *reference;
    lw_x64_rec_op ops[2];
    const lw_x64_rec_conv_op* a = &p.ops[index].data.conv;
    const lw_x64_rec_conv_op* b = &p.ops[index + 1u].data.conv;
    uint64_t pixels = (uint64_t)a->output_height * a->output_width;
    int ok;
    memcpy(ops, &p.ops[index], sizeof(ops));
    ops[0].data.conv.output_offset = p.arena_bytes;
    ops[1].data.conv.input_offset = p.arena_bytes;
    p.ops = ops; p.op_count = 2u;
    p.arena_bytes += pixels * a->output_channels * sizeof(float);
    r.program = &p;
    r.arena = (uint8_t*)malloc((size_t)p.arena_bytes);
    if (r.arena == NULL) return 0;
    memcpy(r.arena, reference->arena, (size_t)reference->program->arena_bytes);
    ok = lw_x64_rec_instance_run_op(&r, 0u, error) == LW_STATUS_OK &&
         lw_x64_rec_instance_run_op(&r, 1u, error) == LW_STATUS_OK &&
         lw_x64_rec_instance_run_dw_pw_pair(fused, index, error) == LW_STATUS_OK &&
         memcmp(r.arena + (size_t)b->output_offset, fused->arena + (size_t)b->output_offset,
                (size_t)pixels * b->output_channels * sizeof(float)) == 0;
    memcpy(reference->arena, r.arena, (size_t)reference->program->arena_bytes);
    free(r.arena);
    return ok;
}

static int reject_invalid(lw_x64_rec_instance* instance, uint32_t index, lw_error* error) {
    lw_x64_rec_program p = *instance->program;
    lw_x64_rec_instance r = *instance;
    lw_x64_rec_op ops[2];
    uint8_t* saved = (uint8_t*)malloc((size_t)p.arena_bytes);
    uint64_t output, scratch;
    int ok = 0;
    if (saved == NULL) return 0;
    memcpy(saved, instance->arena, (size_t)p.arena_bytes);
    memcpy(ops, &p.ops[index], sizeof(ops));
    p.ops = ops; p.op_count = 2u; r.program = &p;
    output = ops[1].data.conv.output_offset; scratch = ops[0].data.conv.scratch_bytes;
    ops[1].data.conv.output_offset = ops[0].data.conv.input_offset;
    if (lw_x64_rec_instance_run_dw_pw_pair(&r, 0u, error) != LW_STATUS_INVALID_ARGUMENT) goto cleanup;
    ops[1].data.conv.output_offset = p.arena_bytes;
    if (lw_x64_rec_instance_run_dw_pw_pair(&r, 0u, error) != LW_STATUS_INVALID_ARGUMENT) goto cleanup;
    ops[1].data.conv.output_offset = output;
    ops[0].data.conv.scratch_bytes = 0u;
    if (lw_x64_rec_instance_run_dw_pw_pair(&r, 0u, error) != LW_STATUS_INVALID_ARGUMENT) goto cleanup;
    ops[0].data.conv.scratch_bytes = scratch;
    if (lw_x64_rec_instance_run_op(&r, 0u, error) != LW_STATUS_INVALID_ARGUMENT ||
        lw_x64_rec_instance_run_op(&r, 1u, error) != LW_STATUS_INVALID_ARGUMENT) goto cleanup;
    ops[0].flags = 0u;
    if (lw_x64_rec_instance_run_dw_pw_pair(&r, 0u, error) != LW_STATUS_INVALID_ARGUMENT ||
        lw_x64_rec_instance_run_dw_pw_pair(&r, UINT32_MAX, error) != LW_STATUS_INVALID_ARGUMENT ||
        lw_x64_rec_instance_run_dw_pw_pair(NULL, 0u, error) != LW_STATUS_INVALID_ARGUMENT) goto cleanup;
    ok = memcmp(saved, instance->arena, (size_t)p.arena_bytes) == 0;
cleanup:
    free(saved);
    return ok;
}

static int run_width(lw_model* model, uint32_t width, uint32_t workers) {
    lw_x64_rec_program* program = NULL;
    lw_x64_rec_instance *reference = NULL, *fused = NULL;
    lw_thread_pool* pool = NULL;
    lw_error error;
    uint64_t count, n;
    uint32_t i, pairs = 0u;
    int result = 1;
    lw_error_init(&error);
    if (lw_x64_rec_backend_compile(model, width, &program, &error) != LW_X64_REC_COMPILE_OK ||
        lw_x64_rec_instance_create(program, &reference, &error) != LW_STATUS_OK ||
        lw_x64_rec_instance_create(program, &fused, &error) != LW_STATUS_OK) goto cleanup;
    memset(reference->arena, 0, (size_t)program->arena_bytes);
    if (width == 192u && workers == 1u && dump_program) {
        for (i = 0u; i < program->op_count; ++i) {
            if (program->ops[i].kind == LW_X64_REC_OP_DEPTHWISE) {
                fprintf(stderr, "DW op=%u semantic=%u C=%u next:", i,
                    program->ops[i].semantic_begin, program->ops[i].data.conv.input_channels);
                for (uint32_t j = i + 1u; j < program->op_count && j <= i + 4u; ++j)
                    fprintf(stderr, " %u(s%u)", program->ops[j].kind, program->ops[j].semantic_begin);
                fprintf(stderr, "\n");
            }
        }
    }
    memset(fused->arena, 0, (size_t)program->arena_bytes);
    if (workers > 1u) {
        pool = lw_thread_pool_create(workers);
        if (pool == NULL) goto cleanup;
        lw_x64_rec_instance_set_thread_pool(reference, pool, workers);
        lw_x64_rec_instance_set_thread_pool(fused, pool, workers);
    }
    {
        float* input = lw_x64_rec_instance_input(reference, &count);
        for (n = 0u; n < count; ++n) input[n] = (float)((int32_t)(n % 193u) - 96) / 97.0f;
        memcpy(lw_x64_rec_instance_input(fused, NULL), input, (size_t)count * sizeof(float));
    }
    for (i = 0u; i < program->op_count; ++i) {
        if (program->ops[i].flags & LW_X64_REC_OP_DW_PW_BEGIN) {
            if (!pairs && !reject_invalid(fused, i, &error)) goto cleanup;
            if (!compare_pair(reference, fused, i, &error)) goto cleanup;
            ++pairs; ++i;
        } else if (lw_x64_rec_instance_run_op(reference, i, &error) != LW_STATUS_OK ||
                   lw_x64_rec_instance_run_op(fused, i, &error) != LW_STATUS_OK) goto cleanup;
    }
    if (pairs == 0u) goto cleanup;
    {
        const lw_x64_rec_value* a = &program->values[program->ctc.activation_value];
        if (memcmp(reference->arena + (size_t)a->offset, fused->arena + (size_t)a->offset,
                   (size_t)a->bytes) != 0 ||
            lw_x64_rec_ctc_execute(program, reference,
                (const float*)(const void*)(reference->arena + (size_t)a->offset), &error) != LW_STATUS_OK ||
            lw_x64_rec_instance_run(fused, &error) != LW_STATUS_OK ||
            memcmp(reference->best_indices, fused->best_indices,
                   (size_t)program->time_steps * sizeof(uint32_t)) != 0 ||
            memcmp(reference->best_probabilities, fused->best_probabilities,
                   (size_t)program->time_steps * sizeof(float)) != 0) goto cleanup;
    }
    printf("DW/PW width=%u workers=%u pairs=%u arena=%llu scratch=%llu bitwise PASS\n",
        width, workers, pairs, (unsigned long long)program->arena_bytes,
        (unsigned long long)program->scratch_bytes);
    result = 0;
cleanup:
    if (result) fprintf(stderr, "DW/PW width=%u workers=%u pairs=%u failed: %s\n",
                        width, workers, pairs, error.message);
    lw_thread_pool_free(pool);
    lw_x64_rec_instance_free(reference); lw_x64_rec_instance_free(fused);
    lw_x64_rec_program_free(program);
    return result;
}

int main(int argc, char** argv) {
    const uint32_t widths[] = {192u, 320u, 480u, 640u, 960u};
    lw_model* model = NULL;
    lw_error error;
    uint32_t i;
    int result = 0;
    if (argc != 2 && (argc != 3 || strcmp(argv[2], "--dump") != 0)) return 2;
    dump_program = argc == 3;
    lw_error_init(&error);
    if (!range_contract()) { fprintf(stderr, "DW range contract failed\n"); return 1; }
    if (lw_model_load(argv[1], NULL, &model, &error) != LW_STATUS_OK) {
        fprintf(stderr, "%s\n", error.message); return 1;
    }
    for (i = 0u; i < sizeof(widths) / sizeof(widths[0]); ++i) {
        if (run_width(model, widths[i], 1u) || run_width(model, widths[i], 4u)) { result = 1; break; }
    }
    lw_model_free(model);
    return result;
}
