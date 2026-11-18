#include "x64_rec_backend_internal.h"
#include "ctc_projection_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int kernel_tails(void) {
    const uint32_t tails[] = {1u, 2u, 3u, 5u, 11u, 12u, 13u, 95u, 96u, 97u};
    const lw_nhwc_pointwise_fn kernels[] = {
        lw_avx2_fma_nhwc_pointwise_f32, lw_avx2_fma_nhwc_pointwise_4x16_f32,
        lw_avx2_fma_nhwc_pointwise_3x32_f32, lw_avx2_fma_nhwc_pointwise_2x32_f32};
    /* Deliberately non-vector-sized channels, including a scalar GELU tail. */
    float input[97 * 7], hidden[97 * 65], scratch[12 * 65 + 16];
    float expected[97 * 19], actual[97 * 19 + 16], residual[97 * 19];
    float expand[5 * 7 * 16], project[2 * 65 * 16];
    float hb[65], hp[65], ob[19], op[19];
    lw_nhwc_epilogue a = {hb, NULL, LW_NHWC_ACT_GELU, 0u, 0.0f, 0.0f, hp};
    lw_nhwc_epilogue b = {ob, residual, LW_NHWC_ACT_NONE, 0u, 0.0f, 0.0f, op};
    uint32_t i, j, k;
    for (i = 0u; i < 97u * 7u; ++i) input[i] = (float)((int32_t)(i % 43u) - 21) / 31.0f;
    for (i = 0u; i < 5u * 7u * 16u; ++i) expand[i] = (float)((int32_t)(i % 29u) - 14) / 53.0f;
    for (i = 0u; i < 2u * 65u * 16u; ++i) project[i] = (float)((int32_t)(i % 37u) - 18) / 59.0f;
    for (i = 0u; i < 97u * 19u; ++i) residual[i] = (float)(i % 17u) / 23.0f;
    for (i = 0u; i < 65u; ++i) { hb[i] = (float)i / 71.0f; hp[i] = -(float)i / 137.0f; }
    for (i = 0u; i < 19u; ++i) { ob[i] = -(float)i / 47.0f; op[i] = (float)i / 101.0f; }
    for (k = 0u; k < 4u; ++k) {
        for (j = 0u; j < sizeof(tails) / sizeof(tails[0]); ++j) {
            uint32_t n = tails[j];
            for (i = 0u; i < 97u * 19u + 16u; ++i) actual[i] = 12345.0f;
            for (i = 0u; i < 12u * 65u + 16u; ++i) scratch[i] = 12345.0f;
            kernels[k](input, expand, &a, hidden, n, 7u, 65u);
            kernels[k](hidden, project, &b, expected, n, 65u, 19u);
            lw_nhwc_ffn_tile_f32(input, expand, &a, project, &b, actual, scratch,
                n, 7u, 65u, 19u, 12u, kernels[k], kernels[k]);
            if (memcmp(expected, actual, (size_t)n * 19u * sizeof(float)) != 0) return 0;
            for (i = n * 19u; i < 97u * 19u + 16u; ++i) if (actual[i] != 12345.0f) return 0;
            for (i = 12u * 65u; i < 12u * 65u + 16u; ++i) if (scratch[i] != 12345.0f) return 0;
        }
        /* Equal-width residual FFN, including a tile boundary: projection
         * overwrites its original input only after expanding the whole tile. */
        b.residual = input;
        kernels[k](input, expand, &a, hidden, 97u, 7u, 65u);
        kernels[k](hidden, project, &b, expected, 97u, 65u, 7u);
        memcpy(actual, input, sizeof(input));
        b.residual = actual;
        lw_nhwc_ffn_tile_f32(actual, expand, &a, project, &b, actual, scratch,
            97u, 7u, 65u, 7u, 12u, kernels[k], kernels[k]);
        if (memcmp(actual, expected, sizeof(input)) != 0) return 0;
        b.residual = residual;
    }
    return 1;
}

static int compare_pair(lw_x64_rec_instance* reference, lw_x64_rec_instance* tiled,
                        uint32_t index, lw_error* error) {
    const lw_x64_rec_conv_op* a = &tiled->program->ops[index].data.conv;
    const lw_x64_rec_conv_op* b = &tiled->program->ops[index + 1u].data.conv;
    size_t pixels = (size_t)a->input_height * a->input_width;
    if (tiled->program->ops[index].flags & LW_X64_REC_OP_FFN_LOCAL) {
        /* Reconstruct the unfused graph's full hidden plane independently of
         * the candidate planner. Its offsets must never enter the candidate. */
        lw_x64_rec_program p = *reference->program;
        lw_x64_rec_instance r = *reference;
        lw_x64_rec_op ops[2];
        uint64_t hidden_bytes = pixels * a->output_channels * sizeof(float);
        int ok;
        memcpy(ops, &p.ops[index], sizeof(ops));
        ops[0].data.conv.output_offset = p.arena_bytes;
        ops[1].data.conv.input_offset = p.arena_bytes;
        p.ops = ops;
        p.op_count = 2u;
        p.arena_bytes += hidden_bytes;
        r.program = &p;
        r.arena = (uint8_t*)malloc((size_t)p.arena_bytes);
        if (r.arena == NULL) return 0;
        memcpy(r.arena, reference->arena, (size_t)reference->program->arena_bytes);
        ok = lw_x64_rec_instance_run_op(&r, 0u, error) == LW_STATUS_OK &&
             lw_x64_rec_instance_run_op(&r, 1u, error) == LW_STATUS_OK &&
             lw_x64_rec_instance_run_ffn_pair(tiled, index, error) == LW_STATUS_OK &&
             memcmp(r.arena + (size_t)b->output_offset,
                    tiled->arena + (size_t)b->output_offset,
                    pixels * b->output_channels * sizeof(float)) == 0;
        memcpy(reference->arena, r.arena, (size_t)reference->program->arena_bytes);
        free(r.arena);
        return ok;
    }
    if (lw_x64_rec_instance_run_op(reference, index, error) != LW_STATUS_OK ||
        lw_x64_rec_instance_run_op(reference, index + 1u, error) != LW_STATUS_OK ||
        lw_x64_rec_instance_run_ffn_pair(tiled, index, error) != LW_STATUS_OK) return 0;
    return memcmp(reference->arena + (size_t)a->output_offset,
                  tiled->arena + (size_t)a->output_offset,
                  pixels * a->output_channels * sizeof(float)) == 0 &&
           memcmp(reference->arena + (size_t)b->output_offset,
                  tiled->arena + (size_t)b->output_offset,
                  pixels * b->output_channels * sizeof(float)) == 0;
}

static int tails_and_rejections(lw_x64_rec_instance* reference,
                                lw_x64_rec_instance* tiled, uint32_t index,
                                lw_error* error) {
    const uint32_t tails[] = {1u, 2u, 3u, 5u, 11u, 12u, 13u, 95u, 96u, 97u};
    lw_x64_rec_program short_program = *tiled->program;
    lw_x64_rec_op ops[2];
    lw_x64_rec_instance a = *reference, b = *tiled;
    size_t capacity = (size_t)short_program.arena_bytes;
    uint8_t* saved = (uint8_t*)malloc(capacity);
    uint32_t j, k;
    int ok = 0;
    if (saved == NULL) return 0;
    memcpy(saved, tiled->arena, capacity);
    memcpy(ops, &short_program.ops[index], sizeof(ops));
    short_program.ops = ops;
    short_program.op_count = 2u;
    a.program = b.program = &short_program;
    for (j = 0u; j < sizeof(tails) / sizeof(tails[0]); ++j) {
        /* A smaller independent prefix tests all row-kernel and FFN tails. */
        uint64_t original = (uint64_t)tiled->program->ops[index].data.conv.input_height *
                            tiled->program->ops[index].data.conv.input_width;
        if (tails[j] > original) continue;
        for (k = 0u; k < 2u; ++k) {
            ops[k].data.conv.input_height = ops[k].data.conv.output_height = 1u;
            ops[k].data.conv.input_width = ops[k].data.conv.output_width = tails[j];
        }
        memcpy(a.arena, saved, capacity);
        memcpy(b.arena, saved, capacity);
        if (!compare_pair(&a, &b, 0u, error)) goto cleanup;
    }
    /* Fail before writing: overlapping unread input, escaping arena,
     * disconnected pair, noneligible activation, unmarked pair and bad index. */
    {
        uint64_t offset = ops[1].data.conv.output_offset;
        uint16_t flags = ops[0].flags;
        ops[0].flags &= (uint16_t)(UINT16_MAX ^ LW_X64_REC_OP_FFN_INPLACE);
        ops[1].data.conv.output_offset = ops[0].data.conv.input_offset;
        if (lw_x64_rec_instance_run_ffn_pair(&b, 0u, error) != LW_STATUS_INVALID_ARGUMENT) goto cleanup;
        ops[1].data.conv.output_offset = short_program.arena_bytes;
        if (lw_x64_rec_instance_run_ffn_pair(&b, 0u, error) != LW_STATUS_INVALID_ARGUMENT) goto cleanup;
        ops[1].data.conv.output_offset = offset;
        ops[0].flags = flags;
        ++ops[1].data.conv.input_offset;
        if (lw_x64_rec_instance_run_ffn_pair(&b, 0u, error) != LW_STATUS_INVALID_ARGUMENT) goto cleanup;
        --ops[1].data.conv.input_offset;
        ops[0].data.conv.activation = LW_NHWC_ACT_NONE;
        if (lw_x64_rec_instance_run_ffn_pair(&b, 0u, error) != LW_STATUS_INVALID_ARGUMENT) goto cleanup;
        ops[0].data.conv.activation = LW_NHWC_ACT_GELU;
        if (ops[0].flags & LW_X64_REC_OP_FFN_LOCAL) {
            uint64_t scratch = ops[0].data.conv.scratch_bytes;
            ops[0].data.conv.scratch_bytes = 0u;
            if (lw_x64_rec_instance_run_ffn_pair(&b, 0u, error) != LW_STATUS_INVALID_ARGUMENT) goto cleanup;
            ops[0].data.conv.scratch_bytes = scratch;
            if (lw_x64_rec_instance_run_op(&b, 0u, error) != LW_STATUS_INVALID_ARGUMENT ||
                lw_x64_rec_instance_run_op(&b, 1u, error) != LW_STATUS_INVALID_ARGUMENT) goto cleanup;
        }
        ops[0].flags = 0u;
        if (lw_x64_rec_instance_run_ffn_pair(&b, 0u, error) != LW_STATUS_INVALID_ARGUMENT ||
            lw_x64_rec_instance_run_ffn_pair(&b, UINT32_MAX, error) != LW_STATUS_INVALID_ARGUMENT ||
            lw_x64_rec_instance_run_ffn_pair(NULL, 0u, error) != LW_STATUS_INVALID_ARGUMENT) goto cleanup;
    }
    if (memcmp(a.arena, b.arena, capacity) != 0) goto cleanup;
    ok = 1;
cleanup:
    memcpy(reference->arena, saved, capacity);
    memcpy(tiled->arena, saved, capacity);
    free(saved);
    return ok;
}

static int run_width(lw_model* model, uint32_t width, uint32_t workers) {
    lw_x64_rec_program* program = NULL;
    lw_x64_rec_instance *reference = NULL, *tiled = NULL;
    lw_thread_pool* pool = NULL;
    lw_error error;
    uint64_t count = 0u, element;
    uint32_t i, pairs = 0u, inplace = 0u;
    int result = 1;
    lw_error_init(&error);
    if (lw_x64_rec_backend_compile(model, width, &program, &error) != LW_X64_REC_COMPILE_OK ||
        lw_x64_rec_instance_create(program, &reference, &error) != LW_STATUS_OK ||
        lw_x64_rec_instance_create(program, &tiled, &error) != LW_STATUS_OK) goto cleanup;
    memset(reference->arena, 0, (size_t)program->arena_bytes);
    memset(tiled->arena, 0, (size_t)program->arena_bytes);
    if (workers > 1u) {
        pool = lw_thread_pool_create(workers);
        if (pool == NULL) goto cleanup;
        lw_x64_rec_instance_set_thread_pool(reference, pool, workers);
        lw_x64_rec_instance_set_thread_pool(tiled, pool, workers);
    }
    {
        float* input = lw_x64_rec_instance_input(reference, &count);
        for (element = 0u; element < count; ++element)
            input[element] = (float)((int32_t)(element % 193u) - 96) / 97.0f;
        memcpy(lw_x64_rec_instance_input(tiled, NULL), input, (size_t)count * sizeof(float));
    }
    for (i = 0u; i < program->op_count; ++i) {
        if (program->ops[i].flags & LW_X64_REC_OP_FFN_BEGIN) {
            if (pairs == 0u && !tails_and_rejections(reference, tiled, i, &error)) goto cleanup;
            if (!compare_pair(reference, tiled, i, &error)) goto cleanup;
            if (program->ops[i].flags & LW_X64_REC_OP_FFN_INPLACE) ++inplace;
            ++pairs;
            ++i;
        } else if (lw_x64_rec_instance_run_op(reference, i, &error) != LW_STATUS_OK ||
                   lw_x64_rec_instance_run_op(tiled, i, &error) != LW_STATUS_OK) goto cleanup;
    }
    if (pairs == 0u) goto cleanup;
    {
        const lw_x64_rec_value* activation = &program->values[program->ctc.activation_value];
        if (memcmp(reference->arena + (size_t)activation->offset,
                   tiled->arena + (size_t)activation->offset, (size_t)activation->bytes) != 0 ||
            lw_x64_rec_ctc_execute(program, reference,
                (const float*)(const void*)(reference->arena + (size_t)activation->offset), &error) != LW_STATUS_OK ||
            lw_x64_rec_instance_run(tiled, &error) != LW_STATUS_OK ||
            memcmp(reference->best_indices, tiled->best_indices,
                   (size_t)program->time_steps * sizeof(uint32_t)) != 0 ||
            memcmp(reference->best_probabilities, tiled->best_probabilities,
                   (size_t)program->time_steps * sizeof(float)) != 0) goto cleanup;
    }
    printf("FFN width=%u workers=%u pairs=%u inplace=%u arena=%llu scratch=%llu bitwise PASS\n",
        width, workers, pairs, inplace, (unsigned long long)program->arena_bytes,
        (unsigned long long)program->scratch_bytes);
    result = 0;
cleanup:
    if (result) fprintf(stderr, "FFN width=%u workers=%u failed: %s\n", width, workers, error.message);
    lw_thread_pool_free(pool);
    lw_x64_rec_instance_free(reference);
    lw_x64_rec_instance_free(tiled);
    lw_x64_rec_program_free(program);
    return result;
}

int main(int argc, char** argv) {
    const uint32_t widths[] = {192u, 320u, 480u, 640u, 960u};
    lw_model* model = NULL;
    lw_error error;
    uint32_t i;
    int result = 0;
    if (argc != 2) return 2;
    lw_error_init(&error);
    if (!kernel_tails()) { fprintf(stderr, "FFN kernel tails failed\n"); return 1; }
    if (lw_model_load(argv[1], NULL, &model, &error) != LW_STATUS_OK) {
        fprintf(stderr, "%s\n", error.message);
        return 1;
    }
    for (i = 0u; i < sizeof(widths) / sizeof(widths[0]); ++i) {
        if (run_width(model, widths[i], 1u) || run_width(model, widths[i], 4u)) {
            result = 1;
            break;
        }
    }
    lw_model_free(model);
    return result;
}
