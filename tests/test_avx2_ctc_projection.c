#include "cpu_features.h"
#include "packed_matmul_internal.h"
#include "simd_kernels.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

#define ROWS 7u
#define INNER 3u

/* Linux ASan sees an exact-sized heap allocation. On Windows, also place
 * the unpadded bias immediately before an inaccessible page: a vector load
 * past the last class then fails even without a sanitizer toolchain. */
static float* allocate_bias(uint32_t columns, void** allocation) {
#ifdef _WIN32
    SYSTEM_INFO info;
    size_t data_bytes;
    void* memory;
    DWORD previous;
    GetSystemInfo(&info);
    data_bytes = ((size_t)columns * sizeof(float) + info.dwPageSize - 1u) /
                 info.dwPageSize * info.dwPageSize;
    memory = VirtualAlloc(NULL, data_bytes + info.dwPageSize,
                          MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (memory == NULL) return NULL;
    if (!VirtualProtect((unsigned char*)memory + data_bytes, info.dwPageSize,
                        PAGE_NOACCESS, &previous)) {
        VirtualFree(memory, 0u, MEM_RELEASE);
        return NULL;
    }
    *allocation = memory;
    return (float*)((unsigned char*)memory + data_bytes) - columns;
#else
    *allocation = malloc((size_t)columns * sizeof(float));
    return (float*)*allocation;
#endif
}

static void free_bias(void* allocation) {
#ifdef _WIN32
    if (allocation != NULL) VirtualFree(allocation, 0u, MEM_RELEASE);
#else
    free(allocation);
#endif
}

static int run_case(uint32_t columns) {
    static const float input[ROWS][INNER] = {
        {1.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f}
    };
    uint64_t packed_count;
    void* bias_allocation = NULL;
    float* bias = allocate_bias(columns, &bias_allocation);
    float* weights = (float*)calloc((size_t)INNER * columns, sizeof(float));
    float* padded_bias = NULL;
    float* packed = NULL;
    int result = 1;
    if (!lw_packed_matmul_weight_count(INNER, columns, &packed_count)) goto cleanup;
    packed = (float*)malloc((size_t)packed_count * sizeof(float));
    padded_bias = (float*)calloc((columns + 15u) / 16u * 16u, sizeof(float));
    if (bias == NULL || weights == NULL || packed == NULL || padded_bias == NULL)
        goto cleanup;
    for (uint32_t column = 0u; column < columns; ++column) {
        bias[column] = (float)(column % 7u) * 0.03125f;
    }
    bias[0] = 0.5f;
    weights[columns - 1u] = 4.0f;
    weights[columns + 1u] = 3.0f;
    weights[2u * columns + columns - 1u] = 5.0f;
    memcpy(padded_bias, bias, (size_t)columns * sizeof(float));
    lw_pack_matmul_weights_f32(weights, INNER, columns, packed);
    for (uint32_t with_bias = 0u; with_bias < 2u; ++with_bias) {
        const float* active_bias = with_bias != 0u ? bias : NULL;
        for (uint32_t rows = 1u; rows <= ROWS; ++rows) {
            uint32_t indices[ROWS];
            float scores[ROWS];
            float probabilities[ROWS] = {0.0f};
            float padded_probabilities[ROWS] = {0.0f};
            /* Supply the scalar head result; the production argmax kernel
             * processes four-row blocks and has a separate tail dispatcher. */
            for (uint32_t row = 0u; row < rows; ++row) {
                indices[row] = row == 2u || row == 4u ? 0u :
                    (row == 3u || row == 5u ? 1u : columns - 1u);
                scores[row] = row == 0u ? 4.0f : row == 1u ? 8.0f :
                    (row == 3u || row == 5u ? 3.0f : row == 6u ? 5.0f : 0.0f);
                if (active_bias != NULL) scores[row] += active_bias[indices[row]];
            }
            lw_avx2_ctc_row_probabilities_f32(
                &input[0][0], packed, active_bias, indices, scores,
                probabilities, rows, INNER, columns);
            lw_avx2_ctc_row_probabilities_f32(
                &input[0][0], packed, with_bias != 0u ? padded_bias : NULL,
                indices, scores, padded_probabilities, rows, INNER, columns);
            if (memcmp(probabilities, padded_probabilities, sizeof(probabilities)) != 0) {
                fprintf(stderr, "CTC unpadded/padded bias mismatch: classes=%u rows=%u\n",
                        columns, rows);
                goto cleanup;
            }
            for (uint32_t row = 0u; row < rows; ++row) {
                const uint32_t expected_best = row == 2u || row == 4u ? 0u :
                    (row == 3u || row == 5u ? 1u : columns - 1u);
                float expected_score = row == 0u ? 4.0f : row == 1u ? 8.0f :
                    (row == 3u || row == 5u ? 3.0f : row == 6u ? 5.0f : 0.0f);
                float expected_probability = 0.0f;
                if (active_bias != NULL) expected_score += active_bias[expected_best];
                if (indices[row] != expected_best || scores[row] != expected_score) {
                    fprintf(stderr, "CTC argmax mismatch: classes=%u row=%u bias=%u\n",
                            columns, row, with_bias);
                    goto cleanup;
                }
                if (expected_best != 0u &&
                    (row == 0u || expected_best != indices[row - 1u])) {
                    float sum = 0.0f;
                    for (uint32_t column = 0u; column < columns; ++column) {
                        float value = 0.0f;
                        for (uint32_t inner = 0u; inner < INNER; ++inner) {
                            value = fmaf(input[row][inner], weights[inner * columns + column], value);
                        }
                        if (active_bias != NULL) value += active_bias[column];
                        sum += expf(value - expected_score);
                    }
                    expected_probability = 1.0f / sum;
                }
                if (!isfinite(probabilities[row]) ||
                    fabsf(probabilities[row] - expected_probability) >
                        fmaxf(1.0e-7f, expected_probability * 3.0e-5f)) {
                    fprintf(stderr, "CTC probability mismatch: classes=%u row=%u bias=%u\n",
                            columns, row, with_bias);
                    goto cleanup;
                }
            }
        }
    }
    result = 0;
cleanup:
    free_bias(bias_allocation);
    free(weights);
    free(packed);
    free(padded_bias);
    if (result != 0) fprintf(stderr, "CTC tail case failed: classes=%u\n", columns);
    return result;
}

int main(void) {
    const lw_cpu_capabilities capabilities = lw_get_cpu_capabilities();
    if (!capabilities.has_avx2_fma) {
        puts("AVX2+FMA unavailable; skipping CTC vector-tail regression");
        return 77;
    }
    for (uint32_t columns = 2u; columns <= 48u; ++columns) {
        if (run_case(columns) != 0) return 1;
    }
    if (run_case(6626u) != 0 || run_case(18386u) != 0) return 1;
    puts("AVX2 CTC: all column tails, row tails, blank/repeat and bias boundaries passed");
    return 0;
}
