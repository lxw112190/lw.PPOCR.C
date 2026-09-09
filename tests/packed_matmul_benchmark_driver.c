#include "cpu_features.h"
#include "packed_matmul_internal.h"
#include "simd_kernels.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <time.h>
#endif

/* PP-OCRv6 tiny REC terminal CTC projection: [1,40,80] x [80,6906]. */
#define TERMINAL_ROWS 40u
#define TERMINAL_INNER 80u
#define TERMINAL_COLUMNS 6906u

static double monotonic_seconds(void) {
#if defined(_WIN32)
    LARGE_INTEGER counter;
    LARGE_INTEGER frequency;
    if (!QueryPerformanceFrequency(&frequency) || !QueryPerformanceCounter(&counter) ||
        frequency.QuadPart == 0) {
        return 0.0;
    }
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0.0;
    }
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
#endif
}

static int parse_positive_u32(const char* text, uint32_t* value) {
    char* end = NULL;
    unsigned long parsed;
    if (text == NULL || value == NULL || text[0] == '\0' || text[0] == '-') {
        return 0;
    }
    parsed = strtoul(text, &end, 10);
    if (end == text || *end != '\0' || parsed == 0ul || parsed > UINT32_MAX) {
        return 0;
    }
    *value = (uint32_t)parsed;
    return 1;
}

static int allocation_size(uint64_t count, size_t element_size, size_t* bytes) {
    if (bytes == NULL || element_size == 0u || count > SIZE_MAX / element_size) {
        return 0;
    }
    *bytes = (size_t)count * element_size;
    return 1;
}

static void fill_values(float* values, uint64_t count, uint32_t seed) {
    uint64_t index;
    uint32_t state = seed;
    for (index = 0u; index < count; ++index) {
        state = state * 1664525u + 1013904223u;
        values[(size_t)index] = (float)((int32_t)(state >> 9u) % 1021) / 511.0f;
    }
}

static uint64_t checksum_bytes(const void* data, size_t bytes) {
    const unsigned char* values = (const unsigned char*)data;
    uint64_t hash = UINT64_C(14695981039346656037);
    size_t index;
    for (index = 0u; index < bytes; ++index) {
        hash ^= values[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void add_bias_and_argmax(const float* logits, const float* bias, float* output,
                                uint32_t* best_indices) {
    uint32_t row;
    for (row = 0u; row < TERMINAL_ROWS; ++row) {
        const size_t row_base = (size_t)row * TERMINAL_COLUMNS;
        uint32_t best_index = 0u;
        uint32_t column;
        for (column = 0u; column < TERMINAL_COLUMNS; ++column) {
            const float value = logits[row_base + column] + bias[column];
            output[row_base + column] = value;
            if (column != 0u && value > output[row_base + best_index]) {
                best_index = column;
            }
        }
        best_indices[row] = best_index;
    }
}

int main(int argc, char** argv) {
    const lw_simd_level simd_level = lw_detect_simd_level();
    uint32_t iterations = 3u;
    const uint64_t input_count = (uint64_t)TERMINAL_ROWS * TERMINAL_INNER;
    const uint64_t weight_count = (uint64_t)TERMINAL_INNER * TERMINAL_COLUMNS;
    const uint64_t output_count = (uint64_t)TERMINAL_ROWS * TERMINAL_COLUMNS;
    uint64_t packed_count = 0u;
    size_t input_bytes;
    size_t weight_bytes;
    size_t packed_bytes;
    size_t output_bytes;
    float* input = NULL;
    float* weights = NULL;
    float* packed_weights = NULL;
    float* bias = NULL;
    float* scalar_logits = NULL;
    float* scalar_output = NULL;
    float* avx2_output = NULL;
    uint32_t* scalar_indices = NULL;
    uint32_t* avx2_indices = NULL;
    uint32_t iteration;
    double scalar_started;
    double scalar_finished;
    double avx2_started;
    double avx2_finished;
    double scalar_ms;
    double avx2_ms;
    uint64_t output_checksum;
    uint64_t argmax_checksum;
    int status = 1;

    if (argc >= 2 && !parse_positive_u32(argv[1], &iterations)) {
        fprintf(stderr, "usage: packed-matmul-benchmark-driver [iterations=3]\n");
        return 2;
    }
    if (argc > 2 || iterations > 100u ||
        !lw_packed_matmul_weight_count(TERMINAL_INNER, TERMINAL_COLUMNS, &packed_count) ||
        !allocation_size(input_count, sizeof(float), &input_bytes) ||
        !allocation_size(weight_count, sizeof(float), &weight_bytes) ||
        !allocation_size(packed_count, sizeof(float), &packed_bytes) ||
        !allocation_size(output_count, sizeof(float), &output_bytes)) {
        fprintf(stderr, "usage: packed-matmul-benchmark-driver [iterations=3]\n");
        return 2;
    }

    if (!lw_simd_level_is_avx2(simd_level)) {
        printf("{\"schema_version\":1,\"backend\":\"%s\",\"supported\":false,"
               "\"rows\":%u,\"inner_dimension\":%u,\"columns\":%u,"
               "\"iterations\":%u}\n",
               lw_simd_level_name(simd_level), TERMINAL_ROWS, TERMINAL_INNER,
               TERMINAL_COLUMNS, iterations);
        return 0;
    }

    input = (float*)malloc(input_bytes);
    weights = (float*)malloc(weight_bytes);
    packed_weights = (float*)malloc(packed_bytes);
    bias = (float*)malloc((size_t)TERMINAL_COLUMNS * sizeof(float));
    scalar_logits = (float*)malloc(output_bytes);
    scalar_output = (float*)malloc(output_bytes);
    avx2_output = (float*)malloc(output_bytes);
    scalar_indices = (uint32_t*)malloc((size_t)TERMINAL_ROWS * sizeof(uint32_t));
    avx2_indices = (uint32_t*)malloc((size_t)TERMINAL_ROWS * sizeof(uint32_t));
    if (input == NULL || weights == NULL || packed_weights == NULL || bias == NULL ||
        scalar_logits == NULL || scalar_output == NULL || avx2_output == NULL ||
        scalar_indices == NULL || avx2_indices == NULL) {
        fprintf(stderr, "terminal MatMul benchmark allocation failed\n");
        goto cleanup;
    }

    fill_values(input, input_count, 17u);
    fill_values(weights, weight_count, 31u);
    fill_values(bias, TERMINAL_COLUMNS, 47u);
    lw_pack_matmul_weights_f32(weights, TERMINAL_INNER, TERMINAL_COLUMNS, packed_weights);

    lw_scalar_packed_matmul_shared_f32(input, packed_weights, scalar_logits, 1u,
                                       TERMINAL_ROWS, TERMINAL_INNER, TERMINAL_COLUMNS);
    add_bias_and_argmax(scalar_logits, bias, scalar_output, scalar_indices);
    lw_avx2_packed_matmul_bias_argmax_f32(
        input, packed_weights, bias, avx2_output, avx2_indices, 1u, TERMINAL_ROWS,
        TERMINAL_INNER, TERMINAL_COLUMNS);
    if (memcmp(scalar_output, avx2_output, output_bytes) != 0 ||
        memcmp(scalar_indices, avx2_indices, (size_t)TERMINAL_ROWS * sizeof(uint32_t)) != 0) {
        fprintf(stderr, "terminal fused MatMul differs from scalar reference\n");
        goto cleanup;
    }

    /* Warm up both paths once so instruction dispatch and page faults are excluded. */
    lw_scalar_packed_matmul_shared_f32(input, packed_weights, scalar_logits, 1u,
                                       TERMINAL_ROWS, TERMINAL_INNER, TERMINAL_COLUMNS);
    add_bias_and_argmax(scalar_logits, bias, scalar_output, scalar_indices);
    lw_avx2_packed_matmul_bias_argmax_f32(
        input, packed_weights, bias, avx2_output, avx2_indices, 1u, TERMINAL_ROWS,
        TERMINAL_INNER, TERMINAL_COLUMNS);

    scalar_started = monotonic_seconds();
    for (iteration = 0u; iteration < iterations; ++iteration) {
        lw_scalar_packed_matmul_shared_f32(input, packed_weights, scalar_logits, 1u,
                                           TERMINAL_ROWS, TERMINAL_INNER, TERMINAL_COLUMNS);
        add_bias_and_argmax(scalar_logits, bias, scalar_output, scalar_indices);
    }
    scalar_finished = monotonic_seconds();
    avx2_started = monotonic_seconds();
    for (iteration = 0u; iteration < iterations; ++iteration) {
        lw_avx2_packed_matmul_bias_argmax_f32(
            input, packed_weights, bias, avx2_output, avx2_indices, 1u, TERMINAL_ROWS,
            TERMINAL_INNER, TERMINAL_COLUMNS);
    }
    avx2_finished = monotonic_seconds();
    if (scalar_started <= 0.0 || scalar_finished <= scalar_started || avx2_started <= 0.0 ||
        avx2_finished <= avx2_started || memcmp(scalar_output, avx2_output, output_bytes) != 0 ||
        memcmp(scalar_indices, avx2_indices, (size_t)TERMINAL_ROWS * sizeof(uint32_t)) != 0) {
        fprintf(stderr, "terminal MatMul benchmark result contract failed\n");
        goto cleanup;
    }

    scalar_ms = (scalar_finished - scalar_started) * 1000.0 / iterations;
    avx2_ms = (avx2_finished - avx2_started) * 1000.0 / iterations;
    output_checksum = checksum_bytes(avx2_output, output_bytes);
    argmax_checksum = checksum_bytes(avx2_indices, (size_t)TERMINAL_ROWS * sizeof(uint32_t));
    printf("{\"schema_version\":1,\"backend\":\"%s\",\"supported\":true,"
           "\"rows\":%u,\"inner_dimension\":%u,\"columns\":%u,"
           "\"iterations\":%u,\"scalar_ms\":%.6f,\"avx2_ms\":%.6f,"
           "\"speedup\":%.6f,\"output_checksum\":\"0x%016" PRIx64
           "\",\"argmax_checksum\":\"0x%016" PRIx64 "\"}\n",
           lw_simd_level_name(simd_level), TERMINAL_ROWS, TERMINAL_INNER,
           TERMINAL_COLUMNS, iterations, scalar_ms, avx2_ms, scalar_ms / avx2_ms,
           output_checksum, argmax_checksum);
    status = 0;

cleanup:
    free(avx2_indices);
    free(scalar_indices);
    free(avx2_output);
    free(scalar_output);
    free(scalar_logits);
    free(bias);
    free(packed_weights);
    free(weights);
    free(input);
    return status;
}
