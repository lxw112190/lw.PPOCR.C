#include "cpu_features.h"
#include "scalar_kernels.h"
#include "simd_kernels.h"

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <time.h>
#endif

typedef struct benchmark_case {
    const char* name;
    uint32_t channels;
    uint32_t height;
    uint32_t width_divisor;
} benchmark_case;

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
        values[(size_t)index] =
            -6.0f + 12.0f * (float)(state & UINT32_C(0xffff)) / 65535.0f;
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

static int run_case(const benchmark_case* item, uint32_t target_width, uint32_t iterations,
                    int first) {
    const uint32_t width = target_width / item->width_divisor;
    const uint64_t count = (uint64_t)item->channels * item->height * width;
    size_t bytes;
    float* input = NULL;
    float* reference = NULL;
    float* output = NULL;
    uint32_t iteration;
    uint64_t index;
    double scalar_started;
    double scalar_finished;
    double avx2_started;
    double avx2_finished;
    double scalar_ms;
    double avx2_ms;
    float maximum_error = 0.0f;
    uint64_t checksum;
    int ok = 0;

    if (width == 0u || !allocation_size(count, sizeof(float), &bytes)) {
        fprintf(stderr, "invalid Erf benchmark geometry: %s\n", item->name);
        goto cleanup;
    }
    input = (float*)malloc(bytes);
    reference = (float*)malloc(bytes);
    output = (float*)malloc(bytes);
    if (input == NULL || reference == NULL || output == NULL) {
        fprintf(stderr, "Erf benchmark allocation failed: %s\n", item->name);
        goto cleanup;
    }
    fill_values(input, count, 17u + item->channels + item->height);
    if (lw_scalar_erf_f32(input, reference, count) != LW_STATUS_OK) {
        fprintf(stderr, "scalar Erf failed: %s\n", item->name);
        goto cleanup;
    }
    lw_avx2_erf_f32(input, output, count);
    for (index = 0u; index < count; ++index) {
        const float error = fabsf(output[(size_t)index] - reference[(size_t)index]);
        if (error > maximum_error) {
            maximum_error = error;
        }
    }
    if (maximum_error > 5.0e-7f) {
        fprintf(stderr, "Erf approximation error %.9g exceeds limit: %s\n",
                (double)maximum_error, item->name);
        goto cleanup;
    }

    /* Warm up both paths once so dispatch and page faults are excluded. */
    (void)lw_scalar_erf_f32(input, reference, count);
    lw_avx2_erf_f32(input, output, count);
    scalar_started = monotonic_seconds();
    for (iteration = 0u; iteration < iterations; ++iteration) {
        if (lw_scalar_erf_f32(input, reference, count) != LW_STATUS_OK) {
            fprintf(stderr, "scalar Erf benchmark failed: %s\n", item->name);
            goto cleanup;
        }
    }
    scalar_finished = monotonic_seconds();
    avx2_started = monotonic_seconds();
    for (iteration = 0u; iteration < iterations; ++iteration) {
        lw_avx2_erf_f32(input, output, count);
    }
    avx2_finished = monotonic_seconds();
    if (scalar_started <= 0.0 || scalar_finished <= scalar_started ||
        avx2_started <= 0.0 || avx2_finished <= avx2_started) {
        fprintf(stderr, "Erf benchmark timer failed: %s\n", item->name);
        goto cleanup;
    }

    scalar_ms = (scalar_finished - scalar_started) * 1000.0 / iterations;
    avx2_ms = (avx2_finished - avx2_started) * 1000.0 / iterations;
    checksum = checksum_bytes(output, bytes);
    printf("%s{\"name\":\"%s\",\"channels\":%u,\"height\":%u,\"width\":%u,"
           "\"elements\":%" PRIu64 ",\"scalar_ms\":%.6f,\"avx2_ms\":%.6f,"
           "\"speedup\":%.6f,\"max_abs_error\":%.9g,"
           "\"checksum\":\"0x%016" PRIx64 "\"}",
           first ? "" : ",", item->name, item->channels, item->height, width, count,
           scalar_ms, avx2_ms, scalar_ms / avx2_ms, (double)maximum_error, checksum);
    ok = 1;

cleanup:
    free(output);
    free(reference);
    free(input);
    return ok;
}

int main(int argc, char** argv) {
    static const benchmark_case cases[] = {
        {"rec-node2-24x24", 24u, 24u, 2u},
        {"rec-96x12-a", 96u, 12u, 4u},
        {"rec-96x6", 96u, 6u, 4u},
        {"rec-192x6", 192u, 6u, 4u},
        {"rec-192x3", 192u, 3u, 4u},
        {"rec-320x3", 320u, 3u, 4u},
    };
    uint32_t target_width = 320u;
    uint32_t iterations = 3u;
    size_t index;
    const lw_simd_level simd_level = lw_detect_simd_level();

    if (argc > 3 || (argc >= 2 && !parse_positive_u32(argv[1], &target_width)) ||
        (argc >= 3 && !parse_positive_u32(argv[2], &iterations)) ||
        (target_width != 320u && target_width != 960u) || iterations > 100u) {
        fprintf(stderr, "usage: erf-benchmark-driver [target-width=320] [iterations=3]\n");
        return 2;
    }
    if (!lw_simd_level_is_avx2(simd_level)) {
        printf("{\"schema_version\":1,\"backend\":\"%s\",\"supported\":false,"
               "\"target_width\":%u,\"iterations\":%u,\"cases\":[] }\n",
               lw_simd_level_name(simd_level), target_width, iterations);
        return 0;
    }
    printf("{\"schema_version\":1,\"backend\":\"%s\",\"supported\":true,"
           "\"target_width\":%u,\"iterations\":%u,\"cases\":[",
           lw_simd_level_name(simd_level), target_width, iterations);
    for (index = 0u; index < sizeof(cases) / sizeof(cases[0]); ++index) {
        if (!run_case(&cases[index], target_width, iterations, index == 0u)) {
            return 1;
        }
    }
    printf("]}\n");
    return 0;
}
