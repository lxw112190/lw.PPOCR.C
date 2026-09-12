#include "cpu_features.h"
#include "lw_infer.h"
#include "packed_conv3x3_internal.h"
#include "scalar_kernels.h"

#include <inttypes.h>
#include <math.h>
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

typedef struct benchmark_case {
    const char* name;
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t input_height;
    uint32_t input_width_divisor;
    uint32_t fixed_input_width;
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
        values[(size_t)index] = (float)((int32_t)(state >> 9u) % 1021) / 511.0f;
    }
}

static uint64_t checksum_bytes(const void* data, size_t bytes) {
    const unsigned char* values = (const unsigned char*)data;
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t index;
    for (index = 0u; index < bytes; ++index) {
        hash ^= values[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

#if defined(LW_AVX2_FMA_CONV3X3_DISPATCH)
static float max_abs_difference(const float* expected, const float* actual, uint64_t count) {
    uint64_t index;
    float maximum = 0.0f;
    for (index = 0u; index < count; ++index) {
        const float difference = fabsf(expected[(size_t)index] - actual[(size_t)index]);
        if (difference > maximum) {
            maximum = difference;
        }
    }
    return maximum;
}
#endif

static lw_status run_dispatched(const float* input, const float* weights, const float* bias,
                                uint32_t output_channels, float* output,
                                const int32_t input_dimensions[4],
                                const int32_t output_dimensions[4]) {
    const int32_t weight_dimensions[4] = {
        (int32_t)output_channels, input_dimensions[1], 3, 3};
    const int32_t kernel[2] = {3, 3};
    const int32_t strides[2] = {2, 2};
    const int32_t dilations[2] = {1, 1};
    const int32_t pads[4] = {1, 1, 1, 1};
    return lw_scalar_conv2d_f32(input, weights, bias, output_channels, output,
                                input_dimensions, weight_dimensions, output_dimensions,
                                kernel, strides, dilations, pads, 1u);
}

static int run_case(const benchmark_case* item, uint32_t target_width, uint32_t iterations,
                    int first) {
    const uint32_t input_width = item->fixed_input_width != 0u
                                     ? item->fixed_input_width
                                     : target_width / item->input_width_divisor;
    const uint32_t output_height = item->input_height / 2u;
    const uint32_t output_width = input_width / 2u;
    const uint64_t input_count =
        (uint64_t)item->input_channels * item->input_height * input_width;
    const uint64_t weight_count =
        (uint64_t)item->output_channels * item->input_channels * 9u;
    const uint64_t output_count =
        (uint64_t)item->output_channels * output_height * output_width;
    uint64_t packed_count = 0u;
    size_t input_bytes;
    size_t weight_bytes;
    size_t output_bytes;
    size_t packed_bytes;
    float* input = NULL;
    float* weights = NULL;
    float* packed_weights = NULL;
    float* bias = NULL;
    float* reference = NULL;
    float* output = NULL;
    float* packed_output = NULL;
    int32_t input_dimensions[4] = {1, 0, 0, 0};
    int32_t output_dimensions[4] = {1, 0, 0, 0};
    double scalar_started;
    double scalar_finished;
    double dispatched_started;
    double dispatched_finished;
    double packed_started;
    double packed_finished;
    double scalar_ms;
    double dispatched_ms;
    double packed_ms;
    uint64_t checksum;
    uint32_t iteration;
    int ok = 0;

    if (input_width == 0u || output_height == 0u || output_width == 0u ||
        !allocation_size(input_count, sizeof(float), &input_bytes) ||
        !allocation_size(weight_count, sizeof(float), &weight_bytes) ||
        !allocation_size(output_count, sizeof(float), &output_bytes) ||
        !lw_packed_conv3x3_stride2_weight_count(
            item->input_channels, item->output_channels, &packed_count) ||
        !allocation_size(packed_count, sizeof(float), &packed_bytes) ||
        item->input_channels > INT32_MAX || item->output_channels > INT32_MAX ||
        item->input_height > INT32_MAX || input_width > INT32_MAX ||
        output_height > INT32_MAX || output_width > INT32_MAX) {
        fprintf(stderr, "invalid benchmark geometry: %s\n", item->name);
        goto cleanup;
    }
    input = (float*)malloc(input_bytes);
    weights = (float*)malloc(weight_bytes);
    packed_weights = (float*)malloc(packed_bytes);
    bias = (float*)malloc((size_t)item->output_channels * sizeof(float));
    reference = (float*)malloc(output_bytes);
    output = (float*)malloc(output_bytes);
    packed_output = (float*)malloc(output_bytes);
    if (input == NULL || weights == NULL || packed_weights == NULL || bias == NULL ||
        reference == NULL || output == NULL || packed_output == NULL) {
        fprintf(stderr, "benchmark allocation failed: %s\n", item->name);
        goto cleanup;
    }
    fill_values(input, input_count, 17u + item->input_channels);
    fill_values(weights, weight_count, 31u + item->output_channels);
    fill_values(bias, item->output_channels, 47u + item->input_height);
    lw_pack_conv3x3_stride2_weights_f32(
        weights, item->input_channels, item->output_channels, packed_weights);
    input_dimensions[1] = (int32_t)item->input_channels;
    input_dimensions[2] = (int32_t)item->input_height;
    input_dimensions[3] = (int32_t)input_width;
    output_dimensions[1] = (int32_t)item->output_channels;
    output_dimensions[2] = (int32_t)output_height;
    output_dimensions[3] = (int32_t)output_width;

    lw_scalar_conv3x3_stride2_pad1_f32(input, weights, bias, reference,
                                        input_dimensions, output_dimensions);
    if (run_dispatched(input, weights, bias, item->output_channels, output,
                       input_dimensions, output_dimensions) != LW_STATUS_OK ||
        memcmp(reference, output, output_bytes) != 0) {
        fprintf(stderr, "stride-2 Conv result mismatch: %s\n", item->name);
        goto cleanup;
    }
    lw_packed_conv3x3_stride2_pad1_f32(
        input, packed_weights, bias, packed_output, input_dimensions, output_dimensions);
#if defined(LW_AVX2_FMA_CONV3X3_DISPATCH)
    if (!isfinite(max_abs_difference(reference, packed_output, output_count)) ||
        max_abs_difference(reference, packed_output, output_count) > 1.0e-2f) {
#else
    if (memcmp(reference, packed_output, output_bytes) != 0) {
#endif
        fprintf(stderr, "packed stride-2 Conv result mismatch: %s\n", item->name);
        goto cleanup;
    }

    scalar_started = monotonic_seconds();
    for (iteration = 0u; iteration < iterations; ++iteration) {
        lw_scalar_conv3x3_stride2_pad1_f32(input, weights, bias, reference,
                                            input_dimensions, output_dimensions);
    }
    scalar_finished = monotonic_seconds();
    dispatched_started = monotonic_seconds();
    for (iteration = 0u; iteration < iterations; ++iteration) {
        if (run_dispatched(input, weights, bias, item->output_channels, output,
                           input_dimensions, output_dimensions) != LW_STATUS_OK) {
            fprintf(stderr, "stride-2 Conv dispatch failed: %s\n", item->name);
            goto cleanup;
        }
    }
    dispatched_finished = monotonic_seconds();
    packed_started = monotonic_seconds();
    for (iteration = 0u; iteration < iterations; ++iteration) {
        lw_packed_conv3x3_stride2_pad1_f32(
            input, packed_weights, bias, packed_output, input_dimensions,
            output_dimensions);
    }
    packed_finished = monotonic_seconds();
    if (scalar_started <= 0.0 || scalar_finished <= scalar_started ||
        dispatched_started <= 0.0 || dispatched_finished <= dispatched_started ||
        packed_started <= 0.0 || packed_finished <= packed_started ||
        memcmp(reference, output, output_bytes) != 0 ||
#if defined(LW_AVX2_FMA_CONV3X3_DISPATCH)
        !isfinite(max_abs_difference(reference, packed_output, output_count)) ||
        max_abs_difference(reference, packed_output, output_count) > 1.0e-2f) {
#else
        memcmp(reference, packed_output, output_bytes) != 0) {
#endif
        fprintf(stderr, "stride-2 Conv benchmark failed: %s\n", item->name);
        goto cleanup;
    }
    scalar_ms = (scalar_finished - scalar_started) * 1000.0 / iterations;
    dispatched_ms = (dispatched_finished - dispatched_started) * 1000.0 / iterations;
    packed_ms = (packed_finished - packed_started) * 1000.0 / iterations;
    checksum = checksum_bytes(output, output_bytes);
    printf("%s{\"name\":\"%s\",\"input\":[1,%u,%u,%u],"
           "\"output\":[1,%u,%u,%u],\"scalar_ms\":%.6f,"
           "\"dispatched_ms\":%.6f,\"speedup\":%.6f,"
           "\"packed_ms\":%.6f,\"packed_speedup\":%.6f,"
           "\"checksum\":\"0x%016" PRIx64 "\"}",
           first ? "" : ",", item->name, item->input_channels, item->input_height,
           input_width, item->output_channels, output_height, output_width, scalar_ms,
           dispatched_ms, scalar_ms / dispatched_ms, packed_ms,
           dispatched_ms / packed_ms, checksum);
    ok = 1;

cleanup:
    free(packed_output);
    free(output);
    free(reference);
    free(bias);
    free(packed_weights);
    free(weights);
    free(input);
    return ok;
}

int main(int argc, char** argv) {
    static const benchmark_case cases[] = {
        {"stem-3x48", 3u, 48u, 48u, 1u, 0u},
        {"rec-node6-24x48", 24u, 48u, 24u, 2u, 0u},
        {"early-96x48", 96u, 48u, 24u, 2u, 0u},
        {"medium-det-64x128", 64u, 64u, 128u, 1u, 0u},
        {"medium-det-64x64", 64u, 64u, 64u, 1u, 0u},
        {"medium-det-64x32", 64u, 64u, 32u, 1u, 0u},
        {"medium-det-node8-128x256", 128u, 64u, 256u, 0u, 256u},
    };
    uint32_t target_width = 960u;
    uint32_t iterations = 3u;
    size_t index;
    if (argc > 3 || (argc >= 2 && !parse_positive_u32(argv[1], &target_width)) ||
        (argc >= 3 && !parse_positive_u32(argv[2], &iterations)) ||
        (target_width != 320u && target_width != 960u) || iterations > 100u) {
        fprintf(stderr, "usage: conv3x3-stride2-benchmark-driver [target-width=960] "
                        "[iterations=3]\n");
        return 2;
    }
    printf("{\"schema_version\":1,\"backend\":\"%s\",\"target_width\":%u,"
           "\"iterations\":%u,\"cases\":[",
           lw_simd_level_name(lw_detect_simd_level()), target_width, iterations);
    for (index = 0u; index < sizeof(cases) / sizeof(cases[0]); ++index) {
        if (!run_case(&cases[index], target_width, iterations, index == 0u)) {
            return 1;
        }
    }
    printf("]}\n");
    return 0;
}
