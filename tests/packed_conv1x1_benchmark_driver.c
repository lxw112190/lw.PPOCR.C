#include "cpu_features.h"
#include "lw_infer.h"
#include "packed_conv_internal.h"
#include "simd_kernels.h"

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
    uint32_t height;
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

static float max_abs_difference(const float* left, const float* right, uint64_t count) {
    uint64_t index;
    float maximum = 0.0f;
    for (index = 0u; index < count; ++index) {
        float difference = fabsf(left[(size_t)index] - right[(size_t)index]);
        if (!isfinite(difference)) {
            return INFINITY;
        }
        if (difference > maximum) {
            maximum = difference;
        }
    }
    return maximum;
}
typedef void (*packed_conv1x1_kernel)(
    const float* input, const float* packed_weights, const float* bias, float* output,
    const int32_t input_dimensions[4], const int32_t output_dimensions[4]);

static int measure_kernel(packed_conv1x1_kernel kernel, const float* input,
                          const float* packed_weights, const float* bias, float* output,
                          const int32_t input_dimensions[4], const int32_t output_dimensions[4],
                          uint32_t iterations, double* milliseconds) {
    uint32_t iteration;
    double started;
    double finished;
    if (kernel == NULL || milliseconds == NULL || iterations == 0u) {
        return 0;
    }
    started = monotonic_seconds();
    for (iteration = 0u; iteration < iterations; ++iteration) {
        kernel(input, packed_weights, bias, output, input_dimensions, output_dimensions);
    }
    finished = monotonic_seconds();
    if (started <= 0.0 || finished <= started) {
        return 0;
    }
    *milliseconds = (finished - started) * 1000.0 / (double)iterations;
    return *milliseconds > 0.0 && isfinite(*milliseconds);
}

static void sort_samples(double* samples, uint32_t count) {
    uint32_t i;
    for (i = 1u; i < count; ++i) {
        double value = samples[i];
        uint32_t j = i;
        while (j > 0u && samples[j - 1u] > value) {
            samples[j] = samples[j - 1u];
            --j;
        }
        samples[j] = value;
    }
}

static double sample_median(double* samples, uint32_t count) {
    sort_samples(samples, count);
    if ((count & 1u) != 0u) {
        return samples[count / 2u];
    }
    return (samples[count / 2u - 1u] + samples[count / 2u]) * 0.5;
}

static double sample_percentile90(double* samples, uint32_t count) {
    uint32_t index;
    sort_samples(samples, count);
    index = (count * 9u + 9u) / 10u;
    if (index == 0u) {
        index = 1u;
    }
    if (index > count) {
        index = count;
    }
    return samples[index - 1u];
}

static int run_case(const benchmark_case* item, uint32_t target_width, uint32_t iterations,
                    int first) {
    enum { ABBA_ROUNDS = 7 };
    uint32_t spatial_width = target_width / 4u;
    uint64_t spatial = (uint64_t)item->height * spatial_width;
    uint64_t input_count = (uint64_t)item->input_channels * spatial;
    uint64_t weight_count = (uint64_t)item->input_channels * item->output_channels;
    uint64_t output_count = (uint64_t)item->output_channels * spatial;
    uint64_t packed_count = 0u;
    size_t input_bytes;
    size_t weight_bytes;
    size_t output_bytes;
    size_t packed_bytes;
    float* input = NULL;
    float* weights = NULL;
    float* bias = NULL;
    float* packed = NULL;
    float* reference = NULL;
    float* output = NULL;
    int32_t input_dimensions[4] = {1, 0, 0, 0};
    int32_t output_dimensions[4] = {1, 0, 0, 0};
    double scalar_started;
    double scalar_finished;
    double scalar_ms;
    double avx2_samples[ABBA_ROUNDS];
    double fma_samples[ABBA_ROUNDS];
    double avx2_ms;
    double avx2_min;
    double avx2_max;
    double avx2_p90;
    double fma_ms = 0.0;
    double fma_min = 0.0;
    double fma_max = 0.0;
    double fma_p90 = 0.0;
    double fma8_samples[ABBA_ROUNDS];
    double fma8_ms = 0.0;
    uint64_t checksum;
    uint64_t fma_checksum = 0u;
    uint64_t fma8_checksum = 0u;
    float avx2_max_abs_error;
    float fma_max_abs_error = 0.0f;
    float fma_max_relative_error = 0.0f;
    float fma8_max_abs_error = 0.0f;
    uint32_t round;
    int has_fma = 0;
    int has_fma8 = 0;
    int ok = 0;
    const lw_cpu_capabilities capabilities = lw_get_cpu_capabilities();

    if (!lw_packed_conv1x1_weight_count(item->input_channels, item->output_channels,
                                        &packed_count) ||
        !allocation_size(input_count, sizeof(float), &input_bytes) ||
        !allocation_size(weight_count, sizeof(float), &weight_bytes) ||
        !allocation_size(output_count, sizeof(float), &output_bytes) ||
        !allocation_size(packed_count, sizeof(float), &packed_bytes) ||
        item->input_channels > INT32_MAX || item->output_channels > INT32_MAX ||
        item->height > INT32_MAX || spatial_width > INT32_MAX) {
        fprintf(stderr, "invalid benchmark geometry: %s\n", item->name);
        goto cleanup;
    }
    input = (float*)malloc(input_bytes);
    weights = (float*)malloc(weight_bytes);
    bias = (float*)malloc((size_t)item->output_channels * sizeof(float));
    packed = (float*)malloc(packed_bytes);
    reference = (float*)malloc(output_bytes);
    output = (float*)malloc(output_bytes);
    if (input == NULL || weights == NULL || bias == NULL || packed == NULL ||
        reference == NULL || output == NULL) {
        fprintf(stderr, "benchmark allocation failed: %s\n", item->name);
        goto cleanup;
    }
    fill_values(input, input_count, 17u + item->input_channels);
    fill_values(weights, weight_count, 31u + item->output_channels);
    fill_values(bias, item->output_channels, 47u + item->height);
    lw_pack_conv1x1_weights_f32(weights, item->input_channels, item->output_channels, packed);
    input_dimensions[1] = (int32_t)item->input_channels;
    input_dimensions[2] = (int32_t)item->height;
    input_dimensions[3] = (int32_t)spatial_width;
    output_dimensions[1] = (int32_t)item->output_channels;
    output_dimensions[2] = (int32_t)item->height;
    output_dimensions[3] = (int32_t)spatial_width;

    lw_scalar_packed_conv1x1_f32(input, packed, bias, reference, input_dimensions,
                                 output_dimensions);
    if (!measure_kernel(lw_avx2_packed_conv1x1_f32, input, packed, bias, output,
                        input_dimensions, output_dimensions, 1u, &scalar_ms)) {
        fprintf(stderr, "direct AVX2 warmup failed: %s\n", item->name);
        goto cleanup;
    }
    avx2_max_abs_error = max_abs_difference(reference, output, output_count);
    if (!isfinite(avx2_max_abs_error) || avx2_max_abs_error > 1.0e-3f) {
        fprintf(stderr, "direct AVX2 correctness check failed: %s\n", item->name);
        goto cleanup;
    }

    scalar_started = monotonic_seconds();
    for (round = 0u; round < iterations; ++round) {
        lw_scalar_packed_conv1x1_f32(input, packed, bias, reference, input_dimensions,
                                     output_dimensions);
    }
    scalar_finished = monotonic_seconds();
    if (scalar_started <= 0.0 || scalar_finished <= scalar_started) {
        fprintf(stderr, "scalar benchmark timer failed: %s\n", item->name);
        goto cleanup;
    }
    scalar_ms = (scalar_finished - scalar_started) * 1000.0 / (double)iterations;

    has_fma = capabilities.has_avx2_fma != 0;
#if defined(_M_X64) || defined(__x86_64__)
    has_fma8 = has_fma;
#endif
    if (has_fma) {
        for (round = 0u; round < ABBA_ROUNDS; ++round) {
            double avx2_first;
            double avx2_second;
            double fma_first;
            double fma_second;
            if ((round & 1u) == 0u) {
                if (!measure_kernel(lw_avx2_packed_conv1x1_f32, input, packed, bias, output,
                                    input_dimensions, output_dimensions, iterations, &avx2_first) ||
                    !measure_kernel(lw_avx2_fma_packed_conv1x1_f32, input, packed, bias, output,
                                    input_dimensions, output_dimensions, iterations, &fma_first) ||
                    !measure_kernel(lw_avx2_fma_packed_conv1x1_f32, input, packed, bias, output,
                                    input_dimensions, output_dimensions, iterations, &fma_second) ||
                    !measure_kernel(lw_avx2_packed_conv1x1_f32, input, packed, bias, output,
                                    input_dimensions, output_dimensions, iterations, &avx2_second)) {
                    fprintf(stderr, "direct AVX2/FMA timer failed: %s\n", item->name);
                    goto cleanup;
                }
            } else {
                if (!measure_kernel(lw_avx2_fma_packed_conv1x1_f32, input, packed, bias, output,
                                    input_dimensions, output_dimensions, iterations, &fma_first) ||
                    !measure_kernel(lw_avx2_packed_conv1x1_f32, input, packed, bias, output,
                                    input_dimensions, output_dimensions, iterations, &avx2_first) ||
                    !measure_kernel(lw_avx2_packed_conv1x1_f32, input, packed, bias, output,
                                    input_dimensions, output_dimensions, iterations, &avx2_second) ||
                    !measure_kernel(lw_avx2_fma_packed_conv1x1_f32, input, packed, bias, output,
                                    input_dimensions, output_dimensions, iterations, &fma_second)) {
                    fprintf(stderr, "direct FMA/AVX2 timer failed: %s\n", item->name);
                    goto cleanup;
                }
            }
            avx2_samples[round] = (avx2_first + avx2_second) * 0.5;
            fma_samples[round] = (fma_first + fma_second) * 0.5;
        }
        avx2_ms = sample_median(avx2_samples, ABBA_ROUNDS);
        fma_ms = sample_median(fma_samples, ABBA_ROUNDS);
        avx2_min = avx2_samples[0];
        avx2_max = avx2_samples[ABBA_ROUNDS - 1u];
        avx2_p90 = sample_percentile90(avx2_samples, ABBA_ROUNDS);
        fma_min = fma_samples[0];
        fma_max = fma_samples[ABBA_ROUNDS - 1u];
        fma_p90 = sample_percentile90(fma_samples, ABBA_ROUNDS);
        lw_avx2_fma_packed_conv1x1_f32(input, packed, bias, output,
                                       input_dimensions, output_dimensions);
        fma_max_abs_error = max_abs_difference(reference, output, output_count);
        fma_checksum = checksum_bytes(output, output_bytes);
        lw_avx2_packed_conv1x1_f32(input, packed, bias, output,
                                   input_dimensions, output_dimensions);
        if (has_fma8) {
            for (round = 0u; round < ABBA_ROUNDS; ++round) {
                double fma_first;
                double fma_second;
                double fma8_first;
                double fma8_second;
                if ((round & 1u) == 0u) {
                    if (!measure_kernel(lw_avx2_fma_packed_conv1x1_f32, input, packed, bias,
                                        output, input_dimensions, output_dimensions, iterations,
                                        &fma_first) ||
                        !measure_kernel(lw_avx2_fma_packed_conv1x1_8x8_f32, input, packed, bias,
                                        output, input_dimensions, output_dimensions, iterations,
                                        &fma8_first) ||
                        !measure_kernel(lw_avx2_fma_packed_conv1x1_8x8_f32, input, packed, bias,
                                        output, input_dimensions, output_dimensions, iterations,
                                        &fma8_second) ||
                        !measure_kernel(lw_avx2_fma_packed_conv1x1_f32, input, packed, bias,
                                        output, input_dimensions, output_dimensions, iterations,
                                        &fma_second)) {
                        fprintf(stderr, "direct FMA/8x8 timer failed: %s\n", item->name);
                        goto cleanup;
                    }
                } else {
                    if (!measure_kernel(lw_avx2_fma_packed_conv1x1_8x8_f32, input, packed, bias,
                                        output, input_dimensions, output_dimensions, iterations,
                                        &fma8_first) ||
                        !measure_kernel(lw_avx2_fma_packed_conv1x1_f32, input, packed, bias,
                                        output, input_dimensions, output_dimensions, iterations,
                                        &fma_first) ||
                        !measure_kernel(lw_avx2_fma_packed_conv1x1_f32, input, packed, bias,
                                        output, input_dimensions, output_dimensions, iterations,
                                        &fma_second) ||
                        !measure_kernel(lw_avx2_fma_packed_conv1x1_8x8_f32, input, packed, bias,
                                        output, input_dimensions, output_dimensions, iterations,
                                        &fma8_second)) {
                        fprintf(stderr, "direct 8x8/FMA timer failed: %s\n", item->name);
                        goto cleanup;
                    }
                }
                fma8_samples[round] = (fma8_first + fma8_second) * 0.5;
            }
            fma8_ms = sample_median(fma8_samples, ABBA_ROUNDS);
            lw_avx2_fma_packed_conv1x1_8x8_f32(input, packed, bias, output,
                                                input_dimensions, output_dimensions);
            fma8_max_abs_error = max_abs_difference(reference, output, output_count);
            fma8_checksum = checksum_bytes(output, output_bytes);
            if (!isfinite(fma8_max_abs_error) || fma8_max_abs_error > 1.0e-2f ||
                fma8_ms <= 0.0 || fma8_checksum != fma_checksum) {
                fprintf(stderr, "8x8 FMA candidate correctness check failed: %s\n", item->name);
                goto cleanup;
            }
            lw_avx2_packed_conv1x1_f32(input, packed, bias, output,
                                       input_dimensions, output_dimensions);
        }
    } else {
        if (!measure_kernel(lw_avx2_packed_conv1x1_f32, input, packed, bias, output,
                            input_dimensions, output_dimensions, iterations, &avx2_ms)) {
            fprintf(stderr, "direct AVX2 timer failed: %s\n", item->name);
            goto cleanup;
        }
        avx2_min = avx2_ms;
        avx2_max = avx2_ms;
        avx2_p90 = avx2_ms;
    }
    avx2_max_abs_error = max_abs_difference(reference, output, output_count);
    if (!isfinite(avx2_max_abs_error) || avx2_max_abs_error > 1.0e-3f ||
        scalar_ms <= 0.0 || avx2_ms <= 0.0) {
        fprintf(stderr, "direct AVX2 benchmark contract failed: %s\n", item->name);
        goto cleanup;
    }
    if (has_fma) {
        lw_avx2_fma_packed_conv1x1_f32(input, packed, bias, output,
                                       input_dimensions, output_dimensions);
        fma_max_abs_error = max_abs_difference(reference, output, output_count);
        fma_checksum = checksum_bytes(output, output_bytes);
        fma_max_relative_error = 0.0f;
        {
            uint64_t index;
            for (index = 0u; index < output_count; ++index) {
                float reference_value = reference[(size_t)index];
                float difference = fabsf(reference_value - output[(size_t)index]);
                float relative = difference / fmaxf(fabsf(reference_value), 1.0e-12f);
                if (!isfinite(relative)) {
                    fma_max_relative_error = INFINITY;
                    break;
                }
                if (relative > fma_max_relative_error) {
                    fma_max_relative_error = relative;
                }
            }
        }
        if (!isfinite(fma_max_abs_error) || fma_max_abs_error > 1.0e-2f ||
            !isfinite(fma_max_relative_error)) {
            fprintf(stderr, "FMA candidate correctness check failed: %s\n", item->name);
            goto cleanup;
        }
        lw_avx2_packed_conv1x1_f32(input, packed, bias, output,
                                   input_dimensions, output_dimensions);
    }
    checksum = checksum_bytes(output, output_bytes);
    printf("%s{\"name\":\"%s\",\"input_channels\":%u,\"output_channels\":%u,"
           "\"height\":%u,\"width\":%u,\"batch\":1,\"scalar_ms\":%.6f,"
           "\"avx2_available\":%s,\"avx2_ms\":%.6f,\"avx2_min_ms\":%.6f,"
           "\"avx2_max_ms\":%.6f,\"avx2_p90_ms\":%.6f,\"avx2_speedup\":%.6f,"
           "\"dispatched_ms\":%.6f,\"speedup\":%.6f,\"checksum\":\"0x%016" PRIx64 "\"",
           first ? "" : ",", item->name, item->input_channels, item->output_channels,
           item->height, spatial_width, scalar_ms, capabilities.simd == LW_SIMD_LEVEL_AVX2 ? "true" : "false",
           avx2_ms, avx2_min, avx2_max, avx2_p90, scalar_ms / avx2_ms,
           avx2_ms, scalar_ms / avx2_ms, checksum);
    if (has_fma) {
        printf(",\"fma_ms\":%.6f,\"fma_min_ms\":%.6f,\"fma_max_ms\":%.6f,"
               "\"fma_p90_ms\":%.6f,\"fma_speedup\":%.6f,\"fma_vs_avx2\":%.6f,"
               "\"fma_max_abs_error\":%.9g,\"fma_max_relative_error\":%.9g,"
               "\"fma_checksum\":\"0x%016" PRIx64 "\"",
               fma_ms, fma_min, fma_max, fma_p90, scalar_ms / fma_ms,
               avx2_ms / fma_ms, (double)fma_max_abs_error,
               (double)fma_max_relative_error, fma_checksum);
        if (has_fma8) {
            printf(",\"fma8_ms\":%.6f,\"fma8_vs_fma\":%.6f,"
                   "\"fma8_max_abs_error\":%.9g,\"fma8_checksum\":\"0x%016" PRIx64 "\"",
                   fma8_ms, fma_ms / fma8_ms, (double)fma8_max_abs_error, fma8_checksum);
        }
    }
    printf("}");
    ok = 1;

cleanup:
    free(output);
    free(reference);
    free(packed);
    free(bias);
    free(weights);
    free(input);
    return ok;
}
int main(int argc, char** argv) {
    static const benchmark_case cases[] = {
        {"early-96x192", 96u, 192u, 12u},
        {"rec-48x96-h12", 48u, 96u, 12u},
        {"rec-96x48-h12", 96u, 48u, 12u},
        {"rec-96x192-h6", 96u, 192u, 6u},
        {"rec-192x96-h6", 192u, 96u, 6u},
        {"rec-160x320-h3", 160u, 320u, 3u},
        {"rec-320x160-h3", 320u, 160u, 3u},
        {"middle-192x384", 192u, 384u, 6u},
        {"middle-384x192", 384u, 192u, 6u},
        {"late-384x768", 384u, 768u, 3u},
        {"late-768x384", 768u, 384u, 3u},
        {"medium-512x1024", 512u, 1024u, 6u},
        {"medium-1024x512", 1024u, 512u, 6u},
        {"medium-1536x768", 1536u, 768u, 3u},
    };
    uint32_t target_width = 960u;
    uint32_t iterations = 3u;
    size_t index;
    if (argc > 3 || (argc >= 2 && !parse_positive_u32(argv[1], &target_width)) ||
        (argc >= 3 && !parse_positive_u32(argv[2], &iterations)) ||
        (target_width != 320u && target_width != 960u) || iterations > 100u) {
        fprintf(stderr, "usage: packed-conv1x1-benchmark-driver [target-width=960] "
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
