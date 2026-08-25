#if !defined(_WIN32) && !defined(__APPLE__)
#  define _POSIX_C_SOURCE 200809L
#endif

#include "lw_infer.h"
#include "ppm_image.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  include <psapi.h>
#  include <shellapi.h>
#elif defined(__APPLE__)
#  include <mach/mach_time.h>
#  include <sys/resource.h>
#else
#  include <sys/resource.h>
#  include <time.h>
#  include <unistd.h>
#endif

#define LW_BENCHMARK_MAX_ITERATIONS 10000u

typedef struct lw_process_memory {
    uint64_t current_rss_bytes;
    uint64_t peak_rss_bytes;
} lw_process_memory;

static double monotonic_seconds(void) {
#if defined(_WIN32)
    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    if (!QueryPerformanceFrequency(&frequency) ||
        !QueryPerformanceCounter(&counter) || frequency.QuadPart <= 0) {
        return 0.0;
    }
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#elif defined(__APPLE__)
    mach_timebase_info_data_t timebase;
    uint64_t ticks = mach_absolute_time();
    if (mach_timebase_info(&timebase) != KERN_SUCCESS || timebase.denom == 0u) {
        return 0.0;
    }
    return (double)ticks * (double)timebase.numer /
        (double)timebase.denom / 1000000000.0;
#else
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0.0;
    }
    return (double)value.tv_sec + (double)value.tv_nsec / 1000000000.0;
#endif
}

static lw_process_memory process_memory(void) {
    lw_process_memory result;
    memset(&result, 0, sizeof(result));
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters;
    memset(&counters, 0, sizeof(counters));
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(
            GetCurrentProcess(), &counters, sizeof(counters))) {
        result.current_rss_bytes = (uint64_t)counters.WorkingSetSize;
        result.peak_rss_bytes = (uint64_t)counters.PeakWorkingSetSize;
    }
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0 && usage.ru_maxrss > 0) {
#  if defined(__APPLE__)
        result.peak_rss_bytes = (uint64_t)usage.ru_maxrss;
#  else
        result.peak_rss_bytes = (uint64_t)usage.ru_maxrss * 1024u;
#  endif
    }
#  if defined(__linux__)
    {
        FILE* statm = fopen("/proc/self/statm", "r");
        unsigned long long total_pages = 0u;
        unsigned long long resident_pages = 0u;
        long page_size = sysconf(_SC_PAGESIZE);
        if (statm != NULL && page_size > 0 &&
            fscanf(statm, "%llu %llu", &total_pages, &resident_pages) == 2) {
            (void)total_pages;
            result.current_rss_bytes =
                (uint64_t)resident_pages * (uint64_t)page_size;
        }
        if (statm != NULL) {
            fclose(statm);
        }
    }
#  endif
#endif
    return result;
}

static uint64_t file_size_utf8(const char* path_utf8) {
    FILE* file = lw_example_open_read_utf8(path_utf8);
    long length;
    if (file == NULL || fseek(file, 0, SEEK_END) != 0) {
        if (file != NULL) {
            fclose(file);
        }
        return 0u;
    }
    length = ftell(file);
    fclose(file);
    return length < 0 ? 0u : (uint64_t)length;
}

static int compare_double(const void* left, const void* right) {
    double a = *(const double*)left;
    double b = *(const double*)right;
    return (a > b) - (a < b);
}

static void print_json_string(const char* text) {
    const unsigned char* cursor = (const unsigned char*)text;
    putchar('"');
    while (*cursor != 0u) {
        unsigned char value = *cursor++;
        if (value == '"' || value == '\\') {
            putchar('\\');
            putchar((int)value);
        } else if (value < 0x20u) {
            printf("\\u%04x", (unsigned int)value);
        } else {
            putchar((int)value);
        }
    }
    putchar('"');
}

static int recognize_once(
    lw_recognizer* recognizer,
    const lw_example_ppm_image* image,
    char* text,
    uint64_t text_capacity,
    lw_recognition_result* recognition) {
    lw_error error;
    lw_status status;
    lw_recognition_result_init(recognition);
    lw_error_init(&error);
    status = lw_recognizer_recognize_bgr_u8(
        recognizer, image->pixels, image->byte_count,
        image->width, image->height, image->width * 3u,
        text, text_capacity, recognition, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "recognition failed: %s: %s\n",
                lw_status_string(status), error.message);
        return 0;
    }
    return 1;
}

static int benchmark_main(int argc, char** argv) {
    lw_example_ppm_image image;
    lw_recognizer* recognizer = NULL;
    lw_recognizer_info info;
    lw_recognition_result recognition;
    lw_error error;
    lw_status status;
    uint32_t warmup_count = 2u;
    uint32_t iteration_count = 10u;
    uint32_t index;
    uint64_t model_bytes;
    uint64_t io_bytes;
    char* text = NULL;
    char* reference_text = NULL;
    float reference_score = 0.0f;
    uint32_t reference_count = 0u;
    double* latencies = NULL;
    double* sorted = NULL;
    double create_start;
    double create_end;
    double sum = 0.0;
    double median;
    double p95;
    lw_process_memory memory_after_warmup;
    lw_process_memory memory_final;
    int64_t rss_growth;
    int result = 1;
    memset(&image, 0, sizeof(image));
    if (argc < 4 || argc > 6 ||
        (argc >= 5 && !lw_example_parse_positive_u32(argv[4], &warmup_count)) ||
        (argc >= 6 && !lw_example_parse_positive_u32(argv[5], &iteration_count)) ||
        warmup_count > LW_BENCHMARK_MAX_ITERATIONS ||
        iteration_count > LW_BENCHMARK_MAX_ITERATIONS) {
        fprintf(stderr,
                "usage: lw-rec-benchmark <rec.lwm> <dictionary.txt> <image.ppm> "
                "[warmup=2] [iterations=10]\n");
        return 2;
    }
    if (!lw_example_ppm_image_load_bgr(argv[3], &image) ||
        image.width > UINT32_MAX / 3u) {
        fprintf(stderr, "invalid P6 PPM image: %s\n", argv[3]);
        goto cleanup;
    }
    model_bytes = file_size_utf8(argv[1]);
    create_start = monotonic_seconds();
    lw_error_init(&error);
    status = lw_recognizer_create(argv[1], argv[2], NULL, &recognizer, &error);
    create_end = monotonic_seconds();
    if (status != LW_STATUS_OK || create_start <= 0.0 || create_end < create_start) {
        fprintf(stderr, "create failed: %s: %s\n",
                lw_status_string(status), error.message);
        goto cleanup;
    }
    lw_recognizer_info_init(&info);
    if (lw_recognizer_get_info(recognizer, &info) != LW_STATUS_OK ||
        info.max_text_capacity == 0u || info.max_text_capacity > SIZE_MAX) {
        fprintf(stderr, "unable to query recognizer information\n");
        goto cleanup;
    }
    text = (char*)malloc((size_t)info.max_text_capacity);
    reference_text = (char*)malloc((size_t)info.max_text_capacity);
    latencies = (double*)malloc((size_t)iteration_count * sizeof(*latencies));
    sorted = (double*)malloc((size_t)iteration_count * sizeof(*sorted));
    if (text == NULL || reference_text == NULL || latencies == NULL || sorted == NULL) {
        fprintf(stderr, "benchmark allocation failed\n");
        goto cleanup;
    }
    for (index = 0u; index < warmup_count; ++index) {
        if (!recognize_once(
                recognizer, &image, text, info.max_text_capacity, &recognition)) {
            goto cleanup;
        }
        if (index == 0u) {
            memcpy(reference_text, text, strlen(text) + 1u);
            reference_score = recognition.score;
            reference_count = recognition.emitted_count;
        } else if (strcmp(reference_text, text) != 0 ||
                   memcmp(&reference_score, &recognition.score,
                          sizeof(reference_score)) != 0 ||
                   reference_count != recognition.emitted_count) {
            fprintf(stderr, "warm-up result is not deterministic\n");
            goto cleanup;
        }
    }
    memory_after_warmup = process_memory();
    for (index = 0u; index < iteration_count; ++index) {
        double start = monotonic_seconds();
        double end;
        if (start <= 0.0 || !recognize_once(
                recognizer, &image, text, info.max_text_capacity, &recognition)) {
            goto cleanup;
        }
        end = monotonic_seconds();
        if (end < start || strcmp(reference_text, text) != 0 ||
            memcmp(&reference_score, &recognition.score,
                   sizeof(reference_score)) != 0 ||
            reference_count != recognition.emitted_count) {
            fprintf(stderr, "timed result is invalid or non-deterministic\n");
            goto cleanup;
        }
        latencies[index] = (end - start) * 1000.0;
        sum += latencies[index];
    }
    memory_final = process_memory();
    memcpy(sorted, latencies, (size_t)iteration_count * sizeof(*sorted));
    qsort(sorted, iteration_count, sizeof(*sorted), compare_double);
    if ((iteration_count & 1u) != 0u) {
        median = sorted[iteration_count / 2u];
    } else {
        median = (sorted[iteration_count / 2u - 1u] +
                  sorted[iteration_count / 2u]) / 2.0;
    }
    index = (95u * iteration_count + 99u) / 100u - 1u;
    p95 = sorted[index];
    io_bytes = ((uint64_t)3u * info.input_height * info.target_width +
                (uint64_t)info.time_steps * info.class_count) * sizeof(float);
    rss_growth = (int64_t)memory_final.current_rss_bytes -
        (int64_t)memory_after_warmup.current_rss_bytes;
    printf("{");
    printf("\"schema_version\":1,\"backend\":\"scalar\",");
    printf("\"text\":");
    print_json_string(reference_text);
    printf(",\"score\":%.9g,\"characters\":%u,", reference_score,
           reference_count);
    printf("\"image_width\":%u,\"image_height\":%u,", image.width, image.height);
    printf("\"target_width\":%u,\"time_steps\":%u,", info.target_width,
           info.time_steps);
    printf("\"model_file_bytes\":%llu,\"workspace_bytes\":%llu,",
           (unsigned long long)model_bytes,
           (unsigned long long)info.workspace_size);
    printf("\"preallocated_io_bytes\":%llu,",
           (unsigned long long)io_bytes);
    printf("\"recognizer_create_ms\":%.6f,\"warmup\":%u,\"iterations\":%u,",
           (create_end - create_start) * 1000.0, warmup_count, iteration_count);
    printf("\"latency_ms\":{");
    printf("\"mean\":%.6f,\"median\":%.6f,\"p95\":%.6f,",
           sum / iteration_count, median, p95);
    printf("\"min\":%.6f,\"max\":%.6f},", sorted[0],
           sorted[iteration_count - 1u]);
    printf("\"throughput_per_second\":%.6f,", 1000.0 / (sum / iteration_count));
    printf("\"rss_after_warmup_bytes\":%llu,\"rss_final_bytes\":%llu,",
           (unsigned long long)memory_after_warmup.current_rss_bytes,
           (unsigned long long)memory_final.current_rss_bytes);
    printf("\"rss_growth_bytes\":%lld,\"peak_rss_bytes\":%llu}\n",
           (long long)rss_growth,
           (unsigned long long)memory_final.peak_rss_bytes);
    result = 0;
cleanup:
    free(sorted);
    free(latencies);
    free(reference_text);
    free(text);
    lw_recognizer_free(recognizer);
    lw_example_ppm_image_free(&image);
    return result;
}

#if defined(_WIN32)
int main(void) {
    wchar_t** wide_argv;
    char** utf8_argv;
    int argc;
    int index;
    int result = 2;
    SetConsoleOutputCP(CP_UTF8);
    wide_argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (wide_argv == NULL || argc <= 0) {
        return 2;
    }
    utf8_argv = (char**)calloc((size_t)argc, sizeof(*utf8_argv));
    if (utf8_argv == NULL) {
        LocalFree(wide_argv);
        return 2;
    }
    for (index = 0; index < argc; ++index) {
        int bytes = WideCharToMultiByte(
            CP_UTF8, WC_ERR_INVALID_CHARS, wide_argv[index], -1,
            NULL, 0, NULL, NULL);
        if (bytes <= 0) {
            goto cleanup;
        }
        utf8_argv[index] = (char*)malloc((size_t)bytes);
        if (utf8_argv[index] == NULL ||
            WideCharToMultiByte(
                CP_UTF8, WC_ERR_INVALID_CHARS, wide_argv[index], -1,
                utf8_argv[index], bytes, NULL, NULL) <= 0) {
            goto cleanup;
        }
    }
    result = benchmark_main(argc, utf8_argv);
cleanup:
    for (index = 0; index < argc; ++index) {
        free(utf8_argv[index]);
    }
    free(utf8_argv);
    LocalFree(wide_argv);
    return result;
}
#else
int main(int argc, char** argv) {
    return benchmark_main(argc, argv);
}
#endif
