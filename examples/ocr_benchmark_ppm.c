#if !defined(_WIN32) && !defined(__APPLE__)
#  define _POSIX_C_SOURCE 200809L
#endif

#include "lw_infer.h"

/* Reusable-handle benchmark for DET and the complete DET/CLS/REC pipeline. */
#include "cpu_features.h"
#include "ppm_image.h"

#include <errno.h>
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

#define LW_OCR_BENCHMARK_MAX_ITERATIONS 1000u

typedef struct lw_process_memory {
    uint64_t current_rss_bytes;
    uint64_t peak_rss_bytes;
} lw_process_memory;

static double monotonic_seconds(void) {
#if defined(_WIN32)
    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    if (!QueryPerformanceFrequency(&frequency) || !QueryPerformanceCounter(&counter) ||
        frequency.QuadPart <= 0) {
        return 0.0;
    }
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#elif defined(__APPLE__)
    mach_timebase_info_data_t timebase;
    uint64_t ticks = mach_absolute_time();
    if (mach_timebase_info(&timebase) != KERN_SUCCESS || timebase.denom == 0u) {
        return 0.0;
    }
    return (double)ticks * (double)timebase.numer / (double)timebase.denom / 1000000000.0;
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
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
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
            result.current_rss_bytes = (uint64_t)resident_pages * (uint64_t)page_size;
        }
        if (statm != NULL) {
            fclose(statm);
        }
    }
#  endif
#endif
    return result;
}

static int parse_positive_u32(const char* text, uint32_t* value) {
    char* end = NULL;
    unsigned long parsed;
    errno = 0;
    parsed = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0u || parsed > UINT32_MAX) {
        return 0;
    }
    *value = (uint32_t)parsed;
    return 1;
}

static int compare_double(const void* left, const void* right) {
    double a = *(const double*)left;
    double b = *(const double*)right;
    return (a > b) - (a < b);
}

static int allocation_fits(uint64_t count, size_t element_size) {
    return element_size != 0u && count <= (uint64_t)SIZE_MAX / (uint64_t)element_size;
}

static double percentile(double* values, uint32_t count, uint32_t percentage) {
    uint32_t index;
    qsort(values, count, sizeof(*values), compare_double);
    index = (percentage * count + 99u) / 100u - 1u;
    return values[index];
}

static int run_detector(lw_detector* detector, const lw_example_ppm_image* image,
                        lw_detection_box* boxes, uint32_t box_capacity,
                        lw_detection_result* result) {
    lw_error error;
    lw_status status;
    lw_detection_result_init(result);
    lw_error_init(&error);
    status = lw_detector_detect_bgr_u8(detector, image->pixels, image->byte_count, image->width,
                                       image->height, image->width * 3u, boxes, box_capacity,
                                       result, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "detection failed: %s: %s\n", lw_status_string(status), error.message);
        return 0;
    }
    return 1;
}

static int run_ocr(lw_ocr* ocr, const lw_example_ppm_image* image, lw_ocr_line* lines,
                   uint32_t line_capacity, char* text, uint64_t text_capacity,
                   lw_ocr_result* result) {
    lw_error error;
    lw_status status;
    lw_ocr_result_init(result);
    lw_error_init(&error);
    status = lw_ocr_run_bgr_u8(ocr, image->pixels, image->byte_count, image->width, image->height,
                               image->width * 3u, lines, line_capacity, text, text_capacity, result,
                               &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "OCR failed: %s: %s\n", lw_status_string(status), error.message);
        return 0;
    }
    return 1;
}

static int benchmark_main(int argc, char** argv) {
    lw_example_ppm_image image;
    lw_detector* detector = NULL;
    lw_detector_info detector_info;
    lw_detection_box* boxes = NULL;
    lw_detection_result detection;
    lw_ocr* ocr = NULL;
    lw_ocr_info ocr_info;
    lw_ocr_line* lines = NULL;
    lw_ocr_result ocr_result;
    char* text = NULL;
    char* reference_text = NULL;
    double* detector_times = NULL;
    double* ocr_times = NULL;
    double* sorted = NULL;
    uint32_t warmup_count = 2u;
    uint32_t iteration_count = 10u;
    uint32_t worker_count = 0u;
    uint32_t rec_target_width = 320u;
    uint32_t index;
    uint32_t reference_line_count = 0u;
    uint64_t reference_text_capacity = 0u;
    double detector_sum = 0.0;
    double ocr_sum = 0.0;
    lw_process_memory memory_after_warmup;
    lw_process_memory memory_final;
    lw_error error;
    lw_status status;
    int exit_code = 1;

    memset(&image, 0, sizeof(image));
    if (argc < 6 || argc > 10 || (argc >= 7 && !parse_positive_u32(argv[6], &warmup_count)) ||
        (argc >= 8 && !parse_positive_u32(argv[7], &iteration_count)) ||
        (argc >= 9 && !parse_positive_u32(argv[8], &worker_count)) ||
        (argc >= 10 && !parse_positive_u32(argv[9], &rec_target_width)) ||
        warmup_count > LW_OCR_BENCHMARK_MAX_ITERATIONS ||
        iteration_count > LW_OCR_BENCHMARK_MAX_ITERATIONS) {
        fprintf(stderr, "usage: lw-ocr-benchmark <det.lwm> <cls.lwm> <rec.lwm> "
                        "<dictionary.txt> <image.ppm> [warmup=2] [iterations=10] "
                        "[workers=platform-default] [rec-target-width=320]\n");
        return 2;
    }
    if (!lw_example_ppm_image_load_bgr(argv[5], &image) || image.width > UINT32_MAX / 3u) {
        fprintf(stderr, "invalid P6 PPM image: %s\n", argv[5]);
        goto cleanup;
    }
    lw_error_init(&error);
    status = lw_detector_create(argv[1], NULL, &detector, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "detector create failed: %s: %s\n", lw_status_string(status),
                error.message);
        goto cleanup;
    }
    lw_detector_info_init(&detector_info);
    if (lw_detector_get_info(detector, &detector_info) != LW_STATUS_OK ||
        detector_info.max_candidates == 0u ||
        !allocation_fits(detector_info.max_candidates, sizeof(*boxes))) {
        fprintf(stderr, "unable to query detector capacity\n");
        goto cleanup;
    }
    boxes = (lw_detection_box*)calloc(detector_info.max_candidates, sizeof(*boxes));
    {
        lw_ocr_options options;
        lw_ocr_options_init(&options);
        if (worker_count != 0u) {
            options.worker_count = worker_count;
        }
        options.recognizer.target_width = rec_target_width;
        lw_error_init(&error);
        status = lw_ocr_create(argv[1], argv[2], argv[3], argv[4], &options, &ocr, &error);
    }
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "OCR create failed: %s: %s\n", lw_status_string(status), error.message);
        goto cleanup;
    }
    lw_ocr_info_init(&ocr_info);
    if (lw_ocr_get_info(ocr, &ocr_info) != LW_STATUS_OK || ocr_info.max_line_capacity == 0u ||
        !allocation_fits(ocr_info.max_line_capacity, sizeof(*lines)) ||
        ocr_info.max_text_capacity == 0u || ocr_info.max_text_capacity > SIZE_MAX) {
        fprintf(stderr, "unable to query OCR capacity\n");
        goto cleanup;
    }
    lines = (lw_ocr_line*)calloc(ocr_info.max_line_capacity, sizeof(*lines));
    text = (char*)malloc((size_t)ocr_info.max_text_capacity);
    reference_text = (char*)malloc((size_t)ocr_info.max_text_capacity);
    detector_times = (double*)malloc((size_t)iteration_count * sizeof(*detector_times));
    ocr_times = (double*)malloc((size_t)iteration_count * sizeof(*ocr_times));
    sorted = (double*)malloc((size_t)iteration_count * sizeof(*sorted));
    if (boxes == NULL || lines == NULL || text == NULL || reference_text == NULL ||
        detector_times == NULL || ocr_times == NULL || sorted == NULL) {
        fprintf(stderr, "benchmark allocation failed\n");
        goto cleanup;
    }
    for (index = 0u; index < warmup_count; ++index) {
        if (!run_detector(detector, &image, boxes, detector_info.max_candidates, &detection) ||
            !run_ocr(ocr, &image, lines, ocr_info.max_line_capacity, text,
                     ocr_info.max_text_capacity, &ocr_result)) {
            goto cleanup;
        }
        if (index == 0u) {
            reference_line_count = ocr_result.line_count;
            reference_text_capacity = ocr_result.required_text_capacity;
            memcpy(reference_text, text, (size_t)reference_text_capacity);
        } else if (reference_line_count != ocr_result.line_count ||
                   reference_text_capacity != ocr_result.required_text_capacity ||
                   memcmp(reference_text, text, (size_t)reference_text_capacity) != 0) {
            fprintf(stderr, "warm-up OCR result is not deterministic\n");
            goto cleanup;
        }
    }
    memory_after_warmup = process_memory();
    for (index = 0u; index < iteration_count; ++index) {
        double start = monotonic_seconds();
        double end;
        if (start <= 0.0 ||
            !run_detector(detector, &image, boxes, detector_info.max_candidates, &detection)) {
            goto cleanup;
        }
        end = monotonic_seconds();
        detector_times[index] = (end - start) * 1000.0;
        detector_sum += detector_times[index];

        start = monotonic_seconds();
        if (start <= 0.0 || !run_ocr(ocr, &image, lines, ocr_info.max_line_capacity, text,
                                     ocr_info.max_text_capacity, &ocr_result)) {
            goto cleanup;
        }
        end = monotonic_seconds();
        if (reference_line_count != ocr_result.line_count ||
            reference_text_capacity != ocr_result.required_text_capacity ||
            memcmp(reference_text, text, (size_t)reference_text_capacity) != 0) {
            fprintf(stderr, "timed OCR result is not deterministic\n");
            goto cleanup;
        }
        ocr_times[index] = (end - start) * 1000.0;
        ocr_sum += ocr_times[index];
    }
    memcpy(sorted, detector_times, (size_t)iteration_count * sizeof(*sorted));
    memory_final = process_memory();
    printf("{\"schema_version\":1,\"backend\":\"%s\",", lw_simd_level_name(lw_detect_simd_level()));
    printf("\"image_width\":%u,\"image_height\":%u,\"lines\":%u,\"workers\":%u,"
           "\"rec_target_width\":%u,",
           image.width, image.height, reference_line_count, ocr_info.worker_count,
           rec_target_width);
    printf("\"warmup\":%u,\"iterations\":%u,", warmup_count, iteration_count);
    printf("\"detector_ms\":{\"mean\":%.6f,\"p95\":%.6f},", detector_sum / iteration_count,
           percentile(sorted, iteration_count, 95u));
    memcpy(sorted, ocr_times, (size_t)iteration_count * sizeof(*sorted));
    printf("\"ocr_ms\":{\"mean\":%.6f,\"p95\":%.6f},", ocr_sum / iteration_count,
           percentile(sorted, iteration_count, 95u));
    printf("\"after_detector_ms\":%.6f,\"throughput_per_second\":%.6f,",
           (ocr_sum - detector_sum) / iteration_count, 1000.0 / (ocr_sum / iteration_count));
    printf("\"rss_after_warmup_bytes\":%llu,\"rss_final_bytes\":%llu,"
           "\"peak_rss_bytes\":%llu}\n",
           (unsigned long long)memory_after_warmup.current_rss_bytes,
           (unsigned long long)memory_final.current_rss_bytes,
           (unsigned long long)memory_final.peak_rss_bytes);
    exit_code = 0;

cleanup:
    free(sorted);
    free(ocr_times);
    free(detector_times);
    free(reference_text);
    free(text);
    free(lines);
    lw_ocr_free(ocr);
    free(boxes);
    lw_detector_free(detector);
    lw_example_ppm_image_free(&image);
    return exit_code;
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
        int bytes = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide_argv[index], -1, NULL,
                                        0, NULL, NULL);
        if (bytes <= 0) {
            goto cleanup;
        }
        utf8_argv[index] = (char*)malloc((size_t)bytes);
        if (utf8_argv[index] == NULL ||
            WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide_argv[index], -1,
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
