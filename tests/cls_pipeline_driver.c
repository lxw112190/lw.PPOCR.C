#include "cls_internal.h"
#include "lw_infer.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <shellapi.h>
#endif

static int parse_u32(const char* text, uint32_t* value) {
    char* end = NULL;
    unsigned long parsed;
    errno = 0;
    parsed = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed > UINT32_MAX) {
        return 0;
    }
    *value = (uint32_t)parsed;
    return 1;
}

static int read_file(const char* path, uint8_t** bytes, size_t* byte_count) {
    FILE* file = fopen(path, "rb");
    long length;
    uint8_t* data;
    size_t count;
    *bytes = NULL;
    *byte_count = 0u;
    if (file == NULL || fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        if (file != NULL) {
            fclose(file);
        }
        return 0;
    }
    data = (uint8_t*)malloc(length == 0 ? 1u : (size_t)length);
    if (data == NULL) {
        fclose(file);
        return 0;
    }
    count = fread(data, 1u, (size_t)length, file);
    if (fclose(file) != 0 || count != (size_t)length) {
        free(data);
        return 0;
    }
    *bytes = data;
    *byte_count = count;
    return 1;
}

static int write_file(const char* path, const void* bytes, size_t byte_count) {
    FILE* file = fopen(path, "wb");
    size_t count;
    if (file == NULL) {
        return 0;
    }
    count = fwrite(bytes, 1u, byte_count, file);
    return fclose(file) == 0 && count == byte_count;
}

static int run_preprocess(int argc, char** argv) {
    const uint64_t output_count = UINT64_C(3) * LW_CLS_INPUT_HEIGHT * LW_CLS_INPUT_WIDTH;
    uint8_t* source = NULL;
    size_t source_bytes = 0u;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t resized_width;
    float* output = NULL;
    lw_status status;
    int result = 1;
    if (argc != 7 || !parse_u32(argv[3], &width) || !parse_u32(argv[4], &height) ||
        !parse_u32(argv[5], &stride) || !read_file(argv[2], &source, &source_bytes)) {
        fprintf(stderr, "invalid preprocess arguments or source file\n");
        return 2;
    }
    if (output_count > SIZE_MAX / sizeof(*output) ||
        (output = (float*)malloc((size_t)output_count * sizeof(*output))) == NULL) {
        fprintf(stderr, "preprocess output allocation failed\n");
        goto cleanup;
    }
    status = lw_cls_preprocess_bgr_u8(source, source_bytes, width, height, stride, output,
                                      output_count - 1u, &resized_width);
    if (status != LW_STATUS_INVALID_SHAPE) {
        fprintf(stderr, "preprocessor accepted an incorrect output element count\n");
        goto cleanup;
    }
    status = lw_cls_preprocess_bgr_u8(source, source_bytes, width, height, stride, output,
                                      output_count, &resized_width);
    if (status != LW_STATUS_OK ||
        !write_file(argv[6], output, (size_t)output_count * sizeof(*output))) {
        fprintf(stderr, "preprocess failed: %s\n", lw_status_string(status));
        goto cleanup;
    }
    printf("resized_width=%u elements=%llu\n", resized_width, (unsigned long long)output_count);
    result = 0;

cleanup:
    free(output);
    free(source);
    return result;
}

static int run_pipeline(int argc, char** argv) {
    lw_classifier* classifier = NULL;
    lw_classification_result classification;
    lw_error error;
    uint8_t* source = NULL;
    size_t source_bytes = 0u;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    lw_status status;
    int result = 1;
    if (argc != 7 || !parse_u32(argv[4], &width) || !parse_u32(argv[5], &height) ||
        !parse_u32(argv[6], &stride) || !read_file(argv[3], &source, &source_bytes)) {
        fprintf(stderr, "invalid pipeline arguments or source file\n");
        return 2;
    }
    lw_error_init(&error);
    status = lw_classifier_create(argv[2], NULL, &classifier, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "classifier create failed: %s: %s\n", lw_status_string(status),
                error.message);
        goto cleanup;
    }
    lw_classification_result_init(&classification);
    lw_error_init(&error);
    status = lw_classifier_classify_bgr_u8(classifier, source, source_bytes, width, height, stride,
                                           &classification, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "classification failed: %s: %s\n", lw_status_string(status), error.message);
        goto cleanup;
    }
    printf("label=%u score=%.9g orientation=%u resized_width=%u\n", classification.label,
           (double)classification.score, classification.orientation_degrees,
           classification.resized_width);
    result = 0;

cleanup:
    lw_classifier_free(classifier);
    free(source);
    return result;
}

static int driver_main(int argc, char** argv) {
    if (argc < 2) {
        return 2;
    }
    if (strcmp(argv[1], "preprocess") == 0) {
        return run_preprocess(argc, argv);
    }
    if (strcmp(argv[1], "pipeline") == 0) {
        return run_pipeline(argc, argv);
    }
    fprintf(stderr, "unknown mode\n");
    return 2;
}

#if defined(_WIN32)
int main(void) {
    wchar_t** wide_argv;
    char** utf8_argv;
    int argc;
    int index;
    int result = 2;
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
        int byte_count = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide_argv[index], -1,
                                             NULL, 0, NULL, NULL);
        if (byte_count <= 0) {
            goto cleanup;
        }
        utf8_argv[index] = (char*)malloc((size_t)byte_count);
        if (utf8_argv[index] == NULL ||
            WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide_argv[index], -1,
                                utf8_argv[index], byte_count, NULL, NULL) <= 0) {
            goto cleanup;
        }
    }
    result = driver_main(argc, utf8_argv);
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
    return driver_main(argc, argv);
}
#endif
