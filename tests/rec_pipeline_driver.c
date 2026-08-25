#include "lw_infer.h"
#include "rec_internal.h"

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
    size_t count;
    uint8_t* data;
    *bytes = NULL;
    *byte_count = 0u;
    if (file == NULL || fseek(file, 0, SEEK_END) != 0 ||
        (length = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0) {
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
    uint8_t* source = NULL;
    size_t source_bytes = 0u;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t target_width;
    uint32_t resized_width;
    uint64_t output_count;
    float* output = NULL;
    lw_status status;
    int result = 1;
    if (argc != 8 || !parse_u32(argv[3], &width) || !parse_u32(argv[4], &height) ||
        !parse_u32(argv[5], &stride) || !parse_u32(argv[6], &target_width) ||
        !read_file(argv[2], &source, &source_bytes)) {
        fprintf(stderr, "invalid preprocess arguments or source file\n");
        return 2;
    }
    output_count = (uint64_t)3u * LW_REC_INPUT_HEIGHT * target_width;
    if (output_count > SIZE_MAX / sizeof(*output) ||
        (output = (float*)malloc((size_t)output_count * sizeof(*output))) == NULL) {
        fprintf(stderr, "preprocess output allocation failed\n");
        goto cleanup;
    }
    status = lw_rec_preprocess_bgr_u8(
        source, source_bytes, width, height, stride, target_width,
        output, output_count - 1u, &resized_width);
    if (status != LW_STATUS_INVALID_SHAPE) {
        fprintf(stderr, "preprocessor accepted an incorrect output element count\n");
        goto cleanup;
    }
    status = lw_rec_preprocess_bgr_u8(
        source, source_bytes, width, height, stride, target_width,
        output, output_count, &resized_width);
    if (status != LW_STATUS_OK ||
        !write_file(argv[7], output, (size_t)output_count * sizeof(*output))) {
        fprintf(stderr, "preprocess failed: %s\n", lw_status_string(status));
        goto cleanup;
    }
    printf("resized_width=%u elements=%llu\n", resized_width,
           (unsigned long long)output_count);
    result = 0;
cleanup:
    free(output);
    free(source);
    return result;
}

static int decode_to_file(
    const char* dictionary_path,
    const float* probabilities,
    uint64_t probability_count,
    uint32_t time_steps,
    uint32_t class_count,
    const char* output_path) {
    lw_rec_dictionary* dictionary = NULL;
    lw_error error;
    lw_status status;
    uint64_t required = 0u;
    float score = 0.0f;
    uint32_t emitted = 0u;
    char* text = NULL;
    int result = 1;
    lw_error_init(&error);
    status = lw_rec_dictionary_load(dictionary_path, &dictionary, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "dictionary load failed: %s: %s\n",
                lw_status_string(status), error.message);
        goto cleanup;
    }
    if (lw_rec_dictionary_class_count(dictionary) != class_count) {
        fprintf(stderr, "dictionary class count mismatch\n");
        goto cleanup;
    }
    lw_error_init(&error);
    status = lw_rec_ctc_decode_f32(
        dictionary, probabilities, probability_count, time_steps, class_count,
        NULL, 0u, &required, &score, &emitted, &error);
    if (status != LW_STATUS_OK || required == 0u || required > SIZE_MAX) {
        fprintf(stderr, "CTC size query failed: %s: %s\n",
                lw_status_string(status), error.message);
        goto cleanup;
    }
    text = (char*)malloc((size_t)required);
    if (text == NULL) {
        fprintf(stderr, "CTC text allocation failed\n");
        goto cleanup;
    }
    lw_error_init(&error);
    status = lw_rec_ctc_decode_f32(
        dictionary, probabilities, probability_count, time_steps, class_count,
        text, required - 1u, &required, &score, &emitted, &error);
    if (status != LW_STATUS_OUT_OF_BOUNDS) {
        fprintf(stderr, "CTC decoder accepted an undersized text buffer\n");
        goto cleanup;
    }
    lw_error_init(&error);
    status = lw_rec_ctc_decode_f32(
        dictionary, probabilities, probability_count, time_steps, class_count,
        text, required, &required, &score, &emitted, &error);
    if (status != LW_STATUS_OK || !write_file(output_path, text, (size_t)required - 1u)) {
        fprintf(stderr, "CTC decode failed: %s: %s\n",
                lw_status_string(status), error.message);
        goto cleanup;
    }
    printf("score=%.9g chars=%u bytes=%llu\n", score, emitted,
           (unsigned long long)(required - 1u));
    result = 0;
cleanup:
    free(text);
    lw_rec_dictionary_free(dictionary);
    return result;
}

static int run_decode(int argc, char** argv) {
    uint8_t* bytes = NULL;
    size_t byte_count = 0u;
    uint32_t time_steps;
    uint32_t class_count;
    int result;
    if (argc != 7 || !parse_u32(argv[4], &time_steps) ||
        !parse_u32(argv[5], &class_count) || !read_file(argv[3], &bytes, &byte_count) ||
        byte_count % sizeof(float) != 0u) {
        fprintf(stderr, "invalid decode arguments or probability file\n");
        free(bytes);
        return 2;
    }
    result = decode_to_file(
        argv[2], (const float*)(const void*)bytes, byte_count / sizeof(float),
        time_steps, class_count, argv[6]);
    free(bytes);
    return result;
}

static int run_pipeline(int argc, char** argv) {
    lw_recognizer* recognizer = NULL;
    uint8_t* source = NULL;
    size_t source_bytes = 0u;
    char* text = NULL;
    lw_recognizer_options options;
    lw_recognizer_info info;
    lw_recognition_result recognition;
    lw_error error;
    lw_status status;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t target_width;
    int result = 1;
    if (argc != 10 || !parse_u32(argv[5], &width) || !parse_u32(argv[6], &height) ||
        !parse_u32(argv[7], &stride) || !parse_u32(argv[8], &target_width) ||
        target_width > INT32_MAX || !read_file(argv[4], &source, &source_bytes)) {
        fprintf(stderr, "invalid pipeline arguments or source file\n");
        return 2;
    }
    lw_recognizer_options_init(&options);
    options.target_width = target_width;
    lw_error_init(&error);
    status = lw_recognizer_create(
        argv[2], argv[3], &options, &recognizer, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "pipeline recognizer create failed: %s: %s\n",
                lw_status_string(status), error.message);
        goto cleanup;
    }
    lw_recognizer_info_init(&info);
    if (lw_recognizer_get_info(recognizer, &info) != LW_STATUS_OK ||
        info.max_text_capacity == 0u || info.max_text_capacity > SIZE_MAX) {
        fprintf(stderr, "pipeline recognizer info failed\n");
        goto cleanup;
    }
    text = (char*)malloc((size_t)info.max_text_capacity);
    if (text == NULL) {
        fprintf(stderr, "pipeline text allocation failed\n");
        goto cleanup;
    }
    lw_recognition_result_init(&recognition);
    lw_error_init(&error);
    status = lw_recognizer_recognize_bgr_u8(
        recognizer, source, source_bytes, width, height, stride,
        text, info.max_text_capacity, &recognition, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "pipeline recognition failed: %s: %s\n",
                lw_status_string(status), error.message);
        goto cleanup;
    }
    if (!write_file(argv[9], text, (size_t)recognition.required_text_capacity - 1u)) {
        fprintf(stderr, "pipeline text write failed\n");
        goto cleanup;
    }
    printf("score=%.9g chars=%u bytes=%llu\n", recognition.score,
           recognition.emitted_count,
           (unsigned long long)(recognition.required_text_capacity - 1u));
    fprintf(stderr, "resized_width=%u output_steps=%u\n",
            recognition.resized_width, recognition.time_steps);
    result = 0;
cleanup:
    free(text);
    free(source);
    lw_recognizer_free(recognizer);
    return result;
}

static int driver_main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "expected preprocess, decode, or pipeline mode\n");
        return 2;
    }
    if (strcmp(argv[1], "preprocess") == 0) {
        return run_preprocess(argc, argv);
    }
    if (strcmp(argv[1], "decode") == 0) {
        return run_decode(argc, argv);
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
        int byte_count = WideCharToMultiByte(
            CP_UTF8, WC_ERR_INVALID_CHARS, wide_argv[index], -1,
            NULL, 0, NULL, NULL);
        if (byte_count <= 0) {
            goto cleanup;
        }
        utf8_argv[index] = (char*)malloc((size_t)byte_count);
        if (utf8_argv[index] == NULL ||
            WideCharToMultiByte(
                CP_UTF8, WC_ERR_INVALID_CHARS, wide_argv[index], -1,
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
