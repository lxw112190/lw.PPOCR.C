#include "executor_internal.h"
#include "lw_infer.h"
#include "rec_internal.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    lw_model* model = NULL;
    lw_session* session = NULL;
    uint8_t* source = NULL;
    size_t source_bytes = 0u;
    float* input = NULL;
    float* probabilities = NULL;
    lw_tensor_desc input_desc;
    lw_tensor_desc output_desc;
    lw_error error;
    lw_status status;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t target_width;
    uint32_t resized_width;
    uint64_t input_count;
    uint64_t probability_count = 1u;
    uint32_t index;
    int result = 1;
    if (argc != 10 || !parse_u32(argv[5], &width) || !parse_u32(argv[6], &height) ||
        !parse_u32(argv[7], &stride) || !parse_u32(argv[8], &target_width) ||
        target_width > INT32_MAX || !read_file(argv[4], &source, &source_bytes)) {
        fprintf(stderr, "invalid pipeline arguments or source file\n");
        return 2;
    }
    input_count = (uint64_t)3u * LW_REC_INPUT_HEIGHT * target_width;
    if (input_count > SIZE_MAX / sizeof(*input) ||
        (input = (float*)malloc((size_t)input_count * sizeof(*input))) == NULL) {
        fprintf(stderr, "pipeline input allocation failed\n");
        goto cleanup;
    }
    status = lw_rec_preprocess_bgr_u8(
        source, source_bytes, width, height, stride, target_width,
        input, input_count, &resized_width);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "pipeline preprocess failed: %s\n", lw_status_string(status));
        goto cleanup;
    }
    lw_error_init(&error);
    status = lw_model_load(argv[2], NULL, &model, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "pipeline model load failed: %s: %s\n",
                lw_status_string(status), error.message);
        goto cleanup;
    }
    lw_tensor_desc_init(&input_desc);
    input_desc.dtype = LW_DTYPE_F32;
    input_desc.rank = 4u;
    input_desc.dimensions[0] = 1;
    input_desc.dimensions[1] = 3;
    input_desc.dimensions[2] = (int32_t)LW_REC_INPUT_HEIGHT;
    input_desc.dimensions[3] = (int32_t)target_width;
    lw_error_init(&error);
    status = lw_session_create(model, &input_desc, 1u, NULL, &session, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "pipeline session create failed: %s: %s\n",
                lw_status_string(status), error.message);
        goto cleanup;
    }
    lw_tensor_desc_init(&output_desc);
    if (lw_session_get_output_desc(session, 0u, &output_desc) != LW_STATUS_OK) {
        fprintf(stderr, "pipeline output descriptor failed\n");
        goto cleanup;
    }
    for (index = 0u; index < output_desc.rank; ++index) {
        probability_count *= (uint32_t)output_desc.dimensions[index];
    }
    if (probability_count > SIZE_MAX / sizeof(*probabilities) ||
        (probabilities = (float*)malloc((size_t)probability_count * sizeof(*probabilities))) == NULL) {
        fprintf(stderr, "pipeline probability allocation failed\n");
        goto cleanup;
    }
    lw_error_init(&error);
    status = lw_execute_session_f32(
        session, input, input_count, probabilities, probability_count, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "pipeline graph execution failed: %s: %s\n",
                lw_status_string(status), error.message);
        goto cleanup;
    }
    result = decode_to_file(
        argv[3], probabilities, probability_count,
        (uint32_t)output_desc.dimensions[1], (uint32_t)output_desc.dimensions[2], argv[9]);
    if (result == 0) {
        fprintf(stderr, "resized_width=%u output_steps=%d\n",
                resized_width, output_desc.dimensions[1]);
    }
cleanup:
    free(probabilities);
    free(input);
    free(source);
    lw_session_free(session);
    lw_model_free(model);
    return result;
}

int main(int argc, char** argv) {
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
