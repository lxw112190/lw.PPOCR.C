#include "lw_infer.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int expect_status(lw_status actual, lw_status expected) {
    return actual == expected;
}

int main(int argc, char** argv) {
    lw_recognizer* recognizer = NULL;
    lw_recognizer* limited = NULL;
    lw_recognizer_options options;
    lw_recognizer_info info;
    lw_recognition_result result;
    lw_error error;
    uint8_t source[5u * 24u];
    char* text = NULL;
    uint64_t required;
    lw_status status;
    uint32_t index;
    int return_code = 1;
    if (argc != 3) {
        return 2;
    }
    for (index = 0u; index < sizeof(source); ++index) {
        source[index] = (uint8_t)((index * 37u + 11u) & 0xffu);
    }

    lw_recognizer_options_init(&options);
    options.reserved0 = 1u;
    lw_error_init(&error);
    status = lw_recognizer_create(argv[1], argv[2], &options, &recognizer, &error);
    if (!expect_status(status, LW_STATUS_INVALID_ARGUMENT) || recognizer != NULL) {
        goto cleanup;
    }

    lw_recognizer_options_init(&options);
    options.target_width = 80u;
    lw_error_init(&error);
    status = lw_recognizer_create(argv[1], argv[2], &options, &recognizer, &error);
    if (!expect_status(status, LW_STATUS_OK) || recognizer == NULL) {
        goto cleanup;
    }
    lw_recognizer_info_init(&info);
    if (!expect_status(lw_recognizer_get_info(recognizer, &info), LW_STATUS_OK) ||
        info.target_width != 80u || info.input_height != 48u ||
        info.time_steps == 0u || info.class_count != 6906u ||
        info.max_text_capacity == 0u || info.workspace_size == 0u) {
        goto cleanup;
    }

    lw_recognition_result_init(&result);
    lw_error_init(&error);
    status = lw_recognizer_recognize_bgr_u8(
        recognizer, source, 5u * 24u - 4u, 7u, 5u, 24u,
        NULL, 0u, &result, &error);
    if (!expect_status(status, LW_STATUS_INVALID_SHAPE)) {
        goto cleanup;
    }

    lw_recognition_result_init(&result);
    lw_error_init(&error);
    status = lw_recognizer_recognize_bgr_u8(
        recognizer, source, sizeof(source), 7u, 5u, 24u,
        NULL, 0u, &result, &error);
    if (!expect_status(status, LW_STATUS_OK) ||
        result.required_text_capacity == 0u || result.time_steps != info.time_steps) {
        goto cleanup;
    }
    required = result.required_text_capacity;
    if (required > SIZE_MAX || (text = (char*)malloc((size_t)required)) == NULL) {
        goto cleanup;
    }
    lw_recognition_result_init(&result);
    lw_error_init(&error);
    status = lw_recognizer_recognize_bgr_u8(
        recognizer, source, sizeof(source), 7u, 5u, 24u,
        text, required - 1u, &result, &error);
    if (!expect_status(status, LW_STATUS_OUT_OF_BOUNDS) ||
        result.required_text_capacity != required) {
        goto cleanup;
    }
    lw_recognition_result_init(&result);
    lw_error_init(&error);
    status = lw_recognizer_recognize_bgr_u8(
        recognizer, source, sizeof(source), 7u, 5u, 24u,
        text, required, &result, &error);
    if (!expect_status(status, LW_STATUS_OK) ||
        result.required_text_capacity != required || text[required - 1u] != '\0') {
        goto cleanup;
    }
    memset(&result, 0, sizeof(result));
    lw_error_init(&error);
    status = lw_recognizer_recognize_bgr_u8(
        recognizer, source, sizeof(source), 7u, 5u, 24u,
        text, required, &result, &error);
    if (!expect_status(status, LW_STATUS_INVALID_ARGUMENT)) {
        goto cleanup;
    }

    lw_recognizer_options_init(&options);
    options.target_width = 80u;
    options.max_image_pixels = 34u;
    lw_error_init(&error);
    status = lw_recognizer_create(argv[1], argv[2], &options, &limited, &error);
    if (!expect_status(status, LW_STATUS_OK) || limited == NULL) {
        goto cleanup;
    }
    lw_recognition_result_init(&result);
    lw_error_init(&error);
    status = lw_recognizer_recognize_bgr_u8(
        limited, source, sizeof(source), 7u, 5u, 24u,
        NULL, 0u, &result, &error);
    if (!expect_status(status, LW_STATUS_MEMORY_LIMIT)) {
        goto cleanup;
    }
    return_code = 0;

cleanup:
    free(text);
    lw_recognizer_free(limited);
    lw_recognizer_free(recognizer);
    return return_code;
}
