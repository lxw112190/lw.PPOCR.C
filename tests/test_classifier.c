#include "lw_infer.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static int expect_status(lw_status actual, lw_status expected) {
    return actual == expected;
}

int main(int argc, char** argv) {
    lw_classifier* classifier = NULL;
    lw_classifier* limited = NULL;
    lw_classifier_options options;
    lw_classifier_info info;
    lw_classification_result result;
    lw_classification_result repeated;
    lw_error error;
    uint8_t source[5u * 24u];
    lw_status status;
    uint32_t index;
    int return_code = 1;
    if (argc != 2) {
        return 2;
    }
    for (index = 0u; index < sizeof(source); ++index) {
        source[index] = (uint8_t)((index * 37u + 11u) & 0xffu);
    }

    lw_classifier_options_init(&options);
    options.reserved = 1u;
    lw_error_init(&error);
    status = lw_classifier_create(argv[1], &options, &classifier, &error);
    if (!expect_status(status, LW_STATUS_INVALID_ARGUMENT) || classifier != NULL) {
        goto cleanup;
    }

    lw_classifier_options_init(&options);
    lw_error_init(&error);
    status = lw_classifier_create(argv[1], &options, &classifier, &error);
    if (!expect_status(status, LW_STATUS_OK) || classifier == NULL) {
        goto cleanup;
    }
    lw_classifier_info_init(&info);
    if (!expect_status(lw_classifier_get_info(classifier, &info), LW_STATUS_OK) ||
        info.input_width != 160u || info.input_height != 80u || info.class_count != 2u ||
        info.workspace_size == 0u) {
        goto cleanup;
    }

    lw_classification_result_init(&result);
    lw_error_init(&error);
    status = lw_classifier_classify_bgr_u8(classifier, source, sizeof(source) - 4u, 7u, 5u, 24u,
                                           &result, &error);
    if (!expect_status(status, LW_STATUS_INVALID_SHAPE)) {
        goto cleanup;
    }

    lw_classification_result_init(&result);
    lw_error_init(&error);
    status = lw_classifier_classify_bgr_u8(classifier, source, sizeof(source), 7u, 5u, 24u, &result,
                                           &error);
    if (!expect_status(status, LW_STATUS_OK) || result.label > 1u || !isfinite(result.score) ||
        result.score < 0.5f || result.score > 1.0f || result.resized_width != 112u ||
        result.orientation_degrees != result.label * 180u || result.reserved != 0u) {
        goto cleanup;
    }
    lw_classification_result_init(&repeated);
    lw_error_init(&error);
    status = lw_classifier_classify_bgr_u8(classifier, source, sizeof(source), 7u, 5u, 24u,
                                           &repeated, &error);
    if (!expect_status(status, LW_STATUS_OK) || memcmp(&result, &repeated, sizeof(result)) != 0) {
        goto cleanup;
    }
    memset(&result, 0, sizeof(result));
    lw_error_init(&error);
    status = lw_classifier_classify_bgr_u8(classifier, source, sizeof(source), 7u, 5u, 24u, &result,
                                           &error);
    if (!expect_status(status, LW_STATUS_INVALID_ARGUMENT)) {
        goto cleanup;
    }

    lw_classifier_options_init(&options);
    options.max_image_pixels = 34u;
    lw_error_init(&error);
    status = lw_classifier_create(argv[1], &options, &limited, &error);
    if (!expect_status(status, LW_STATUS_OK) || limited == NULL) {
        goto cleanup;
    }
    lw_classification_result_init(&result);
    lw_error_init(&error);
    status = lw_classifier_classify_bgr_u8(limited, source, sizeof(source), 7u, 5u, 24u, &result,
                                           &error);
    if (!expect_status(status, LW_STATUS_MEMORY_LIMIT)) {
        goto cleanup;
    }
    return_code = 0;

cleanup:
    lw_classifier_free(limited);
    lw_classifier_free(classifier);
    return return_code;
}
