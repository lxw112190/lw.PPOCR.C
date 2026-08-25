#include "lw_infer.h"

#include "cls_internal.h"
#include "error_internal.h"
#include "executor_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define LW_CLS_DEFAULT_MAX_IMAGE_PIXELS UINT64_C(40000000)

struct lw_classifier {
    lw_model* model;
    lw_session* session;
    float* input;
    float probabilities[LW_CLS_CLASS_COUNT];
    uint64_t input_element_count;
    uint64_t max_image_pixels;
    lw_classifier_info info;
};

static void clear_result(lw_classification_result* result) {
    memset(result, 0, sizeof(*result));
    result->struct_size = (uint32_t)sizeof(*result);
}

void lw_classifier_options_init(lw_classifier_options* options) {
    lw_model_options model_options;
    lw_session_options session_options;
    if (options == NULL) {
        return;
    }
    lw_model_options_init(&model_options);
    lw_session_options_init(&session_options);
    memset(options, 0, sizeof(*options));
    options->struct_size = (uint32_t)sizeof(*options);
    options->max_model_file_size = model_options.max_file_size;
    options->max_workspace_size = session_options.max_workspace_size;
    options->max_tensor_size = session_options.max_tensor_size;
    options->max_image_pixels = LW_CLS_DEFAULT_MAX_IMAGE_PIXELS;
}

void lw_classifier_info_init(lw_classifier_info* info) {
    if (info == NULL) {
        return;
    }
    memset(info, 0, sizeof(*info));
    info->struct_size = (uint32_t)sizeof(*info);
}

void lw_classification_result_init(lw_classification_result* result) {
    if (result == NULL) {
        return;
    }
    clear_result(result);
}

static lw_status validate_options(
    const lw_classifier_options* options,
    uint64_t* max_image_pixels,
    lw_model_options* model_options,
    lw_session_options* session_options,
    lw_error* error) {
    lw_classifier_options defaults;
    lw_classifier_options_init(&defaults);
    *max_image_pixels = defaults.max_image_pixels;
    lw_model_options_init(model_options);
    lw_session_options_init(session_options);
    if (options == NULL) {
        return LW_STATUS_OK;
    }
    if (options->struct_size != sizeof(*options) || options->reserved != 0u) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "invalid classifier options structure");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    if (options->max_model_file_size != 0u) {
        model_options->max_file_size = options->max_model_file_size;
    }
    if (options->max_workspace_size != 0u) {
        session_options->max_workspace_size = options->max_workspace_size;
    }
    if (options->max_tensor_size != 0u) {
        session_options->max_tensor_size = options->max_tensor_size;
    }
    if (options->max_image_pixels != 0u) {
        *max_image_pixels = options->max_image_pixels;
    }
    return LW_STATUS_OK;
}

lw_status lw_classifier_create(
    const char* model_path_utf8,
    const lw_classifier_options* options,
    lw_classifier** out_classifier,
    lw_error* error) {
    lw_model_options model_options;
    lw_session_options session_options;
    lw_tensor_desc input_desc;
    lw_tensor_desc output_desc;
    lw_session_info session_info;
    lw_classifier* classifier = NULL;
    uint64_t max_image_pixels;
    lw_status status;
    if (out_classifier != NULL) {
        *out_classifier = NULL;
    }
    if (model_path_utf8 == NULL || model_path_utf8[0] == '\0' ||
        out_classifier == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "model path and output classifier are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    status = validate_options(options, &max_image_pixels, &model_options,
                              &session_options, error);
    if (status != LW_STATUS_OK) {
        return status;
    }
    classifier = (lw_classifier*)calloc(1u, sizeof(*classifier));
    if (classifier == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY,
                     "unable to allocate classifier handle");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    classifier->max_image_pixels = max_image_pixels;
    status = lw_model_load(model_path_utf8, &model_options, &classifier->model, error);
    if (status != LW_STATUS_OK) {
        goto fail;
    }
    lw_tensor_desc_init(&input_desc);
    input_desc.dtype = LW_DTYPE_F32;
    input_desc.rank = 4u;
    input_desc.dimensions[0] = 1;
    input_desc.dimensions[1] = 3;
    input_desc.dimensions[2] = (int32_t)LW_CLS_INPUT_HEIGHT;
    input_desc.dimensions[3] = (int32_t)LW_CLS_INPUT_WIDTH;
    status = lw_session_create(
        classifier->model, &input_desc, 1u, &session_options,
        &classifier->session, error);
    if (status != LW_STATUS_OK) {
        goto fail;
    }
    lw_tensor_desc_init(&output_desc);
    status = lw_session_get_output_desc(classifier->session, 0u, &output_desc);
    if (status != LW_STATUS_OK || output_desc.dtype != LW_DTYPE_F32 ||
        output_desc.rank != 2u || output_desc.dimensions[0] != 1 ||
        output_desc.dimensions[1] != (int32_t)LW_CLS_CLASS_COUNT) {
        lw_set_error(error, LW_STATUS_INVALID_SHAPE,
                     "classifier model output must have shape [1,2]");
        status = LW_STATUS_INVALID_SHAPE;
        goto fail;
    }
    classifier->input_element_count =
        (uint64_t)3u * LW_CLS_INPUT_HEIGHT * LW_CLS_INPUT_WIDTH;
    if (classifier->input_element_count > SIZE_MAX / sizeof(*classifier->input)) {
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS,
                     "classifier input buffer size overflows");
        status = LW_STATUS_OUT_OF_BOUNDS;
        goto fail;
    }
    classifier->input = (float*)malloc(
        (size_t)classifier->input_element_count * sizeof(*classifier->input));
    if (classifier->input == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY,
                     "unable to allocate classifier input buffer");
        status = LW_STATUS_OUT_OF_MEMORY;
        goto fail;
    }
    lw_session_info_init(&session_info);
    status = lw_session_get_info(classifier->session, &session_info);
    if (status != LW_STATUS_OK) {
        lw_set_error(error, status, "unable to read classifier session information");
        goto fail;
    }
    lw_classifier_info_init(&classifier->info);
    classifier->info.input_width = LW_CLS_INPUT_WIDTH;
    classifier->info.input_height = LW_CLS_INPUT_HEIGHT;
    classifier->info.class_count = LW_CLS_CLASS_COUNT;
    classifier->info.workspace_size = session_info.workspace_size;
    *out_classifier = classifier;
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;

fail:
    lw_classifier_free(classifier);
    return status;
}

void lw_classifier_free(lw_classifier* classifier) {
    if (classifier == NULL) {
        return;
    }
    free(classifier->input);
    lw_session_free(classifier->session);
    lw_model_free(classifier->model);
    free(classifier);
}

lw_status lw_classifier_get_info(
    const lw_classifier* classifier,
    lw_classifier_info* info) {
    if (classifier == NULL || info == NULL || info->struct_size != sizeof(*info)) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    *info = classifier->info;
    return LW_STATUS_OK;
}

lw_status lw_classifier_classify_bgr_u8(
    lw_classifier* classifier,
    const uint8_t* source,
    uint64_t source_byte_count,
    uint32_t source_width,
    uint32_t source_height,
    uint32_t source_stride,
    lw_classification_result* result,
    lw_error* error) {
    uint64_t source_pixels;
    uint32_t resized_width = 0u;
    uint32_t label;
    lw_status status;
    if (classifier == NULL || source == NULL || result == NULL ||
        result->struct_size != sizeof(*result)) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "classifier, BGR source, and initialized result are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    clear_result(result);
    if (source_width == 0u || source_height == 0u) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "source image dimensions must be positive");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    source_pixels = (uint64_t)source_width * source_height;
    if (source_pixels > classifier->max_image_pixels) {
        lw_set_error(error, LW_STATUS_MEMORY_LIMIT,
                     "source image exceeds max_image_pixels");
        return LW_STATUS_MEMORY_LIMIT;
    }
    status = lw_cls_preprocess_bgr_u8(
        source, source_byte_count, source_width, source_height, source_stride,
        classifier->input, classifier->input_element_count, &resized_width);
    if (status != LW_STATUS_OK) {
        lw_set_error(error, status, "BGR source layout is invalid");
        return status;
    }
    status = lw_execute_session_f32(
        classifier->session, classifier->input, classifier->input_element_count,
        classifier->probabilities, LW_CLS_CLASS_COUNT, error);
    if (status != LW_STATUS_OK) {
        return status;
    }
    if (!isfinite(classifier->probabilities[0]) ||
        !isfinite(classifier->probabilities[1])) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "classifier output contains a non-finite value");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    label = classifier->probabilities[1] > classifier->probabilities[0] ? 1u : 0u;
    result->label = label;
    result->score = classifier->probabilities[label];
    result->resized_width = resized_width;
    result->orientation_degrees = label == 0u ? 0u : 180u;
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}
