#include "lw_infer.h"

/* Public REC handle: preprocessing -> graph execution -> UTF-8 CTC decoding. */

#include "error_internal.h"
#include "executor_internal.h"
#include "rec_internal.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define LW_REC_DEFAULT_TARGET_WIDTH 320u
#define LW_REC_DEFAULT_MAX_IMAGE_PIXELS UINT64_C(40000000)

struct lw_recognizer {
    lw_model* model;
    lw_session* session;
    lw_rec_dictionary* dictionary;
    float* input;
    float* probabilities;
    uint64_t input_element_count;
    uint64_t probability_element_count;
    uint64_t max_image_pixels;
    lw_recognizer_info info;
};

static void clear_result(lw_recognition_result* result) {
    memset(result, 0, sizeof(*result));
    result->struct_size = (uint32_t)sizeof(*result);
}

void lw_recognizer_options_init(lw_recognizer_options* options) {
    lw_model_options model_options;
    lw_session_options session_options;
    if (options == NULL) {
        return;
    }
    lw_model_options_init(&model_options);
    lw_session_options_init(&session_options);
    memset(options, 0, sizeof(*options));
    options->struct_size = (uint32_t)sizeof(*options);
    options->target_width = LW_REC_DEFAULT_TARGET_WIDTH;
    options->max_model_file_size = model_options.max_file_size;
    options->max_workspace_size = session_options.max_workspace_size;
    options->max_tensor_size = session_options.max_tensor_size;
    options->max_image_pixels = LW_REC_DEFAULT_MAX_IMAGE_PIXELS;
}

void lw_recognizer_info_init(lw_recognizer_info* info) {
    if (info == NULL) {
        return;
    }
    memset(info, 0, sizeof(*info));
    info->struct_size = (uint32_t)sizeof(*info);
}

void lw_recognition_result_init(lw_recognition_result* result) {
    if (result == NULL) {
        return;
    }
    clear_result(result);
}

static lw_status validate_options(const lw_recognizer_options* options, uint32_t* target_width,
                                  uint64_t* max_image_pixels, lw_model_options* model_options,
                                  lw_session_options* session_options, lw_error* error) {
    lw_recognizer_options defaults;
    lw_recognizer_options_init(&defaults);
    *target_width = defaults.target_width;
    *max_image_pixels = defaults.max_image_pixels;
    lw_model_options_init(model_options);
    lw_session_options_init(session_options);
    if (options == NULL) {
        return LW_STATUS_OK;
    }
    if (options->struct_size != sizeof(*options) || options->reserved0 != 0u ||
        options->reserved1 != 0u) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "invalid recognizer options structure");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    if (options->target_width != 0u) {
        *target_width = options->target_width;
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
    if (*target_width > INT32_MAX) {
        lw_set_error(error, LW_STATUS_INVALID_SHAPE,
                     "recognizer target width exceeds the tensor ABI");
        return LW_STATUS_INVALID_SHAPE;
    }
    return LW_STATUS_OK;
}

lw_status lw_recognizer_create(const char* model_path_utf8, const char* dictionary_path_utf8,
                               const lw_recognizer_options* options, lw_recognizer** out_recognizer,
                               lw_error* error) {
    lw_model_options model_options;
    lw_session_options session_options;
    lw_tensor_desc input_desc;
    lw_tensor_desc output_desc;
    lw_session_info session_info;
    lw_recognizer* recognizer = NULL;
    uint32_t target_width;
    uint64_t max_image_pixels;
    uint32_t max_label_bytes;
    lw_status status;
    if (out_recognizer != NULL) {
        *out_recognizer = NULL;
    }
    if (model_path_utf8 == NULL || model_path_utf8[0] == '\0' || dictionary_path_utf8 == NULL ||
        dictionary_path_utf8[0] == '\0' || out_recognizer == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "model path, dictionary path, and output recognizer are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    status = validate_options(options, &target_width, &max_image_pixels, &model_options,
                              &session_options, error);
    if (status != LW_STATUS_OK) {
        return status;
    }
    recognizer = (lw_recognizer*)calloc(1u, sizeof(*recognizer));
    if (recognizer == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate recognizer handle");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    recognizer->max_image_pixels = max_image_pixels;
    status = lw_model_load(model_path_utf8, &model_options, &recognizer->model, error);
    if (status != LW_STATUS_OK) {
        goto fail;
    }
    status = lw_rec_dictionary_load(dictionary_path_utf8, &recognizer->dictionary, error);
    if (status != LW_STATUS_OK) {
        goto fail;
    }
    lw_tensor_desc_init(&input_desc);
    input_desc.dtype = LW_DTYPE_F32;
    input_desc.rank = 4u;
    input_desc.dimensions[0] = 1;
    input_desc.dimensions[1] = 3;
    input_desc.dimensions[2] = (int32_t)LW_REC_INPUT_HEIGHT;
    input_desc.dimensions[3] = (int32_t)target_width;
    status = lw_session_create(recognizer->model, &input_desc, 1u, &session_options,
                               &recognizer->session, error);
    if (status != LW_STATUS_OK) {
        goto fail;
    }
    lw_tensor_desc_init(&output_desc);
    status = lw_session_get_output_desc(recognizer->session, 0u, &output_desc);
    if (status != LW_STATUS_OK || output_desc.dtype != LW_DTYPE_F32 || output_desc.rank != 3u ||
        output_desc.dimensions[0] != 1 || output_desc.dimensions[1] <= 0 ||
        output_desc.dimensions[2] <= 0 ||
        (uint32_t)output_desc.dimensions[2] !=
            lw_rec_dictionary_class_count(recognizer->dictionary)) {
        lw_set_error(error, LW_STATUS_INVALID_SHAPE,
                     "recognizer model output and dictionary are incompatible");
        status = LW_STATUS_INVALID_SHAPE;
        goto fail;
    }
    recognizer->input_element_count = (uint64_t)3u * LW_REC_INPUT_HEIGHT * target_width;
    recognizer->probability_element_count =
        (uint64_t)(uint32_t)output_desc.dimensions[1] * (uint32_t)output_desc.dimensions[2];
    if (recognizer->input_element_count > SIZE_MAX / sizeof(*recognizer->input) ||
        recognizer->probability_element_count > SIZE_MAX / sizeof(*recognizer->probabilities)) {
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "recognizer buffer size overflows");
        status = LW_STATUS_OUT_OF_BOUNDS;
        goto fail;
    }
    recognizer->input =
        (float*)malloc((size_t)recognizer->input_element_count * sizeof(*recognizer->input));
    recognizer->probabilities = (float*)malloc((size_t)recognizer->probability_element_count *
                                               sizeof(*recognizer->probabilities));
    if (recognizer->input == NULL || recognizer->probabilities == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate recognizer buffers");
        status = LW_STATUS_OUT_OF_MEMORY;
        goto fail;
    }
    max_label_bytes = lw_rec_dictionary_max_label_byte_count(recognizer->dictionary);
    if (max_label_bytes == 0u ||
        (uint64_t)(uint32_t)output_desc.dimensions[1] > (UINT64_MAX - 1u) / max_label_bytes) {
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "recognizer maximum text capacity overflows");
        status = LW_STATUS_OUT_OF_BOUNDS;
        goto fail;
    }
    lw_session_info_init(&session_info);
    status = lw_session_get_info(recognizer->session, &session_info);
    if (status != LW_STATUS_OK) {
        lw_set_error(error, status, "unable to read recognizer session information");
        goto fail;
    }
    lw_recognizer_info_init(&recognizer->info);
    recognizer->info.target_width = target_width;
    recognizer->info.input_height = LW_REC_INPUT_HEIGHT;
    recognizer->info.time_steps = (uint32_t)output_desc.dimensions[1];
    recognizer->info.class_count = (uint32_t)output_desc.dimensions[2];
    recognizer->info.max_text_capacity =
        (uint64_t)recognizer->info.time_steps * max_label_bytes + 1u;
    recognizer->info.workspace_size = session_info.workspace_size;
    *out_recognizer = recognizer;
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;

fail:
    lw_recognizer_free(recognizer);
    return status;
}

void lw_recognizer_free(lw_recognizer* recognizer) {
    if (recognizer == NULL) {
        return;
    }
    free(recognizer->probabilities);
    free(recognizer->input);
    lw_session_free(recognizer->session);
    lw_rec_dictionary_free(recognizer->dictionary);
    lw_model_free(recognizer->model);
    free(recognizer);
}

lw_status lw_recognizer_get_info(const lw_recognizer* recognizer, lw_recognizer_info* info) {
    if (recognizer == NULL || info == NULL || info->struct_size != sizeof(*info)) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    *info = recognizer->info;
    return LW_STATUS_OK;
}

lw_status lw_recognizer_recognize_bgr_u8(lw_recognizer* recognizer, const uint8_t* source,
                                         uint64_t source_byte_count, uint32_t source_width,
                                         uint32_t source_height, uint32_t source_stride,
                                         char* text_utf8, uint64_t text_capacity,
                                         lw_recognition_result* result, lw_error* error) {
    uint64_t source_pixels;
    uint32_t resized_width = 0u;
    uint64_t required_capacity = 0u;
    float score = 0.0f;
    uint32_t emitted_count = 0u;
    lw_status status;
    if (recognizer == NULL || source == NULL || result == NULL ||
        result->struct_size != sizeof(*result) || (text_utf8 == NULL && text_capacity != 0u)) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "recognizer, BGR source, and initialized result are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    clear_result(result);
    if (source_width == 0u || source_height == 0u) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "source image dimensions must be positive");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    source_pixels = (uint64_t)source_width * source_height;
    if (source_pixels > recognizer->max_image_pixels) {
        lw_set_error(error, LW_STATUS_MEMORY_LIMIT, "source image exceeds max_image_pixels");
        return LW_STATUS_MEMORY_LIMIT;
    }
    status =
        lw_rec_preprocess_bgr_u8(source, source_byte_count, source_width, source_height,
                                 source_stride, recognizer->info.target_width, recognizer->input,
                                 recognizer->input_element_count, &resized_width);
    if (status != LW_STATUS_OK) {
        lw_set_error(error, status, "BGR source layout is invalid");
        return status;
    }
    status = lw_execute_session_f32(recognizer->session, recognizer->input,
                                    recognizer->input_element_count, recognizer->probabilities,
                                    recognizer->probability_element_count, error);
    if (status != LW_STATUS_OK) {
        return status;
    }
    status = lw_rec_ctc_decode_f32(
        recognizer->dictionary, recognizer->probabilities, recognizer->probability_element_count,
        recognizer->info.time_steps, recognizer->info.class_count, text_utf8, text_capacity,
        &required_capacity, &score, &emitted_count, error);
    result->emitted_count = emitted_count;
    result->score = score;
    result->resized_width = resized_width;
    result->time_steps = recognizer->info.time_steps;
    result->required_text_capacity = required_capacity;
    return status;
}
