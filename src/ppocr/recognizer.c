#include "lw_infer.h"

/* Public REC handle: preprocessing -> graph execution -> UTF-8 CTC decoding. */

#include "error_internal.h"
#include "executor_internal.h"
#include "profile_internal.h"
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
    lw_session* cached_session;
    float* cached_input;
    float* cached_probabilities;
    uint64_t cached_input_element_count;
    uint64_t cached_probability_element_count;
    uint32_t cached_target_width;
    uint32_t cached_time_steps;
    uint64_t max_image_pixels;
    lw_session_options session_options;
    uint32_t current_target_width;
    uint32_t current_time_steps;
    uint32_t adaptive_width_enabled;
    lw_recognizer_info info;
};

static void release_cached_session(lw_recognizer* recognizer) {
    free(recognizer->cached_probabilities);
    free(recognizer->cached_input);
    lw_session_free(recognizer->cached_session);
    recognizer->cached_session = NULL;
    recognizer->cached_input = NULL;
    recognizer->cached_probabilities = NULL;
    recognizer->cached_input_element_count = 0u;
    recognizer->cached_probability_element_count = 0u;
    recognizer->cached_target_width = 0u;
    recognizer->cached_time_steps = 0u;
}

static void activate_cached_session(lw_recognizer* recognizer) {
    lw_session* session = recognizer->session;
    float* input = recognizer->input;
    float* probabilities = recognizer->probabilities;
    uint64_t input_element_count = recognizer->input_element_count;
    uint64_t probability_element_count = recognizer->probability_element_count;
    uint32_t target_width = recognizer->current_target_width;
    uint32_t time_steps = recognizer->current_time_steps;

    recognizer->session = recognizer->cached_session;
    recognizer->input = recognizer->cached_input;
    recognizer->probabilities = recognizer->cached_probabilities;
    recognizer->input_element_count = recognizer->cached_input_element_count;
    recognizer->probability_element_count = recognizer->cached_probability_element_count;
    recognizer->current_target_width = recognizer->cached_target_width;
    recognizer->current_time_steps = recognizer->cached_time_steps;

    recognizer->cached_session = session;
    recognizer->cached_input = input;
    recognizer->cached_probabilities = probabilities;
    recognizer->cached_input_element_count = input_element_count;
    recognizer->cached_probability_element_count = probability_element_count;
    recognizer->cached_target_width = target_width;
    recognizer->cached_time_steps = time_steps;
}

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

static lw_status configure_session(lw_recognizer* recognizer, uint32_t target_width,
                                   lw_session_info* configured_info, lw_error* error) {
    lw_tensor_desc input_desc;
    lw_tensor_desc output_desc;
    lw_session_info session_info;
    lw_session* session = NULL;
    float* input = NULL;
    float* probabilities = NULL;
    uint64_t input_element_count;
    uint64_t probability_element_count;
    lw_status status;

    if (recognizer->cached_session != NULL && recognizer->cached_target_width == target_width) {
        activate_cached_session(recognizer);
        if (configured_info != NULL) {
            lw_session_info_init(configured_info);
            status = lw_session_get_info(recognizer->session, configured_info);
            if (status != LW_STATUS_OK) {
                activate_cached_session(recognizer);
                lw_set_error(error, status, "unable to read recognizer session information");
                return status;
            }
        }
        return LW_STATUS_OK;
    }

    /* Keep at most two concrete widths. Discarding an inactive cache entry
     * before construction bounds peak memory to the old active and candidate
     * sessions; the active session remains valid if construction fails. */
    release_cached_session(recognizer);

    lw_tensor_desc_init(&input_desc);
    input_desc.dtype = LW_DTYPE_F32;
    input_desc.rank = 4u;
    input_desc.dimensions[0] = 1;
    input_desc.dimensions[1] = 3;
    input_desc.dimensions[2] = (int32_t)LW_REC_INPUT_HEIGHT;
    input_desc.dimensions[3] = (int32_t)target_width;
    status = lw_session_create(recognizer->model, &input_desc, 1u, &recognizer->session_options,
                               &session, error);
    if (status != LW_STATUS_OK) {
        return status;
    }

    lw_tensor_desc_init(&output_desc);
    status = lw_session_get_output_desc(session, 0u, &output_desc);
    if (status != LW_STATUS_OK || output_desc.dtype != LW_DTYPE_F32 || output_desc.rank != 3u ||
        output_desc.dimensions[0] != 1 || output_desc.dimensions[1] <= 0 ||
        output_desc.dimensions[2] <= 0 ||
        (uint32_t)output_desc.dimensions[2] !=
            lw_rec_dictionary_class_count(recognizer->dictionary)) {
        lw_session_free(session);
        lw_set_error(error, LW_STATUS_INVALID_SHAPE,
                     "recognizer model output and dictionary are incompatible");
        return LW_STATUS_INVALID_SHAPE;
    }

    input_element_count = (uint64_t)3u * LW_REC_INPUT_HEIGHT * target_width;
    probability_element_count =
        (uint64_t)(uint32_t)output_desc.dimensions[1] * (uint32_t)output_desc.dimensions[2];
    if (input_element_count > SIZE_MAX / sizeof(*input) ||
        probability_element_count > SIZE_MAX / sizeof(*probabilities)) {
        lw_session_free(session);
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "recognizer buffer size overflows");
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    input = (float*)malloc((size_t)input_element_count * sizeof(*input));
    probabilities = (float*)malloc((size_t)probability_element_count * sizeof(*probabilities));
    if (input == NULL || probabilities == NULL) {
        free(probabilities);
        free(input);
        lw_session_free(session);
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate recognizer buffers");
        return LW_STATUS_OUT_OF_MEMORY;
    }

    lw_session_info_init(&session_info);
    status = lw_session_get_info(session, &session_info);
    if (status != LW_STATUS_OK) {
        free(probabilities);
        free(input);
        lw_session_free(session);
        lw_set_error(error, status, "unable to read recognizer session information");
        return status;
    }

    /* Publish the new shape only after every allocation and validation has
     * succeeded. A failed adaptive switch therefore leaves the old session
     * usable and avoids a partially configured recognizer. */
    recognizer->cached_session = recognizer->session;
    recognizer->cached_input = recognizer->input;
    recognizer->cached_probabilities = recognizer->probabilities;
    recognizer->cached_input_element_count = recognizer->input_element_count;
    recognizer->cached_probability_element_count = recognizer->probability_element_count;
    recognizer->cached_target_width = recognizer->current_target_width;
    recognizer->cached_time_steps = recognizer->current_time_steps;
    recognizer->session = session;
    recognizer->input = input;
    recognizer->probabilities = probabilities;
    recognizer->input_element_count = input_element_count;
    recognizer->probability_element_count = probability_element_count;
    recognizer->current_target_width = target_width;
    recognizer->current_time_steps = (uint32_t)output_desc.dimensions[1];
    if (configured_info != NULL) {
        *configured_info = session_info;
    }
    return LW_STATUS_OK;
}

static uint32_t adaptive_target_width(uint32_t source_width, uint32_t source_height,
                                      uint32_t maximum_width) {
    static const uint32_t buckets[] = {192u, 320u, 480u, 640u, 960u};
    uint64_t scaled_width;
    uint32_t index;
    if (source_width == 0u || source_height == 0u || maximum_width <= 320u) {
        return maximum_width;
    }
    scaled_width =
        ((uint64_t)LW_REC_INPUT_HEIGHT * source_width + source_height - 1u) / source_height;
    for (index = 0u; index < sizeof(buckets) / sizeof(buckets[0]); ++index) {
        if (buckets[index] >= maximum_width) {
            break;
        }
        if (scaled_width <= buckets[index]) {
            return buckets[index];
        }
    }
    return maximum_width;
}

lw_status lw_recognizer_enable_adaptive_width(lw_recognizer* recognizer, uint32_t enabled,
                                              lw_error* error) {
    if (recognizer == NULL || enabled > 1u) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "recognizer and a boolean adaptive-width flag are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    recognizer->adaptive_width_enabled = enabled;
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}

uint32_t lw_recognizer_target_width_for_image(const lw_recognizer* recognizer,
                                              uint32_t source_width, uint32_t source_height) {
    if (recognizer == NULL) {
        return 0u;
    }
    if (recognizer->adaptive_width_enabled == 0u) {
        return recognizer->info.target_width;
    }
    return adaptive_target_width(source_width, source_height, recognizer->info.target_width);
}

uint32_t lw_recognizer_current_target_width(const lw_recognizer* recognizer) {
    return recognizer == NULL ? 0u : recognizer->current_target_width;
}

lw_status lw_recognizer_create(const char* model_path_utf8, const char* dictionary_path_utf8,
                               const lw_recognizer_options* options, lw_recognizer** out_recognizer,
                               lw_error* error) {
    lw_model_options model_options;
    lw_session_options session_options;
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
    recognizer->session_options = session_options;
    status = configure_session(recognizer, target_width, &session_info, error);
    if (status != LW_STATUS_OK) {
        goto fail;
    }
    max_label_bytes = lw_rec_dictionary_max_label_byte_count(recognizer->dictionary);
    if (max_label_bytes == 0u ||
        (uint64_t)recognizer->current_time_steps > (UINT64_MAX - 1u) / max_label_bytes) {
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "recognizer maximum text capacity overflows");
        status = LW_STATUS_OUT_OF_BOUNDS;
        goto fail;
    }
    lw_recognizer_info_init(&recognizer->info);
    recognizer->info.target_width = target_width;
    recognizer->info.input_height = LW_REC_INPUT_HEIGHT;
    recognizer->info.time_steps = recognizer->current_time_steps;
    recognizer->info.class_count = lw_rec_dictionary_class_count(recognizer->dictionary);
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
    release_cached_session(recognizer);
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

static lw_status recognizer_recognize_bgr_u8_impl(lw_recognizer* recognizer, const uint8_t* source,
                                                  uint64_t source_byte_count, uint32_t source_width,
                                                  uint32_t source_height, uint32_t source_stride,
                                                  char* text_utf8, uint64_t text_capacity,
                                                  lw_recognition_result* result,
                                                  lw_pipeline_component_profile* profile,
                                                  lw_error* error) {
    uint64_t source_pixels;
    uint32_t resized_width = 0u;
    uint64_t required_capacity = 0u;
    float score = 0.0f;
    uint32_t emitted_count = 0u;
    uint32_t target_width;
    uint64_t started;
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
    target_width = lw_recognizer_target_width_for_image(recognizer, source_width, source_height);
    if (target_width != recognizer->current_target_width) {
        status = configure_session(recognizer, target_width, NULL, error);
        if (status != LW_STATUS_OK) {
            return status;
        }
    }
    started = lw_pipeline_profile_now(profile);
    status =
        lw_rec_preprocess_bgr_u8(source, source_byte_count, source_width, source_height,
                                 source_stride, recognizer->current_target_width, recognizer->input,
                                 recognizer->input_element_count, &resized_width);
    if (status != LW_STATUS_OK) {
        lw_set_error(error, status, "BGR source layout is invalid");
        return status;
    }
    lw_pipeline_profile_add_elapsed(profile == NULL ? NULL : &profile->preprocess_nanoseconds,
                                    started, profile);
    started = lw_pipeline_profile_now(profile);
    status = profile == NULL
                 ? lw_execute_session_f32(
                       recognizer->session, recognizer->input, recognizer->input_element_count,
                       recognizer->probabilities, recognizer->probability_element_count, error)
                 : lw_execute_session_f32_profiled(
                       recognizer->session, recognizer->input, recognizer->input_element_count,
                       recognizer->probabilities, recognizer->probability_element_count,
                       &profile->execution, error);
    lw_pipeline_profile_add_elapsed(profile == NULL ? NULL : &profile->graph_nanoseconds, started,
                                    profile);
    if (status != LW_STATUS_OK) {
        return status;
    }
    started = lw_pipeline_profile_now(profile);
    if (text_utf8 != NULL && text_capacity >= recognizer->info.max_text_capacity) {
        status = lw_rec_ctc_decode_known_capacity_f32(
            recognizer->dictionary, recognizer->probabilities,
            recognizer->probability_element_count, recognizer->current_time_steps,
            recognizer->info.class_count, text_utf8, text_capacity, &required_capacity, &score,
            &emitted_count, error);
    } else {
        status = lw_rec_ctc_decode_f32(recognizer->dictionary, recognizer->probabilities,
                                       recognizer->probability_element_count,
                                       recognizer->current_time_steps, recognizer->info.class_count,
                                       text_utf8, text_capacity, &required_capacity, &score,
                                       &emitted_count, error);
    }
    result->emitted_count = emitted_count;
    result->score = score;
    result->resized_width = resized_width;
    result->time_steps = recognizer->current_time_steps;
    result->required_text_capacity = required_capacity;
    lw_pipeline_profile_add_elapsed(profile == NULL ? NULL : &profile->postprocess_nanoseconds,
                                    started, profile);
    return status;
}

lw_status lw_recognizer_recognize_bgr_u8(lw_recognizer* recognizer, const uint8_t* source,
                                         uint64_t source_byte_count, uint32_t source_width,
                                         uint32_t source_height, uint32_t source_stride,
                                         char* text_utf8, uint64_t text_capacity,
                                         lw_recognition_result* result, lw_error* error) {
    return recognizer_recognize_bgr_u8_impl(recognizer, source, source_byte_count, source_width,
                                            source_height, source_stride, text_utf8, text_capacity,
                                            result, NULL, error);
}

lw_status lw_recognizer_recognize_bgr_u8_profiled(lw_recognizer* recognizer, const uint8_t* source,
                                                  uint64_t source_byte_count, uint32_t source_width,
                                                  uint32_t source_height, uint32_t source_stride,
                                                  char* text_utf8, uint64_t text_capacity,
                                                  lw_recognition_result* result,
                                                  lw_pipeline_component_profile* profile,
                                                  lw_error* error) {
    if (profile == NULL || profile->execution.struct_size != sizeof(profile->execution) ||
        profile->execution.clock == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "an initialized recognizer profile and clock are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    return recognizer_recognize_bgr_u8_impl(recognizer, source, source_byte_count, source_width,
                                            source_height, source_stride, text_utf8, text_capacity,
                                            result, profile, error);
}
