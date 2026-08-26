#include "lw_infer.h"

/* Public DET handle: preprocessing -> graph execution -> DB postprocessing. */

#include "det_internal.h"
#include "error_internal.h"
#include "executor_internal.h"
#include "profile_internal.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define LW_DET_DEFAULT_LIMIT_SIDE_LENGTH 960u
#define LW_DET_MAX_LIMIT_SIDE_LENGTH 4096u
#define LW_DET_DEFAULT_MAX_CANDIDATES 1000u
#define LW_DET_MAX_CANDIDATES 10000u
#define LW_DET_DEFAULT_MAX_IMAGE_PIXELS UINT64_C(40000000)

struct lw_detector {
    lw_model* model;
    lw_session* session;
    float* input;
    float* probabilities;
    uint64_t input_element_count;
    uint64_t probability_element_count;
    uint32_t resized_width;
    uint32_t resized_height;
    lw_session_options session_options;
    lw_detector_info info;
};

static void clear_result(lw_detection_result* result) {
    memset(result, 0, sizeof(*result));
    result->struct_size = (uint32_t)sizeof(*result);
}

void lw_detector_options_init(lw_detector_options* options) {
    lw_model_options model_options;
    lw_session_options session_options;
    if (options == NULL)
        return;
    lw_model_options_init(&model_options);
    lw_session_options_init(&session_options);
    memset(options, 0, sizeof(*options));
    options->struct_size = (uint32_t)sizeof(*options);
    options->limit_side_length = LW_DET_DEFAULT_LIMIT_SIDE_LENGTH;
    options->max_candidates = LW_DET_DEFAULT_MAX_CANDIDATES;
    options->bitmap_threshold = 0.3f;
    options->box_threshold = 0.6f;
    options->unclip_ratio = 1.6f;
    options->max_model_file_size = model_options.max_file_size;
    options->max_workspace_size = session_options.max_workspace_size;
    options->max_tensor_size = session_options.max_tensor_size;
    options->max_image_pixels = LW_DET_DEFAULT_MAX_IMAGE_PIXELS;
}

void lw_detector_info_init(lw_detector_info* info) {
    if (info == NULL)
        return;
    memset(info, 0, sizeof(*info));
    info->struct_size = (uint32_t)sizeof(*info);
}

void lw_detection_result_init(lw_detection_result* result) {
    if (result == NULL)
        return;
    clear_result(result);
}

static lw_status validate_options(const lw_detector_options* options, lw_detector_info* info,
                                  lw_model_options* model_options,
                                  lw_session_options* session_options, lw_error* error) {
    lw_detector_options values;
    lw_detector_options_init(&values);
    if (options != NULL) {
        if (options->struct_size != sizeof(*options) || options->reserved != 0u) {
            lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "invalid detector options structure");
            return LW_STATUS_INVALID_ARGUMENT;
        }
        values = *options;
        if (values.limit_side_length == 0u)
            values.limit_side_length = LW_DET_DEFAULT_LIMIT_SIDE_LENGTH;
        if (values.max_candidates == 0u)
            values.max_candidates = LW_DET_DEFAULT_MAX_CANDIDATES;
        if (values.max_image_pixels == 0u)
            values.max_image_pixels = LW_DET_DEFAULT_MAX_IMAGE_PIXELS;
    }
    if (values.limit_side_length < 32u || values.limit_side_length > LW_DET_MAX_LIMIT_SIDE_LENGTH ||
        values.max_candidates == 0u || values.max_candidates > LW_DET_MAX_CANDIDATES ||
        values.use_dilation > 1u || !isfinite(values.bitmap_threshold) ||
        values.bitmap_threshold < 0.0f || values.bitmap_threshold > 1.0f ||
        !isfinite(values.box_threshold) || values.box_threshold < 0.0f ||
        values.box_threshold > 1.0f || !isfinite(values.unclip_ratio) ||
        values.unclip_ratio <= 0.0f || values.unclip_ratio > 10.0f) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "detector thresholds or limits are invalid");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    lw_model_options_init(model_options);
    lw_session_options_init(session_options);
    if (values.max_model_file_size != 0u)
        model_options->max_file_size = values.max_model_file_size;
    if (values.max_workspace_size != 0u)
        session_options->max_workspace_size = values.max_workspace_size;
    if (values.max_tensor_size != 0u)
        session_options->max_tensor_size = values.max_tensor_size;
    lw_detector_info_init(info);
    info->limit_side_length = values.limit_side_length;
    info->max_candidates = values.max_candidates;
    info->use_dilation = values.use_dilation;
    info->bitmap_threshold = values.bitmap_threshold;
    info->box_threshold = values.box_threshold;
    info->unclip_ratio = values.unclip_ratio;
    info->max_image_pixels = values.max_image_pixels;
    return LW_STATUS_OK;
}

static lw_status ensure_session(lw_detector* detector, uint32_t width, uint32_t height,
                                lw_error* error) {
    lw_tensor_desc input_desc;
    lw_tensor_desc output_desc;
    lw_session* new_session = NULL;
    float* new_input = NULL;
    float* new_probabilities = NULL;
    uint64_t plane;
    uint64_t input_count;
    lw_status status;
    if (detector->session != NULL && detector->resized_width == width &&
        detector->resized_height == height) {
        return LW_STATUS_OK;
    }
    plane = (uint64_t)width * height;
    if (plane > UINT64_MAX / 3u || plane > SIZE_MAX / sizeof(float)) {
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "detector tensor size overflows");
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    input_count = plane * 3u;
    if (input_count > SIZE_MAX / sizeof(float)) {
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "detector input tensor size overflows");
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    lw_tensor_desc_init(&input_desc);
    input_desc.dtype = LW_DTYPE_F32;
    input_desc.rank = 4u;
    input_desc.dimensions[0] = 1;
    input_desc.dimensions[1] = 3;
    input_desc.dimensions[2] = (int32_t)height;
    input_desc.dimensions[3] = (int32_t)width;
    status = lw_session_create(detector->model, &input_desc, 1u, &detector->session_options,
                               &new_session, error);
    if (status != LW_STATUS_OK)
        goto fail;
    lw_tensor_desc_init(&output_desc);
    status = lw_session_get_output_desc(new_session, 0u, &output_desc);
    if (status != LW_STATUS_OK || output_desc.dtype != LW_DTYPE_F32 || output_desc.rank != 4u ||
        output_desc.dimensions[0] != 1 || output_desc.dimensions[1] != 1 ||
        output_desc.dimensions[2] != (int32_t)height ||
        output_desc.dimensions[3] != (int32_t)width) {
        lw_set_error(error, LW_STATUS_INVALID_SHAPE,
                     "detector model output must have shape [1,1,height,width]");
        status = LW_STATUS_INVALID_SHAPE;
        goto fail;
    }
    new_input = (float*)malloc((size_t)input_count * sizeof(*new_input));
    new_probabilities = (float*)malloc((size_t)plane * sizeof(*new_probabilities));
    if (new_input == NULL || new_probabilities == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate detector tensor buffers");
        status = LW_STATUS_OUT_OF_MEMORY;
        goto fail;
    }
    free(detector->probabilities);
    free(detector->input);
    lw_session_free(detector->session);
    detector->session = new_session;
    detector->input = new_input;
    detector->probabilities = new_probabilities;
    detector->input_element_count = input_count;
    detector->probability_element_count = plane;
    detector->resized_width = width;
    detector->resized_height = height;
    return LW_STATUS_OK;

fail:
    free(new_probabilities);
    free(new_input);
    lw_session_free(new_session);
    return status;
}

lw_status lw_detector_create(const char* model_path_utf8, const lw_detector_options* options,
                             lw_detector** out_detector, lw_error* error) {
    lw_model_options model_options;
    lw_detector* detector = NULL;
    lw_status status;
    if (out_detector != NULL)
        *out_detector = NULL;
    if (model_path_utf8 == NULL || model_path_utf8[0] == '\0' || out_detector == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "model path and output detector are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    detector = (lw_detector*)calloc(1u, sizeof(*detector));
    if (detector == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate detector handle");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    status = validate_options(options, &detector->info, &model_options, &detector->session_options,
                              error);
    if (status != LW_STATUS_OK)
        goto fail;
    status = lw_model_load(model_path_utf8, &model_options, &detector->model, error);
    if (status != LW_STATUS_OK)
        goto fail;
    status = ensure_session(detector, 32u, 32u, error);
    if (status != LW_STATUS_OK)
        goto fail;
    *out_detector = detector;
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;

fail:
    lw_detector_free(detector);
    return status;
}

void lw_detector_free(lw_detector* detector) {
    if (detector == NULL)
        return;
    free(detector->probabilities);
    free(detector->input);
    lw_session_free(detector->session);
    lw_model_free(detector->model);
    free(detector);
}

lw_status lw_detector_get_info(const lw_detector* detector, lw_detector_info* info) {
    if (detector == NULL || info == NULL || info->struct_size != sizeof(*info))
        return LW_STATUS_INVALID_ARGUMENT;
    *info = detector->info;
    return LW_STATUS_OK;
}

static lw_status detector_detect_bgr_u8_impl(
    lw_detector* detector, const uint8_t* source, uint64_t source_byte_count, uint32_t source_width,
    uint32_t source_height, uint32_t source_stride, lw_detection_box* boxes, uint32_t box_capacity,
    lw_detection_result* result, lw_pipeline_component_profile* profile, lw_error* error) {
    uint64_t source_pixels;
    uint32_t resized_width;
    uint32_t resized_height;
    uint32_t box_count = 0u;
    float width_ratio;
    float height_ratio;
    uint64_t started;
    lw_status status;
    if (detector == NULL || source == NULL || result == NULL ||
        result->struct_size != sizeof(*result) || (boxes == NULL && box_capacity != 0u)) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "detector, BGR source, initialized result, and valid box buffer are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    clear_result(result);
    if (source_width == 0u || source_height == 0u) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "source image dimensions must be positive");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    source_pixels = (uint64_t)source_width * source_height;
    if (source_pixels > detector->info.max_image_pixels) {
        lw_set_error(error, LW_STATUS_MEMORY_LIMIT, "source image exceeds max_image_pixels");
        return LW_STATUS_MEMORY_LIMIT;
    }
    started = lw_pipeline_profile_now(profile);
    status = lw_det_compute_size(source_width, source_height, detector->info.limit_side_length,
                                 &resized_width, &resized_height, &width_ratio, &height_ratio);
    if (status != LW_STATUS_OK) {
        lw_set_error(error, status, "unable to compute detector input size");
        return status;
    }
    result->resized_width = resized_width;
    result->resized_height = resized_height;
    result->width_ratio = width_ratio;
    result->height_ratio = height_ratio;
    status = ensure_session(detector, resized_width, resized_height, error);
    if (status != LW_STATUS_OK)
        return status;
    status = lw_det_preprocess_bgr_u8(source, source_byte_count, source_width, source_height,
                                      source_stride, resized_width, resized_height, detector->input,
                                      detector->input_element_count);
    if (status != LW_STATUS_OK) {
        lw_set_error(error, status, "BGR source layout is invalid");
        return status;
    }
    lw_pipeline_profile_add_elapsed(profile == NULL ? NULL : &profile->preprocess_nanoseconds,
                                    started, profile);
    started = lw_pipeline_profile_now(profile);
    status = profile == NULL
                 ? lw_execute_session_f32(detector->session, detector->input,
                                          detector->input_element_count, detector->probabilities,
                                          detector->probability_element_count, error)
                 : lw_execute_session_f32_profiled(
                       detector->session, detector->input, detector->input_element_count,
                       detector->probabilities, detector->probability_element_count,
                       &profile->execution, error);
    lw_pipeline_profile_add_elapsed(profile == NULL ? NULL : &profile->graph_nanoseconds, started,
                                    profile);
    if (status != LW_STATUS_OK) {
        return status;
    }
    started = lw_pipeline_profile_now(profile);
    status = lw_db_postprocess_f32(detector->probabilities, resized_width, resized_height,
                                   detector->info.bitmap_threshold, detector->info.box_threshold,
                                   detector->info.unclip_ratio, detector->info.use_dilation,
                                   detector->info.max_candidates, source_width, source_height,
                                   width_ratio, height_ratio, boxes, box_capacity, &box_count);
    lw_pipeline_profile_add_elapsed(profile == NULL ? NULL : &profile->postprocess_nanoseconds,
                                    started, profile);
    result->box_count = box_count;
    result->required_box_capacity = box_count;
    if (status != LW_STATUS_OK) {
        lw_set_error(error, status,
                     status == LW_STATUS_OUT_OF_BOUNDS ? "box buffer capacity is insufficient"
                                                       : "detector postprocessing failed");
        return status;
    }
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}

lw_status lw_detector_detect_bgr_u8(lw_detector* detector, const uint8_t* source,
                                    uint64_t source_byte_count, uint32_t source_width,
                                    uint32_t source_height, uint32_t source_stride,
                                    lw_detection_box* boxes, uint32_t box_capacity,
                                    lw_detection_result* result, lw_error* error) {
    return detector_detect_bgr_u8_impl(detector, source, source_byte_count, source_width,
                                       source_height, source_stride, boxes, box_capacity, result,
                                       NULL, error);
}

lw_status lw_detector_detect_bgr_u8_profiled(
    lw_detector* detector, const uint8_t* source, uint64_t source_byte_count, uint32_t source_width,
    uint32_t source_height, uint32_t source_stride, lw_detection_box* boxes, uint32_t box_capacity,
    lw_detection_result* result, lw_pipeline_component_profile* profile, lw_error* error) {
    if (profile == NULL || profile->execution.struct_size != sizeof(profile->execution) ||
        profile->execution.clock == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "an initialized detector profile and clock are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    return detector_detect_bgr_u8_impl(detector, source, source_byte_count, source_width,
                                       source_height, source_stride, boxes, box_capacity, result,
                                       profile, error);
}
