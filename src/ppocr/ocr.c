#include "lw_infer.h"

#include "crop_internal.h"
#include "error_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define LW_OCR_DEFAULT_MAX_CROP_PIXELS UINT64_C(16000000)

struct lw_ocr {
    lw_detector* detector;
    lw_classifier* classifier;
    lw_recognizer* recognizer;
    lw_detection_box* detected_boxes;
    lw_ocr_line* scratch_lines;
    char* scratch_text;
    uint8_t* crop;
    uint64_t crop_capacity;
    uint64_t max_crop_pixels;
    uint64_t text_capacity_per_line;
    float classifier_threshold;
    lw_ocr_info info;
};

static void clear_result(lw_ocr_result* result) {
    memset(result, 0, sizeof(*result));
    result->struct_size = (uint32_t)sizeof(*result);
}

void lw_ocr_options_init(lw_ocr_options* options) {
    if (options == NULL) return;
    memset(options, 0, sizeof(*options));
    options->struct_size = (uint32_t)sizeof(*options);
    options->use_direction_classification = 1u;
    options->classifier_threshold = 0.9f;
    options->max_crop_pixels = LW_OCR_DEFAULT_MAX_CROP_PIXELS;
    lw_detector_options_init(&options->detector);
    lw_classifier_options_init(&options->classifier);
    lw_recognizer_options_init(&options->recognizer);
}

void lw_ocr_info_init(lw_ocr_info* info) {
    if (info == NULL) return;
    memset(info, 0, sizeof(*info));
    info->struct_size = (uint32_t)sizeof(*info);
}

void lw_ocr_result_init(lw_ocr_result* result) {
    if (result == NULL) return;
    clear_result(result);
}

static lw_status validate_options(
    const lw_ocr_options* options,
    lw_ocr_options* values,
    lw_error* error) {
    lw_ocr_options_init(values);
    if (options != NULL) {
        if (options->struct_size != sizeof(*options) || options->reserved != 0u) {
            lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                         "invalid OCR options structure");
            return LW_STATUS_INVALID_ARGUMENT;
        }
        *values = *options;
        if (values->max_crop_pixels == 0u)
            values->max_crop_pixels = LW_OCR_DEFAULT_MAX_CROP_PIXELS;
    }
    if (values->use_direction_classification > 1u ||
        !isfinite(values->classifier_threshold) ||
        values->classifier_threshold < 0.0f ||
        values->classifier_threshold > 1.0f ||
        values->detector.struct_size != sizeof(values->detector) ||
        values->classifier.struct_size != sizeof(values->classifier) ||
        values->recognizer.struct_size != sizeof(values->recognizer) ||
        values->detector.reserved != 0u ||
        values->classifier.reserved != 0u ||
        values->recognizer.reserved0 != 0u ||
        values->recognizer.reserved1 != 0u) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "OCR classifier policy or nested options are invalid");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    return LW_STATUS_OK;
}

lw_status lw_ocr_create(
    const char* detector_model_path_utf8,
    const char* classifier_model_path_utf8,
    const char* recognizer_model_path_utf8,
    const char* dictionary_path_utf8,
    const lw_ocr_options* options,
    lw_ocr** out_ocr,
    lw_error* error) {
    lw_ocr_options values;
    lw_detector_info detector_info;
    lw_recognizer_info recognizer_info;
    lw_ocr* ocr = NULL;
    uint64_t maximum_text_capacity;
    lw_status status;
    if (out_ocr != NULL) *out_ocr = NULL;
    if (detector_model_path_utf8 == NULL ||
        detector_model_path_utf8[0] == '\0' ||
        recognizer_model_path_utf8 == NULL ||
        recognizer_model_path_utf8[0] == '\0' ||
        dictionary_path_utf8 == NULL || dictionary_path_utf8[0] == '\0' ||
        out_ocr == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "DET model, REC model, dictionary, and output OCR handle are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    status = validate_options(options, &values, error);
    if (status != LW_STATUS_OK) return status;
    if (values.use_direction_classification != 0u &&
        (classifier_model_path_utf8 == NULL ||
         classifier_model_path_utf8[0] == '\0')) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "CLS model is required when direction classification is enabled");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    ocr = (lw_ocr*)calloc(1u, sizeof(*ocr));
    if (ocr == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY,
                     "unable to allocate OCR handle");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    status = lw_detector_create(detector_model_path_utf8, &values.detector,
                                &ocr->detector, error);
    if (status != LW_STATUS_OK) goto fail;
    if (values.use_direction_classification != 0u) {
        status = lw_classifier_create(classifier_model_path_utf8,
                                      &values.classifier,
                                      &ocr->classifier, error);
        if (status != LW_STATUS_OK) goto fail;
    }
    status = lw_recognizer_create(recognizer_model_path_utf8,
                                  dictionary_path_utf8, &values.recognizer,
                                  &ocr->recognizer, error);
    if (status != LW_STATUS_OK) goto fail;
    lw_detector_info_init(&detector_info);
    lw_recognizer_info_init(&recognizer_info);
    if (lw_detector_get_info(ocr->detector, &detector_info) != LW_STATUS_OK ||
        lw_recognizer_get_info(ocr->recognizer, &recognizer_info) != LW_STATUS_OK ||
        detector_info.max_candidates == 0u ||
        recognizer_info.max_text_capacity == 0u) {
        lw_set_error(error, LW_STATUS_INVALID_SHAPE,
                     "unable to query OCR component capacities");
        status = LW_STATUS_INVALID_SHAPE;
        goto fail;
    }
    if ((uint64_t)detector_info.max_candidates >
        UINT64_MAX / recognizer_info.max_text_capacity) {
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS,
                     "OCR maximum text capacity overflows");
        status = LW_STATUS_OUT_OF_BOUNDS;
        goto fail;
    }
    maximum_text_capacity = (uint64_t)detector_info.max_candidates *
        recognizer_info.max_text_capacity;
    if (detector_info.max_candidates > SIZE_MAX / sizeof(*ocr->detected_boxes) ||
        detector_info.max_candidates > SIZE_MAX / sizeof(*ocr->scratch_lines) ||
        maximum_text_capacity > SIZE_MAX) {
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS,
                     "OCR output scratch capacity exceeds the platform");
        status = LW_STATUS_OUT_OF_BOUNDS;
        goto fail;
    }
    ocr->detected_boxes = (lw_detection_box*)calloc(
        detector_info.max_candidates, sizeof(*ocr->detected_boxes));
    ocr->scratch_lines = (lw_ocr_line*)calloc(
        detector_info.max_candidates, sizeof(*ocr->scratch_lines));
    ocr->scratch_text = (char*)malloc((size_t)maximum_text_capacity);
    if (ocr->detected_boxes == NULL || ocr->scratch_lines == NULL ||
        ocr->scratch_text == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY,
                     "unable to allocate OCR output scratch buffers");
        status = LW_STATUS_OUT_OF_MEMORY;
        goto fail;
    }
    ocr->max_crop_pixels = values.max_crop_pixels;
    ocr->text_capacity_per_line = recognizer_info.max_text_capacity;
    ocr->classifier_threshold = values.classifier_threshold;
    lw_ocr_info_init(&ocr->info);
    ocr->info.use_direction_classification =
        values.use_direction_classification;
    ocr->info.max_line_capacity = detector_info.max_candidates;
    ocr->info.max_text_capacity = maximum_text_capacity;
    ocr->info.max_text_capacity_per_line = recognizer_info.max_text_capacity;
    ocr->info.max_crop_pixels = values.max_crop_pixels;
    *out_ocr = ocr;
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;

fail:
    lw_ocr_free(ocr);
    return status;
}

void lw_ocr_free(lw_ocr* ocr) {
    if (ocr == NULL) return;
    free(ocr->crop);
    free(ocr->scratch_text);
    free(ocr->scratch_lines);
    free(ocr->detected_boxes);
    lw_recognizer_free(ocr->recognizer);
    lw_classifier_free(ocr->classifier);
    lw_detector_free(ocr->detector);
    free(ocr);
}

lw_status lw_ocr_get_info(const lw_ocr* ocr, lw_ocr_info* info) {
    if (ocr == NULL || info == NULL || info->struct_size != sizeof(*info))
        return LW_STATUS_INVALID_ARGUMENT;
    *info = ocr->info;
    return LW_STATUS_OK;
}

static lw_status ensure_crop_capacity(
    lw_ocr* ocr,
    uint64_t byte_count,
    lw_error* error) {
    uint8_t* resized;
    if (byte_count <= ocr->crop_capacity) return LW_STATUS_OK;
    if (byte_count > SIZE_MAX) {
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS,
                     "OCR crop capacity exceeds the platform");
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    resized = (uint8_t*)realloc(ocr->crop, (size_t)byte_count);
    if (resized == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY,
                     "unable to allocate OCR crop buffer");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    ocr->crop = resized;
    ocr->crop_capacity = byte_count;
    return LW_STATUS_OK;
}

lw_status lw_ocr_run_bgr_u8(
    lw_ocr* ocr,
    const uint8_t* source,
    uint64_t source_byte_count,
    uint32_t source_width,
    uint32_t source_height,
    uint32_t source_stride,
    lw_ocr_line* lines,
    uint32_t line_capacity,
    char* text_utf8,
    uint64_t text_capacity,
    lw_ocr_result* result,
    lw_error* error) {
    lw_detection_result detection;
    uint32_t line_count = 0u;
    uint64_t text_used = 0u;
    uint32_t index;
    int capacity_query;
    lw_status status;
    if (ocr == NULL || source == NULL || result == NULL ||
        result->struct_size != sizeof(*result) ||
        (lines == NULL && line_capacity != 0u) ||
        (text_utf8 == NULL && text_capacity != 0u)) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "OCR handle, BGR source, initialized result, and valid output buffers are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    clear_result(result);
    capacity_query = lines == NULL && line_capacity == 0u &&
        text_utf8 == NULL && text_capacity == 0u;
    lw_detection_result_init(&detection);
    status = lw_detector_detect_bgr_u8(
        ocr->detector, source, source_byte_count, source_width, source_height,
        source_stride, ocr->detected_boxes, ocr->info.max_line_capacity,
        &detection, error);
    result->detected_count = detection.box_count;
    result->detector_resized_width = detection.resized_width;
    result->detector_resized_height = detection.resized_height;
    if (status != LW_STATUS_OK) return status;
    for (index = 0u; index < detection.box_count; ++index) {
        lw_detection_box* box = &ocr->detected_boxes[index];
        lw_classification_result classification;
        lw_recognition_result recognition;
        lw_ocr_line* line;
        uint32_t crop_width;
        uint32_t crop_height;
        uint64_t crop_bytes;
        uint64_t crop_pixels;
        char* line_text;
        status = lw_crop_quad_size(
            box, &crop_width, &crop_height, &crop_bytes);
        if (status == LW_STATUS_INVALID_SHAPE) continue;
        if (status != LW_STATUS_OK) {
            lw_set_error(error, status, "unable to compute OCR crop dimensions");
            return status;
        }
        crop_pixels = (uint64_t)crop_width * crop_height;
        if (crop_pixels > ocr->max_crop_pixels) {
            lw_set_error(error, LW_STATUS_MEMORY_LIMIT,
                         "OCR crop exceeds max_crop_pixels");
            return LW_STATUS_MEMORY_LIMIT;
        }
        if (crop_width > UINT32_MAX / 3u) {
            lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS,
                         "OCR crop row stride overflows");
            return LW_STATUS_OUT_OF_BOUNDS;
        }
        status = ensure_crop_capacity(ocr, crop_bytes, error);
        if (status != LW_STATUS_OK) return status;
        status = lw_crop_quad_bgr_u8(
            source, source_byte_count, source_width, source_height,
            source_stride, box, ocr->crop, ocr->crop_capacity,
            &crop_width, &crop_height, &crop_bytes);
        if (status == LW_STATUS_INVALID_SHAPE) continue;
        if (status != LW_STATUS_OK) {
            lw_set_error(error, status, "OCR perspective crop failed");
            return status;
        }
        memset(&classification, 0, sizeof(classification));
        if (ocr->classifier != NULL) {
            lw_classification_result_init(&classification);
            status = lw_classifier_classify_bgr_u8(
                ocr->classifier, ocr->crop, crop_bytes, crop_width,
                crop_height, crop_width * 3u, &classification, error);
            if (status != LW_STATUS_OK) return status;
            if ((classification.label & 1u) != 0u &&
                classification.score > ocr->classifier_threshold) {
                lw_rotate_bgr_u8_180(ocr->crop, crop_width, crop_height);
            }
        }
        if (text_used > ocr->info.max_text_capacity -
            ocr->text_capacity_per_line) {
            lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS,
                         "OCR text scratch capacity is exhausted");
            return LW_STATUS_OUT_OF_BOUNDS;
        }
        line_text = ocr->scratch_text + (size_t)text_used;
        lw_recognition_result_init(&recognition);
        status = lw_recognizer_recognize_bgr_u8(
            ocr->recognizer, ocr->crop, crop_bytes, crop_width, crop_height,
            crop_width * 3u, line_text, ocr->text_capacity_per_line,
            &recognition, error);
        if (status != LW_STATUS_OK) return status;
        if (recognition.required_text_capacity == 0u ||
            recognition.required_text_capacity > ocr->text_capacity_per_line) {
            lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS,
                         "recognizer returned an invalid text capacity");
            return LW_STATUS_OUT_OF_BOUNDS;
        }
        line = &ocr->scratch_lines[line_count];
        memset(line, 0, sizeof(*line));
        line->box = *box;
        line->recognition_score = recognition.score;
        line->classification_score = classification.score;
        line->classification_label = classification.label;
        line->applied_rotation_degrees =
            ocr->classifier != NULL && (classification.label & 1u) != 0u &&
            classification.score > ocr->classifier_threshold ? 180u : 0u;
        line->emitted_count = recognition.emitted_count;
        line->text_offset = text_used;
        line->text_length = recognition.required_text_capacity - 1u;
        text_used += recognition.required_text_capacity;
        ++line_count;
    }
    result->line_count = line_count;
    result->required_line_capacity = line_count;
    result->required_text_capacity = text_used;
    if (capacity_query) {
        lw_set_error(error, LW_STATUS_OK, "");
        return LW_STATUS_OK;
    }
    if (line_capacity < line_count || text_capacity < text_used ||
        (line_count != 0u && lines == NULL) ||
        (text_used != 0u && text_utf8 == NULL)) {
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS,
                     "OCR line or text output capacity is insufficient");
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    if (line_count != 0u)
        memcpy(lines, ocr->scratch_lines,
               (size_t)line_count * sizeof(*lines));
    if (text_used != 0u)
        memcpy(text_utf8, ocr->scratch_text, (size_t)text_used);
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}
