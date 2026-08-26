#include "lw_infer.h"

/*
 * Full OCR composition layer: DET -> perspective crop -> optional CLS -> REC.
 * Reusable scratch buffers belong to the handle, so one handle must not be
 * entered concurrently. Applications can create multiple handles for parallel
 * requests without adding locks inside the dependency-free core.
 */

#include "crop_internal.h"
#include "error_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <process.h>
#  include <windows.h>
#elif !defined(__EMSCRIPTEN__)
#  include <pthread.h>
#endif

#define LW_OCR_DEFAULT_MAX_CROP_PIXELS UINT64_C(16000000)
#define LW_OCR_MAX_WORKER_COUNT 16u

#if INTPTR_MAX > INT32_MAX && !defined(__EMSCRIPTEN__)
#  define LW_OCR_DEFAULT_WORKER_COUNT 4u
#else
#  define LW_OCR_DEFAULT_WORKER_COUNT 1u
#endif

typedef struct lw_ocr_crop {
    uint64_t offset;
    uint64_t byte_count;
    uint32_t width;
    uint32_t height;
    uint32_t box_index;
} lw_ocr_crop;

typedef struct lw_ocr_worker_task {
    lw_ocr* ocr;
    uint32_t worker_index;
    uint32_t begin;
    uint32_t end;
    lw_status status;
    lw_error error;
} lw_ocr_worker_task;

struct lw_ocr {
    lw_detector* detector;
    lw_classifier** classifiers;
    lw_recognizer** recognizers;
    uint32_t worker_count;
    lw_detection_box* detected_boxes;
    lw_ocr_line* scratch_lines;
    char* scratch_text;
    lw_ocr_crop* crops;
    lw_ocr_worker_task* worker_tasks;
#if defined(_WIN32)
    HANDLE* worker_threads;
#elif !defined(__EMSCRIPTEN__)
    pthread_t* worker_threads;
#endif
    uint8_t* worker_started;
    uint8_t* crop;
    uint64_t crop_capacity;
    uint64_t max_crop_pixels;
    uint64_t text_capacity_per_line;
    float classifier_threshold;
    lw_ocr_info info;
};

/* Use a width-neutral allocation check so GCC does not reject valid 64-bit
 * builds when a bounded uint32_t count is compared directly with SIZE_MAX. */
static int allocation_fits(uint64_t count, size_t element_size) {
    return element_size != 0u && count <= (uint64_t)(SIZE_MAX / element_size);
}

static void clear_result(lw_ocr_result* result) {
    memset(result, 0, sizeof(*result));
    result->struct_size = (uint32_t)sizeof(*result);
}

void lw_ocr_options_init(lw_ocr_options* options) {
    if (options == NULL)
        return;
    memset(options, 0, sizeof(*options));
    options->struct_size = (uint32_t)sizeof(*options);
    options->use_direction_classification = 1u;
    options->classifier_threshold = 0.9f;
    options->worker_count = LW_OCR_DEFAULT_WORKER_COUNT;
    options->max_crop_pixels = LW_OCR_DEFAULT_MAX_CROP_PIXELS;
    lw_detector_options_init(&options->detector);
    lw_classifier_options_init(&options->classifier);
    lw_recognizer_options_init(&options->recognizer);
}

void lw_ocr_info_init(lw_ocr_info* info) {
    if (info == NULL)
        return;
    memset(info, 0, sizeof(*info));
    info->struct_size = (uint32_t)sizeof(*info);
}

void lw_ocr_result_init(lw_ocr_result* result) {
    if (result == NULL)
        return;
    clear_result(result);
}

static lw_status validate_options(const lw_ocr_options* options, lw_ocr_options* values,
                                  lw_error* error) {
    lw_ocr_options_init(values);
    if (options != NULL) {
        if (options->struct_size != sizeof(*options)) {
            lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "invalid OCR options structure");
            return LW_STATUS_INVALID_ARGUMENT;
        }
        *values = *options;
        if (values->max_crop_pixels == 0u)
            values->max_crop_pixels = LW_OCR_DEFAULT_MAX_CROP_PIXELS;
    }
    if (values->use_direction_classification > 1u || !isfinite(values->classifier_threshold) ||
        values->classifier_threshold < 0.0f || values->classifier_threshold > 1.0f ||
        values->detector.struct_size != sizeof(values->detector) ||
        values->classifier.struct_size != sizeof(values->classifier) ||
        values->recognizer.struct_size != sizeof(values->recognizer) ||
        values->worker_count > LW_OCR_MAX_WORKER_COUNT || values->detector.reserved != 0u ||
        values->classifier.reserved != 0u || values->recognizer.reserved0 != 0u ||
        values->recognizer.reserved1 != 0u) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "OCR classifier policy or nested options are invalid");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    return LW_STATUS_OK;
}

lw_status lw_ocr_create(const char* detector_model_path_utf8,
                        const char* classifier_model_path_utf8,
                        const char* recognizer_model_path_utf8, const char* dictionary_path_utf8,
                        const lw_ocr_options* options, lw_ocr** out_ocr, lw_error* error) {
    lw_ocr_options values;
    lw_detector_info detector_info;
    lw_recognizer_info recognizer_info;
    lw_ocr* ocr = NULL;
    uint64_t maximum_text_capacity;
    lw_status status;
    if (out_ocr != NULL)
        *out_ocr = NULL;
    if (detector_model_path_utf8 == NULL || detector_model_path_utf8[0] == '\0' ||
        recognizer_model_path_utf8 == NULL || recognizer_model_path_utf8[0] == '\0' ||
        dictionary_path_utf8 == NULL || dictionary_path_utf8[0] == '\0' || out_ocr == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "DET model, REC model, dictionary, and output OCR handle are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    status = validate_options(options, &values, error);
    if (status != LW_STATUS_OK)
        return status;
    if (values.worker_count == 0u)
        values.worker_count = LW_OCR_DEFAULT_WORKER_COUNT;
#if defined(__EMSCRIPTEN__)
    if (values.worker_count != 1u) {
        lw_set_error(error, LW_STATUS_UNSUPPORTED,
                     "WebAssembly OCR requires worker_count=1 without pthreads");
        return LW_STATUS_UNSUPPORTED;
    }
#endif
    if (values.use_direction_classification != 0u &&
        (classifier_model_path_utf8 == NULL || classifier_model_path_utf8[0] == '\0')) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "CLS model is required when direction classification is enabled");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    ocr = (lw_ocr*)calloc(1u, sizeof(*ocr));
    if (ocr == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate OCR handle");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    status = lw_detector_create(detector_model_path_utf8, &values.detector, &ocr->detector, error);
    if (status != LW_STATUS_OK)
        goto fail;
    ocr->worker_count = values.worker_count;
    ocr->classifiers = (lw_classifier**)calloc(ocr->worker_count, sizeof(*ocr->classifiers));
    ocr->recognizers = (lw_recognizer**)calloc(ocr->worker_count, sizeof(*ocr->recognizers));
    ocr->worker_tasks = (lw_ocr_worker_task*)calloc(ocr->worker_count, sizeof(*ocr->worker_tasks));
    ocr->worker_started = (uint8_t*)calloc(ocr->worker_count, sizeof(*ocr->worker_started));
#if defined(_WIN32) || !defined(__EMSCRIPTEN__)
    ocr->worker_threads = calloc(ocr->worker_count, sizeof(*ocr->worker_threads));
#endif
    if (ocr->classifiers == NULL || ocr->recognizers == NULL || ocr->worker_tasks == NULL ||
        ocr->worker_started == NULL
#if defined(_WIN32) || !defined(__EMSCRIPTEN__)
        || ocr->worker_threads == NULL
#endif
    ) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate OCR worker pool");
        status = LW_STATUS_OUT_OF_MEMORY;
        goto fail;
    }
    {
        uint32_t worker_index;
        for (worker_index = 0u; worker_index < ocr->worker_count; ++worker_index) {
            if (values.use_direction_classification != 0u) {
                status = lw_classifier_create(classifier_model_path_utf8, &values.classifier,
                                              &ocr->classifiers[worker_index], error);
                if (status != LW_STATUS_OK)
                    goto fail;
            }
            status =
                lw_recognizer_create(recognizer_model_path_utf8, dictionary_path_utf8,
                                     &values.recognizer, &ocr->recognizers[worker_index], error);
            if (status != LW_STATUS_OK)
                goto fail;
        }
    }
    lw_detector_info_init(&detector_info);
    lw_recognizer_info_init(&recognizer_info);
    if (lw_detector_get_info(ocr->detector, &detector_info) != LW_STATUS_OK ||
        lw_recognizer_get_info(ocr->recognizers[0], &recognizer_info) != LW_STATUS_OK ||
        detector_info.max_candidates == 0u || recognizer_info.max_text_capacity == 0u) {
        lw_set_error(error, LW_STATUS_INVALID_SHAPE, "unable to query OCR component capacities");
        status = LW_STATUS_INVALID_SHAPE;
        goto fail;
    }
    if ((uint64_t)detector_info.max_candidates > UINT64_MAX / recognizer_info.max_text_capacity) {
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "OCR maximum text capacity overflows");
        status = LW_STATUS_OUT_OF_BOUNDS;
        goto fail;
    }
    maximum_text_capacity =
        (uint64_t)detector_info.max_candidates * recognizer_info.max_text_capacity;
    if (!allocation_fits(detector_info.max_candidates, sizeof(*ocr->detected_boxes)) ||
        !allocation_fits(detector_info.max_candidates, sizeof(*ocr->scratch_lines)) ||
        maximum_text_capacity > SIZE_MAX) {
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS,
                     "OCR output scratch capacity exceeds the platform");
        status = LW_STATUS_OUT_OF_BOUNDS;
        goto fail;
    }
    ocr->detected_boxes =
        (lw_detection_box*)calloc(detector_info.max_candidates, sizeof(*ocr->detected_boxes));
    ocr->scratch_lines =
        (lw_ocr_line*)calloc(detector_info.max_candidates, sizeof(*ocr->scratch_lines));
    ocr->crops = (lw_ocr_crop*)calloc(detector_info.max_candidates, sizeof(*ocr->crops));
    ocr->scratch_text = (char*)malloc((size_t)maximum_text_capacity);
    if (ocr->detected_boxes == NULL || ocr->scratch_lines == NULL || ocr->crops == NULL ||
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
    ocr->info.use_direction_classification = values.use_direction_classification;
    ocr->info.max_line_capacity = detector_info.max_candidates;
    ocr->info.worker_count = ocr->worker_count;
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
    uint32_t worker_index;
    if (ocr == NULL)
        return;
    free(ocr->crop);
    free(ocr->worker_started);
#if defined(_WIN32) || !defined(__EMSCRIPTEN__)
    free(ocr->worker_threads);
#endif
    free(ocr->worker_tasks);
    free(ocr->crops);
    free(ocr->scratch_text);
    free(ocr->scratch_lines);
    free(ocr->detected_boxes);
    for (worker_index = 0u; worker_index < ocr->worker_count; ++worker_index) {
        lw_recognizer_free(ocr->recognizers == NULL ? NULL : ocr->recognizers[worker_index]);
        lw_classifier_free(ocr->classifiers == NULL ? NULL : ocr->classifiers[worker_index]);
    }
    free(ocr->recognizers);
    free(ocr->classifiers);
    lw_detector_free(ocr->detector);
    free(ocr);
}

lw_status lw_ocr_get_info(const lw_ocr* ocr, lw_ocr_info* info) {
    if (ocr == NULL || info == NULL || info->struct_size != sizeof(*info))
        return LW_STATUS_INVALID_ARGUMENT;
    *info = ocr->info;
    return LW_STATUS_OK;
}

static lw_status ensure_crop_capacity(lw_ocr* ocr, uint64_t byte_count, lw_error* error) {
    uint8_t* resized;
    if (byte_count <= ocr->crop_capacity)
        return LW_STATUS_OK;
    if (byte_count > SIZE_MAX) {
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "OCR crop capacity exceeds the platform");
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    resized = (uint8_t*)realloc(ocr->crop, (size_t)byte_count);
    if (resized == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate OCR crop buffer");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    ocr->crop = resized;
    ocr->crop_capacity = byte_count;
    return LW_STATUS_OK;
}

/* Each worker owns its CLS and REC handles, while crop/output slots are
 * disjoint. This keeps the public OCR handle non-reentrant but lets one request
 * recognize independent text lines concurrently. */
static void process_worker_task(lw_ocr_worker_task* task) {
    lw_ocr* ocr = task->ocr;
    uint32_t index;
    task->status = LW_STATUS_OK;
    lw_error_init(&task->error);
    for (index = task->begin; index < task->end; ++index) {
        lw_ocr_crop* crop = &ocr->crops[index];
        lw_detection_box* box = &ocr->detected_boxes[crop->box_index];
        lw_classification_result classification;
        lw_recognition_result recognition;
        lw_ocr_line* line = &ocr->scratch_lines[index];
        uint8_t* crop_pixels = ocr->crop + (size_t)crop->offset;
        uint64_t text_offset = (uint64_t)index * ocr->text_capacity_per_line;
        char* line_text = ocr->scratch_text + (size_t)text_offset;

        memset(&classification, 0, sizeof(classification));
        if (ocr->classifiers[task->worker_index] != NULL) {
            lw_classification_result_init(&classification);
            task->status = lw_classifier_classify_bgr_u8(
                ocr->classifiers[task->worker_index], crop_pixels, crop->byte_count, crop->width,
                crop->height, crop->width * 3u, &classification, &task->error);
            if (task->status != LW_STATUS_OK)
                return;
            if ((classification.label & 1u) != 0u &&
                classification.score > ocr->classifier_threshold) {
                lw_rotate_bgr_u8_180(crop_pixels, crop->width, crop->height);
            }
        }

        lw_recognition_result_init(&recognition);
        task->status = lw_recognizer_recognize_bgr_u8(
            ocr->recognizers[task->worker_index], crop_pixels, crop->byte_count, crop->width,
            crop->height, crop->width * 3u, line_text, ocr->text_capacity_per_line, &recognition,
            &task->error);
        if (task->status != LW_STATUS_OK)
            return;
        if (recognition.required_text_capacity == 0u ||
            recognition.required_text_capacity > ocr->text_capacity_per_line) {
            task->status = LW_STATUS_OUT_OF_BOUNDS;
            lw_set_error(&task->error, task->status,
                         "recognizer returned an invalid text capacity");
            return;
        }

        memset(line, 0, sizeof(*line));
        line->box = *box;
        line->recognition_score = recognition.score;
        line->classification_score = classification.score;
        line->classification_label = classification.label;
        line->applied_rotation_degrees = ocr->classifiers[task->worker_index] != NULL &&
                                                 (classification.label & 1u) != 0u &&
                                                 classification.score > ocr->classifier_threshold
                                             ? 180u
                                             : 0u;
        line->emitted_count = recognition.emitted_count;
        line->text_offset = text_offset;
        line->text_length = recognition.required_text_capacity - 1u;
    }
}

#if defined(_WIN32)
static unsigned __stdcall worker_entry(void* context) {
    process_worker_task((lw_ocr_worker_task*)context);
    return 0u;
}
#elif !defined(__EMSCRIPTEN__)
static void* worker_entry(void* context) {
    process_worker_task((lw_ocr_worker_task*)context);
    return NULL;
}
#endif

static lw_status run_worker_tasks(lw_ocr* ocr, uint32_t first_crop, uint32_t crop_count,
                                  lw_error* error) {
    uint32_t active_workers = crop_count < ocr->worker_count ? crop_count : ocr->worker_count;
    uint32_t base_count;
    uint32_t remainder;
    uint32_t begin = first_crop;
    uint32_t worker_index;
    lw_status status = LW_STATUS_OK;
    if (active_workers == 0u)
        return LW_STATUS_OK;
    base_count = crop_count / active_workers;
    remainder = crop_count % active_workers;
    memset(ocr->worker_started, 0, ocr->worker_count * sizeof(*ocr->worker_started));
    for (worker_index = 0u; worker_index < active_workers; ++worker_index) {
        uint32_t count = base_count + (worker_index < remainder ? 1u : 0u);
        lw_ocr_worker_task* task = &ocr->worker_tasks[worker_index];
        memset(task, 0, sizeof(*task));
        task->ocr = ocr;
        task->worker_index = worker_index;
        task->begin = begin;
        task->end = begin + count;
        begin += count;
    }

    /* Keep worker zero on the calling thread and launch the remaining ranges.
     * If a platform thread cannot be created, execute that range synchronously
     * so the request remains correct instead of returning a partial result. */
    for (worker_index = 1u; worker_index < active_workers; ++worker_index) {
#if defined(_WIN32)
        ocr->worker_threads[worker_index] = (HANDLE)_beginthreadex(
            NULL, 0u, worker_entry, &ocr->worker_tasks[worker_index], 0u, NULL);
        if (ocr->worker_threads[worker_index] != NULL) {
            ocr->worker_started[worker_index] = 1u;
        } else {
            process_worker_task(&ocr->worker_tasks[worker_index]);
        }
#elif !defined(__EMSCRIPTEN__)
        if (pthread_create(&ocr->worker_threads[worker_index], NULL, worker_entry,
                           &ocr->worker_tasks[worker_index]) == 0) {
            ocr->worker_started[worker_index] = 1u;
        } else {
            process_worker_task(&ocr->worker_tasks[worker_index]);
        }
#else
        process_worker_task(&ocr->worker_tasks[worker_index]);
#endif
    }
    process_worker_task(&ocr->worker_tasks[0]);
    for (worker_index = 1u; worker_index < active_workers; ++worker_index) {
        if (ocr->worker_started[worker_index] == 0u)
            continue;
#if defined(_WIN32)
        WaitForSingleObject(ocr->worker_threads[worker_index], INFINITE);
        CloseHandle(ocr->worker_threads[worker_index]);
        ocr->worker_threads[worker_index] = NULL;
#elif !defined(__EMSCRIPTEN__)
        (void)pthread_join(ocr->worker_threads[worker_index], NULL);
#endif
    }
    for (worker_index = 0u; worker_index < active_workers; ++worker_index) {
        if (ocr->worker_tasks[worker_index].status != LW_STATUS_OK) {
            status = ocr->worker_tasks[worker_index].status;
            lw_set_error(error, status, ocr->worker_tasks[worker_index].error.message);
            break;
        }
    }
    return status;
}

lw_status lw_ocr_run_bgr_u8(lw_ocr* ocr, const uint8_t* source, uint64_t source_byte_count,
                            uint32_t source_width, uint32_t source_height, uint32_t source_stride,
                            lw_ocr_line* lines, uint32_t line_capacity, char* text_utf8,
                            uint64_t text_capacity, lw_ocr_result* result, lw_error* error) {
    lw_detection_result detection;
    uint32_t line_count = 0u;
    uint32_t batch_begin = 0u;
    uint64_t text_used = 0u;
    uint64_t crop_used = 0u;
    uint32_t index;
    int capacity_query;
    lw_status status;
    if (ocr == NULL || source == NULL || result == NULL || result->struct_size != sizeof(*result) ||
        (lines == NULL && line_capacity != 0u) || (text_utf8 == NULL && text_capacity != 0u)) {
        lw_set_error(
            error, LW_STATUS_INVALID_ARGUMENT,
            "OCR handle, BGR source, initialized result, and valid output buffers are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    clear_result(result);
    capacity_query =
        lines == NULL && line_capacity == 0u && text_utf8 == NULL && text_capacity == 0u;

    /* DET writes into handle-owned scratch space so a capacity-only call still
     * executes the exact same pipeline and reports exact output requirements. */
    lw_detection_result_init(&detection);
    status = lw_detector_detect_bgr_u8(ocr->detector, source, source_byte_count, source_width,
                                       source_height, source_stride, ocr->detected_boxes,
                                       ocr->info.max_line_capacity, &detection, error);
    result->detected_count = detection.box_count;
    result->detector_resized_width = detection.resized_width;
    result->detector_resized_height = detection.resized_height;
    if (status != LW_STATUS_OK)
        return status;
    /* Crop all boxes first. Offsets, rather than raw pointers, survive realloc
     * while the aggregate crop buffer grows. */
    for (index = 0u; index < detection.box_count; ++index) {
        lw_detection_box* box = &ocr->detected_boxes[index];
        lw_ocr_crop* crop;
        uint32_t crop_width;
        uint32_t crop_height;
        uint64_t crop_bytes;
        uint64_t crop_pixels;
        status = lw_crop_quad_size(box, &crop_width, &crop_height, &crop_bytes);
        if (status == LW_STATUS_INVALID_SHAPE)
            continue;
        if (status != LW_STATUS_OK) {
            lw_set_error(error, status, "unable to compute OCR crop dimensions");
            return status;
        }
        crop_pixels = (uint64_t)crop_width * crop_height;
        if (crop_pixels > ocr->max_crop_pixels) {
            lw_set_error(error, LW_STATUS_MEMORY_LIMIT, "OCR crop exceeds max_crop_pixels");
            return LW_STATUS_MEMORY_LIMIT;
        }
        if (crop_width > UINT32_MAX / 3u) {
            lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "OCR crop row stride overflows");
            return LW_STATUS_OUT_OF_BOUNDS;
        }
        if (crop_used > UINT64_MAX - crop_bytes) {
            lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "aggregate OCR crop size overflows");
            return LW_STATUS_OUT_OF_BOUNDS;
        }
        status = ensure_crop_capacity(ocr, crop_used + crop_bytes, error);
        if (status != LW_STATUS_OK)
            return status;
        status = lw_crop_quad_bgr_u8(source, source_byte_count, source_width, source_height,
                                     source_stride, box, ocr->crop + (size_t)crop_used, crop_bytes,
                                     &crop_width, &crop_height, &crop_bytes);
        if (status == LW_STATUS_INVALID_SHAPE)
            continue;
        if (status != LW_STATUS_OK) {
            lw_set_error(error, status, "OCR perspective crop failed");
            return status;
        }
        crop = &ocr->crops[line_count];
        crop->offset = crop_used;
        crop->byte_count = crop_bytes;
        crop->width = crop_width;
        crop->height = crop_height;
        crop->box_index = index;
        crop_used += crop_bytes;
        ++line_count;
        if (line_count - batch_begin == ocr->worker_count) {
            status = run_worker_tasks(ocr, batch_begin, line_count - batch_begin, error);
            if (status != LW_STATUS_OK)
                return status;
            batch_begin = line_count;
            crop_used = 0u;
        }
    }
    if (line_count != batch_begin) {
        status = run_worker_tasks(ocr, batch_begin, line_count - batch_begin, error);
        if (status != LW_STATUS_OK)
            return status;
    }
    /* Workers write fixed-size text slots to avoid synchronization. Compact
     * those slots in reading order before publishing the public result. */
    for (index = 0u; index < line_count; ++index) {
        lw_ocr_line* line = &ocr->scratch_lines[index];
        uint64_t line_bytes = line->text_length + 1u;
        if (text_used > ocr->info.max_text_capacity - line_bytes) {
            lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "OCR text scratch capacity is exhausted");
            return LW_STATUS_OUT_OF_BOUNDS;
        }
        memmove(ocr->scratch_text + (size_t)text_used,
                ocr->scratch_text + (size_t)line->text_offset, (size_t)line_bytes);
        line->text_offset = text_used;
        text_used += line_bytes;
    }
    result->line_count = line_count;
    result->required_line_capacity = line_count;
    result->required_text_capacity = text_used;
    if (capacity_query) {
        /* The caller can now allocate exact line/text buffers and call again. */
        lw_set_error(error, LW_STATUS_OK, "");
        return LW_STATUS_OK;
    }
    if (line_capacity < line_count || text_capacity < text_used ||
        (line_count != 0u && lines == NULL) || (text_used != 0u && text_utf8 == NULL)) {
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS,
                     "OCR line or text output capacity is insufficient");
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    /* Publish only after every line succeeded, preventing partially filled
     * caller buffers from looking like a successful OCR response. */
    if (line_count != 0u)
        memcpy(lines, ocr->scratch_lines, (size_t)line_count * sizeof(*lines));
    if (text_used != 0u)
        memcpy(text_utf8, ocr->scratch_text, (size_t)text_used);
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}
