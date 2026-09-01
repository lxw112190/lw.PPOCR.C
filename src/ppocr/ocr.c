#include "lw_infer.h"

/*
 * Full OCR composition layer: DET -> perspective crop -> optional CLS -> REC.
 * Reusable scratch buffers belong to the handle, so one handle must not be
 * entered concurrently. Applications can create multiple handles for parallel
 * requests without adding locks inside the dependency-free core.
 */

#include "crop_internal.h"
#include "cpu_topology.h"
#include "det_internal.h"
#include "error_internal.h"
#include "ocr_internal.h"
#include "profile_internal.h"
#include "rec_internal.h"

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
#  include <unistd.h>
#endif

#define LW_OCR_DEFAULT_MAX_CROP_PIXELS UINT64_C(16000000)
#define LW_OCR_MAX_WORKER_COUNT 16u

typedef struct lw_ocr_crop {
    uint64_t offset;
    uint64_t byte_count;
    uint32_t width;
    uint32_t height;
    uint32_t box_index;
    uint32_t target_width;
    uint8_t scheduled;
} lw_ocr_crop;

typedef struct lw_ocr_worker_task {
    lw_ocr* ocr;
    uint32_t worker_index;
    uint32_t initial_crop_slot;
    lw_status status;
    lw_error error;
    uint32_t profile_enabled;
    uint64_t wall_nanoseconds;
    uint64_t rec_width_sample_count;
    uint64_t rec_resized_width_sum;
    uint64_t rec_target_width_sum;
    uint64_t rec_width_histogram[LW_REC_WIDTH_HISTOGRAM_BUCKET_COUNT];
    lw_pipeline_component_profile classifier_profile;
    lw_pipeline_component_profile recognizer_profile;
} lw_ocr_worker_task;

struct lw_ocr {
    lw_detector* detector;
    lw_classifier** classifiers;
    lw_recognizer** recognizers;
    uint32_t worker_count;
    uint32_t det_intra_op_thread_cap;
    lw_cpu_topology cpu_topology;
    lw_detection_box* detected_boxes;
    lw_ocr_line* scratch_lines;
    char* scratch_text;
    lw_ocr_crop* crops;
    uint32_t* crop_schedule;
    uint32_t crop_schedule_count;
#if defined(_WIN32)
    volatile LONG next_crop_slot;
#else
    uint32_t next_crop_slot;
#endif
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

static uint32_t default_worker_count(void) {
    lw_cpu_topology topology;
    lw_cpu_topology_detect(&topology);
    return lw_parallel_default_line_worker_count(&topology);
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
    options->worker_count = default_worker_count();
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
        values.worker_count = default_worker_count();
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
    lw_cpu_topology_detect(&ocr->cpu_topology);
    ocr->det_intra_op_thread_cap = lw_parallel_default_det_thread_count(&ocr->cpu_topology);
    lw_detector_set_intra_op_thread_count(ocr->detector, ocr->det_intra_op_thread_cap);
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
            if (values.recognizer.target_width > 320u) {
                status =
                    lw_recognizer_enable_adaptive_width(ocr->recognizers[worker_index], 1u, error);
                if (status != LW_STATUS_OK)
                    goto fail;
            }
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
    ocr->crop_schedule =
        (uint32_t*)malloc((size_t)detector_info.max_candidates * sizeof(*ocr->crop_schedule));
    ocr->scratch_text = (char*)malloc((size_t)maximum_text_capacity);
    if (ocr->detected_boxes == NULL || ocr->scratch_lines == NULL || ocr->crops == NULL ||
        ocr->crop_schedule == NULL || ocr->scratch_text == NULL) {
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

void lw_ocr_set_det_intra_op_thread_count(lw_ocr* ocr, uint32_t thread_count) {
    if (ocr == NULL) {
        return;
    }
    if (thread_count == 0u) {
        thread_count = lw_parallel_default_det_thread_count(&ocr->cpu_topology);
    }
    if (thread_count > LW_OCR_MAX_WORKER_COUNT) {
        thread_count = LW_OCR_MAX_WORKER_COUNT;
    }
    ocr->det_intra_op_thread_cap = thread_count;
    lw_detector_set_intra_op_thread_count(ocr->detector, thread_count);
}

uint32_t lw_ocr_get_det_intra_op_thread_count(const lw_ocr* ocr) {
    return ocr == NULL ? 1u : lw_detector_get_intra_op_thread_count(ocr->detector);
}

uint32_t lw_ocr_get_det_intra_op_thread_cap(const lw_ocr* ocr) {
    return ocr == NULL ? 1u : ocr->det_intra_op_thread_cap;
}

void lw_ocr_get_cpu_topology(const lw_ocr* ocr, lw_cpu_topology* topology) {
    if (topology == NULL) {
        return;
    }
    if (ocr == NULL) {
        memset(topology, 0, sizeof(*topology));
        return;
    }
    *topology = ocr->cpu_topology;
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
    free(ocr->crop_schedule);
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
    task->status = LW_STATUS_OK;
    lw_error_init(&task->error);
    for (;;) {
        uint32_t slot;
        uint32_t index;
        if (task->initial_crop_slot != UINT32_MAX) {
            slot = task->initial_crop_slot;
            task->initial_crop_slot = UINT32_MAX;
        } else {
#if defined(_WIN32)
            slot = (uint32_t)(InterlockedIncrement(&ocr->next_crop_slot) - 1);
#elif defined(__EMSCRIPTEN__)
            slot = ocr->next_crop_slot++;
#else
            slot = __atomic_fetch_add(&ocr->next_crop_slot, 1u, __ATOMIC_RELAXED);
#endif
        }
        if (slot >= ocr->crop_schedule_count)
            return;
        index = ocr->crop_schedule[slot];
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
            task->status = task->profile_enabled == 0u
                               ? lw_classifier_classify_bgr_u8(
                                     ocr->classifiers[task->worker_index], crop_pixels,
                                     crop->byte_count, crop->width, crop->height, crop->width * 3u,
                                     &classification, &task->error)
                               : lw_classifier_classify_bgr_u8_profiled(
                                     ocr->classifiers[task->worker_index], crop_pixels,
                                     crop->byte_count, crop->width, crop->height, crop->width * 3u,
                                     &classification, &task->classifier_profile, &task->error);
            if (task->status != LW_STATUS_OK)
                return;
            if ((classification.label & 1u) != 0u &&
                classification.score > ocr->classifier_threshold) {
                lw_rotate_bgr_u8_180(crop_pixels, crop->width, crop->height);
            }
        }

        lw_recognition_result_init(&recognition);
        task->status = task->profile_enabled == 0u
                           ? lw_recognizer_recognize_bgr_u8(
                                 ocr->recognizers[task->worker_index], crop_pixels,
                                 crop->byte_count, crop->width, crop->height, crop->width * 3u,
                                 line_text, ocr->text_capacity_per_line, &recognition, &task->error)
                           : lw_recognizer_recognize_bgr_u8_profiled(
                                 ocr->recognizers[task->worker_index], crop_pixels,
                                 crop->byte_count, crop->width, crop->height, crop->width * 3u,
                                 line_text, ocr->text_capacity_per_line, &recognition,
                                 &task->recognizer_profile, &task->error);
        if (task->status != LW_STATUS_OK)
            return;
        if (recognition.required_text_capacity == 0u ||
            recognition.required_text_capacity > ocr->text_capacity_per_line) {
            task->status = LW_STATUS_OUT_OF_BOUNDS;
            lw_set_error(&task->error, task->status,
                         "recognizer returned an invalid text capacity");
            return;
        }
        if (task->profile_enabled != 0u) {
            uint32_t bucket = lw_rec_width_histogram_bucket(recognition.resized_width);
            lw_profile_add_value(&task->rec_width_sample_count, 1u);
            lw_profile_add_value(&task->rec_resized_width_sum, recognition.resized_width);
            lw_profile_add_value(
                &task->rec_target_width_sum,
                lw_recognizer_current_target_width(ocr->recognizers[task->worker_index]));
            lw_profile_add_value(&task->rec_width_histogram[bucket], 1u);
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

static void execute_worker_task(lw_ocr_worker_task* task) {
    uint64_t started =
        task->profile_enabled == 0u ? 0u : lw_pipeline_profile_now(&task->recognizer_profile);
    process_worker_task(task);
    if (task->profile_enabled != 0u) {
        uint64_t finished = lw_pipeline_profile_now(&task->recognizer_profile);
        task->wall_nanoseconds = finished >= started ? finished - started : 0u;
    }
}

#if defined(_WIN32)
static unsigned __stdcall worker_entry(void* context) {
    execute_worker_task((lw_ocr_worker_task*)context);
    return 0u;
}
#elif !defined(__EMSCRIPTEN__)
static void* worker_entry(void* context) {
    execute_worker_task((lw_ocr_worker_task*)context);
    return NULL;
}
#endif

static lw_status run_worker_tasks(lw_ocr* ocr, uint32_t crop_count,
                                  lw_ocr_execution_profile* profile, lw_error* error) {
    uint8_t worker_used[LW_OCR_MAX_WORKER_COUNT];
    uint32_t active_worker_count;
    uint32_t slot;
    uint64_t started = lw_ocr_profile_now(profile);
    uint64_t critical_nanoseconds = 0u;
    lw_status status = LW_STATUS_OK;
    if (crop_count == 0u)
        return LW_STATUS_OK;
    if (crop_count > ocr->info.max_line_capacity) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "invalid OCR worker crop schedule");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    active_worker_count = crop_count < ocr->worker_count ? crop_count : ocr->worker_count;
    memset(ocr->worker_started, 0, ocr->worker_count * sizeof(*ocr->worker_started));
    memset(worker_used, 0, sizeof(worker_used));
    ocr->crop_schedule_count = crop_count;
#if defined(_WIN32)
    ocr->next_crop_slot = (LONG)active_worker_count;
#else
    ocr->next_crop_slot = active_worker_count;
#endif
    for (slot = 0u; slot < active_worker_count; ++slot) {
        uint32_t worker_index;
        uint32_t candidate;
        uint32_t crop_index = ocr->crop_schedule[slot];
        uint32_t target_width = ocr->crops[crop_index].target_width;
        worker_index = active_worker_count;
        for (candidate = 0u; candidate < active_worker_count; ++candidate) {
            if (worker_used[candidate] == 0u &&
                lw_recognizer_current_target_width(ocr->recognizers[candidate]) == target_width) {
                worker_index = candidate;
                break;
            }
        }
        if (worker_index == active_worker_count) {
            for (candidate = 0u; candidate < active_worker_count; ++candidate) {
                if (worker_used[candidate] == 0u) {
                    worker_index = candidate;
                    break;
                }
            }
        }
        if (worker_index == active_worker_count) {
            lw_set_error(error, LW_STATUS_INVALID_SHAPE, "unable to assign OCR worker");
            return LW_STATUS_INVALID_SHAPE;
        }
        worker_used[worker_index] = 1u;
        {
            lw_ocr_worker_task* task = &ocr->worker_tasks[worker_index];
            memset(task, 0, sizeof(*task));
            task->ocr = ocr;
            task->worker_index = worker_index;
            task->initial_crop_slot = slot;
            task->profile_enabled = 0u;
            task->wall_nanoseconds = 0u;
            if (profile != NULL) {
                task->profile_enabled = 1u;
                lw_pipeline_component_profile_reset(&task->classifier_profile, profile->clock,
                                                    profile->clock_context);
                lw_pipeline_component_profile_reset(&task->recognizer_profile, profile->clock,
                                                    profile->clock_context);
            }
        }
    }

    /* Every active worker claims the next longest remaining crop. Keep worker
     * zero on the calling thread. If another thread cannot be created, run
     * that worker synchronously so its reserved width-affine crop is not lost. */
    for (slot = 1u; slot < active_worker_count; ++slot) {
        uint32_t worker_index = slot;
#if defined(_WIN32)
        ocr->worker_threads[worker_index] = (HANDLE)_beginthreadex(
            NULL, 0u, worker_entry, &ocr->worker_tasks[worker_index], 0u, NULL);
        if (ocr->worker_threads[worker_index] != NULL) {
            ocr->worker_started[worker_index] = 1u;
        } else {
            execute_worker_task(&ocr->worker_tasks[worker_index]);
        }
#elif !defined(__EMSCRIPTEN__)
        if (pthread_create(&ocr->worker_threads[worker_index], NULL, worker_entry,
                           &ocr->worker_tasks[worker_index]) == 0) {
            ocr->worker_started[worker_index] = 1u;
        } else {
            execute_worker_task(&ocr->worker_tasks[worker_index]);
        }
#else
        execute_worker_task(&ocr->worker_tasks[worker_index]);
#endif
    }
    execute_worker_task(&ocr->worker_tasks[0]);
    for (slot = 1u; slot < active_worker_count; ++slot) {
        uint32_t worker_index = slot;
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
    if (profile != NULL) {
        uint64_t finished = lw_ocr_profile_now(profile);
        uint64_t batch_nanoseconds = finished >= started ? finished - started : 0u;
        for (slot = 0u; slot < active_worker_count; ++slot) {
            uint32_t worker_index = slot;
            if (ocr->worker_tasks[worker_index].wall_nanoseconds > critical_nanoseconds) {
                critical_nanoseconds = ocr->worker_tasks[worker_index].wall_nanoseconds;
            }
        }
        lw_profile_add_value(&profile->line_workers_nanoseconds, batch_nanoseconds);
        lw_profile_add_value(&profile->line_worker_critical_nanoseconds, critical_nanoseconds);
        if (batch_nanoseconds > critical_nanoseconds) {
            lw_profile_add_value(&profile->line_dispatch_overhead_nanoseconds,
                                 batch_nanoseconds - critical_nanoseconds);
        }
    }
    for (slot = 0u; slot < active_worker_count; ++slot) {
        uint32_t worker_index = slot;
        if (profile != NULL) {
            uint32_t bucket;
            lw_pipeline_component_profile_accumulate(
                &profile->classifier, &ocr->worker_tasks[worker_index].classifier_profile);
            lw_pipeline_component_profile_accumulate(
                &profile->recognizer, &ocr->worker_tasks[worker_index].recognizer_profile);
            lw_profile_add_value(&profile->rec_width_sample_count,
                                 ocr->worker_tasks[worker_index].rec_width_sample_count);
            lw_profile_add_value(&profile->rec_resized_width_sum,
                                 ocr->worker_tasks[worker_index].rec_resized_width_sum);
            lw_profile_add_value(&profile->rec_target_width_sum,
                                 ocr->worker_tasks[worker_index].rec_target_width_sum);
            for (bucket = 0u; bucket < LW_REC_WIDTH_HISTOGRAM_BUCKET_COUNT; ++bucket) {
                lw_profile_add_value(&profile->rec_width_histogram[bucket],
                                     ocr->worker_tasks[worker_index].rec_width_histogram[bucket]);
            }
        }
        if (ocr->worker_tasks[worker_index].status != LW_STATUS_OK) {
            status = ocr->worker_tasks[worker_index].status;
            lw_set_error(error, status, ocr->worker_tasks[worker_index].error.message);
            break;
        }
    }
    return status;
}

static lw_status crop_and_run_adaptive(lw_ocr* ocr, const uint8_t* source,
                                       uint64_t source_byte_count, uint32_t source_width,
                                       uint32_t source_height, uint32_t source_stride,
                                       uint32_t line_count, lw_ocr_execution_profile* profile,
                                       lw_error* error) {
    uint64_t crop_used = 0u;
    uint32_t scheduled_count = 0u;
    while (scheduled_count < line_count) {
        uint32_t selected_index = UINT32_MAX;
        uint32_t selected_width = 0u;
        uint32_t index;
        lw_ocr_crop* crop;
        lw_detection_box* box;
        uint32_t crop_width;
        uint32_t crop_height;
        uint64_t crop_bytes;
        uint64_t crop_started;
        lw_status status;
        for (index = 0u; index < line_count; ++index) {
            const lw_ocr_crop* candidate = &ocr->crops[index];
            if (candidate->scheduled == 0u &&
                (candidate->target_width > selected_width ||
                 (candidate->target_width == selected_width && index < selected_index))) {
                selected_index = index;
                selected_width = candidate->target_width;
            }
        }
        if (selected_index == UINT32_MAX) {
            lw_set_error(error, LW_STATUS_INVALID_SHAPE,
                         "adaptive REC width scheduling is incomplete");
            return LW_STATUS_INVALID_SHAPE;
        }
        crop = &ocr->crops[selected_index];
        if (crop_used > UINT64_MAX - crop->byte_count) {
            lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "aggregate OCR crop size overflows");
            return LW_STATUS_OUT_OF_BOUNDS;
        }
        status = ensure_crop_capacity(ocr, crop_used + crop->byte_count, error);
        if (status != LW_STATUS_OK) {
            return status;
        }
        box = &ocr->detected_boxes[crop->box_index];
        crop_width = crop->width;
        crop_height = crop->height;
        crop_bytes = crop->byte_count;
        crop_started = lw_ocr_profile_now(profile);
        status = lw_crop_quad_bgr_u8(source, source_byte_count, source_width, source_height,
                                     source_stride, box, ocr->crop + (size_t)crop_used, crop_bytes,
                                     &crop_width, &crop_height, &crop_bytes);
        lw_ocr_profile_add_elapsed(profile == NULL ? NULL : &profile->crop_nanoseconds,
                                   crop_started, profile);
        if (status != LW_STATUS_OK) {
            lw_set_error(error, status, "OCR perspective crop failed");
            return status;
        }
        crop->offset = crop_used;
        crop->byte_count = crop_bytes;
        crop->width = crop_width;
        crop->height = crop_height;
        crop->scheduled = 1u;
        ocr->crop_schedule[scheduled_count] = selected_index;
        crop_used += crop_bytes;
        ++scheduled_count;
    }
    return run_worker_tasks(ocr, line_count, profile, error);
}

static lw_status ocr_run_bgr_u8_impl(lw_ocr* ocr, const uint8_t* source, uint64_t source_byte_count,
                                     uint32_t source_width, uint32_t source_height,
                                     uint32_t source_stride, lw_ocr_line* lines,
                                     uint32_t line_capacity, char* text_utf8,
                                     uint64_t text_capacity, lw_ocr_result* result,
                                     lw_ocr_execution_profile* profile, lw_error* error) {
    lw_detection_result detection;
    uint32_t line_count = 0u;
    uint64_t text_used = 0u;
    uint64_t total_started;
    uint64_t output_started;
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
    total_started = lw_ocr_profile_now(profile);
    capacity_query =
        lines == NULL && line_capacity == 0u && text_utf8 == NULL && text_capacity == 0u;

    /* DET writes into handle-owned scratch space so a capacity-only call still
     * executes the exact same pipeline and reports exact output requirements. */
    lw_detection_result_init(&detection);
    status = profile == NULL
                 ? lw_detector_detect_bgr_u8(ocr->detector, source, source_byte_count, source_width,
                                             source_height, source_stride, ocr->detected_boxes,
                                             ocr->info.max_line_capacity, &detection, error)
                 : lw_detector_detect_bgr_u8_profiled(
                       ocr->detector, source, source_byte_count, source_width, source_height,
                       source_stride, ocr->detected_boxes, ocr->info.max_line_capacity, &detection,
                       &profile->detector, error);
    result->detected_count = detection.box_count;
    result->detector_resized_width = detection.resized_width;
    result->detector_resized_height = detection.resized_height;
    if (status != LW_STATUS_OK)
        return status;
    /* Resolve crop geometry before running REC. The private scheduler can then
     * materialize every crop in longest-first order and dynamically balance
     * the complete request across the available line workers. */
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
        crop = &ocr->crops[line_count];
        crop->offset = 0u;
        crop->byte_count = crop_bytes;
        crop->width = crop_width;
        crop->height = crop_height;
        crop->box_index = index;
        crop->target_width =
            lw_recognizer_target_width_for_image(ocr->recognizers[0], crop_width, crop_height);
        crop->scheduled = 0u;
        ++line_count;
    }
    status = crop_and_run_adaptive(ocr, source, source_byte_count, source_width, source_height,
                                   source_stride, line_count, profile, error);
    if (status != LW_STATUS_OK) {
        return status;
    }
    /* Workers write fixed-size text slots to avoid synchronization. Compact
     * those slots in reading order before publishing the public result. */
    output_started = lw_ocr_profile_now(profile);
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
        lw_ocr_profile_add_elapsed(profile == NULL ? NULL : &profile->output_nanoseconds,
                                   output_started, profile);
        lw_ocr_profile_add_elapsed(profile == NULL ? NULL : &profile->total_nanoseconds,
                                   total_started, profile);
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
    lw_ocr_profile_add_elapsed(profile == NULL ? NULL : &profile->output_nanoseconds,
                               output_started, profile);
    lw_ocr_profile_add_elapsed(profile == NULL ? NULL : &profile->total_nanoseconds, total_started,
                               profile);
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}

lw_status lw_ocr_run_bgr_u8(lw_ocr* ocr, const uint8_t* source, uint64_t source_byte_count,
                            uint32_t source_width, uint32_t source_height, uint32_t source_stride,
                            lw_ocr_line* lines, uint32_t line_capacity, char* text_utf8,
                            uint64_t text_capacity, lw_ocr_result* result, lw_error* error) {
    return ocr_run_bgr_u8_impl(ocr, source, source_byte_count, source_width, source_height,
                               source_stride, lines, line_capacity, text_utf8, text_capacity,
                               result, NULL, error);
}

lw_status lw_ocr_run_bgr_u8_profiled(lw_ocr* ocr, const uint8_t* source, uint64_t source_byte_count,
                                     uint32_t source_width, uint32_t source_height,
                                     uint32_t source_stride, lw_ocr_line* lines,
                                     uint32_t line_capacity, char* text_utf8,
                                     uint64_t text_capacity, lw_ocr_result* result,
                                     lw_ocr_execution_profile* profile, lw_error* error) {
    if (profile == NULL || profile->struct_size != sizeof(*profile) || profile->reserved != 0u ||
        profile->clock == NULL ||
        profile->detector.execution.struct_size != sizeof(profile->detector.execution) ||
        profile->classifier.execution.struct_size != sizeof(profile->classifier.execution) ||
        profile->recognizer.execution.struct_size != sizeof(profile->recognizer.execution)) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "an initialized full OCR profile and clock are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    return ocr_run_bgr_u8_impl(ocr, source, source_byte_count, source_width, source_height,
                               source_stride, lines, line_capacity, text_utf8, text_capacity,
                               result, profile, error);
}
