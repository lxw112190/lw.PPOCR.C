#include "lw_infer.h"

/*
 * Stable WebAssembly adapter for JavaScript.
 *
 * The native C ABI is still evolving before 1.0, so JavaScript must not read
 * lw_ocr_info, lw_ocr_line, or lw_ocr_result directly from WASM memory. This
 * adapter translates them into a small Web ABI made only of 32-bit fields.
 */

#if defined(__EMSCRIPTEN__)
#  include <emscripten/emscripten.h>
#  define LW_WEB_API EMSCRIPTEN_KEEPALIVE
#else
#  define LW_WEB_API
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(LW_NODE_WASM_PROFILE)
#  include "../src/ppocr/profile_internal.h"
#  include <stdio.h>
#endif

#ifndef LW_WEB_ABI_VERSION
#  define LW_WEB_ABI_VERSION 1u
#endif
/* Article screenshots and mobile captures often contain one text line much
 * wider than the native 320-pixel REC preset. The converted REC graph accepts
 * dynamic widths, so the offline page keeps up to 960 pixels of horizontal
 * detail instead of shrinking those lines to 320. Short lines are still
 * aspect-ratio-preserved and padded on the right by the shared preprocessor. */
#define LW_WEB_REC_TARGET_WIDTH 960u

static const char* web_model_path(const char* onnx, const char* lwm) {
    FILE* file = fopen(onnx, "rb");
    if (file == NULL) return lwm;
    fclose(file);
    return onnx;
}

#if defined(LW_NODE_WASM_PROFILE)
static uint64_t web_profile_now(void* context) {
    double milliseconds;
    (void)context;
    milliseconds = emscripten_get_now();
    return milliseconds > 0.0 ? (uint64_t)(milliseconds * 1000000.0) : 0u;
}

static void web_print_ocr_profile(const lw_ocr_execution_profile* profile) {
    const double to_ms = 1.0 / 1000000.0;
    const lw_pipeline_component_profile* det = &profile->detector;
    const lw_pipeline_component_profile* cls = &profile->classifier;
    const lw_pipeline_component_profile* rec = &profile->recognizer;
    (void)fprintf(stderr,
        "LW_WASM_OCR_PROFILE total=%.3f det_preprocess=%.3f det_graph=%.3f "
        "det_postprocess=%.3f crop=%.3f cls=%.3f rec=%.3f "
        "det_compiled=%llu det_fallback=%llu cls_compiled=%llu cls_fallback=%llu "
        "rec_compiled=%llu rec_fallback=%llu\n",
        profile->total_nanoseconds * to_ms,
        det->preprocess_nanoseconds * to_ms,
        det->graph_nanoseconds * to_ms,
        det->postprocess_nanoseconds * to_ms,
        profile->crop_nanoseconds * to_ms,
        (cls->preprocess_nanoseconds + cls->graph_nanoseconds +
         cls->postprocess_nanoseconds) * to_ms,
        (rec->preprocess_nanoseconds + rec->graph_nanoseconds +
         rec->postprocess_nanoseconds) * to_ms,
        (unsigned long long)det->compiled_backend_runs,
        (unsigned long long)det->canonical_fallback_runs,
        (unsigned long long)cls->compiled_backend_runs,
        (unsigned long long)cls->canonical_fallback_runs,
        (unsigned long long)rec->compiled_backend_lines,
        (unsigned long long)rec->canonical_fallback_lines);
    (void)fprintf(stderr,
        "LW_WASM_REC_WIDTH_PROFILE samples=%llu resized_sum=%llu target_sum=%llu padding_ratio=%.6f\n",
        (unsigned long long)profile->rec_width_sample_count,
        (unsigned long long)profile->rec_resized_width_sum,
        (unsigned long long)profile->rec_target_width_sum,
        profile->rec_target_width_sum == 0u ? 0.0 :
            1.0 - (double)profile->rec_resized_width_sum / (double)profile->rec_target_width_sum);
}
#endif

typedef struct lw_web_info {
    uint32_t abi_version;
    uint32_t max_line_capacity;
    uint32_t max_text_capacity;
    uint32_t line_size;
    uint32_t result_size;
} lw_web_info;

typedef struct lw_web_line {
    float x1;
    float y1;
    float x2;
    float y2;
    float x3;
    float y3;
    float x4;
    float y4;
    float detection_score;
    float recognition_score;
    float classification_score;
    uint32_t classification_label;
    uint32_t applied_rotation_degrees;
    uint32_t text_offset;
    uint32_t text_length;
} lw_web_line;

typedef struct lw_web_result {
    uint32_t line_count;
    uint32_t detected_count;
    uint32_t detector_resized_width;
    uint32_t detector_resized_height;
} lw_web_result;

/* Freeze the byte layout consumed by the offline page. Any future incompatible
 * layout needs a new LW_WEB_ABI_VERSION instead of silently changing offsets. */
_Static_assert(sizeof(lw_web_info) == 20u, "unexpected lw_web_info layout");
_Static_assert(offsetof(lw_web_info, abi_version) == 0u,
               "unexpected lw_web_info abi_version offset");
_Static_assert(offsetof(lw_web_info, max_line_capacity) == 4u,
               "unexpected lw_web_info max_line_capacity offset");
_Static_assert(offsetof(lw_web_info, max_text_capacity) == 8u,
               "unexpected lw_web_info max_text_capacity offset");
_Static_assert(offsetof(lw_web_info, line_size) == 12u,
               "unexpected lw_web_info line_size offset");
_Static_assert(offsetof(lw_web_info, result_size) == 16u,
               "unexpected lw_web_info result_size offset");
_Static_assert(sizeof(lw_web_line) == 60u, "unexpected lw_web_line layout");
_Static_assert(offsetof(lw_web_line, x1) == 0u,
               "unexpected lw_web_line x1 offset");
_Static_assert(offsetof(lw_web_line, y1) == 4u,
               "unexpected lw_web_line y1 offset");
_Static_assert(offsetof(lw_web_line, x2) == 8u,
               "unexpected lw_web_line x2 offset");
_Static_assert(offsetof(lw_web_line, y2) == 12u,
               "unexpected lw_web_line y2 offset");
_Static_assert(offsetof(lw_web_line, x3) == 16u,
               "unexpected lw_web_line x3 offset");
_Static_assert(offsetof(lw_web_line, y3) == 20u,
               "unexpected lw_web_line y3 offset");
_Static_assert(offsetof(lw_web_line, x4) == 24u,
               "unexpected lw_web_line x4 offset");
_Static_assert(offsetof(lw_web_line, y4) == 28u,
               "unexpected lw_web_line y4 offset");
_Static_assert(sizeof(lw_web_result) == 16u, "unexpected lw_web_result layout");
_Static_assert(offsetof(lw_web_result, line_count) == 0u,
               "unexpected lw_web_result line_count offset");
_Static_assert(offsetof(lw_web_result, detected_count) == 4u,
               "unexpected lw_web_result detected_count offset");
_Static_assert(offsetof(lw_web_result, detector_resized_width) == 8u,
               "unexpected lw_web_result detector_resized_width offset");
_Static_assert(offsetof(lw_web_result, detector_resized_height) == 12u,
               "unexpected lw_web_result detector_resized_height offset");
_Static_assert(offsetof(lw_web_line, detection_score) == 32u,
               "unexpected lw_web_line detection_score offset");
_Static_assert(offsetof(lw_web_line, recognition_score) == 36u,
               "unexpected lw_web_line recognition_score offset");
_Static_assert(offsetof(lw_web_line, classification_score) == 40u,
               "unexpected lw_web_line classification_score offset");
_Static_assert(offsetof(lw_web_line, classification_label) == 44u,
               "unexpected lw_web_line classification_label offset");
_Static_assert(offsetof(lw_web_line, applied_rotation_degrees) == 48u,
               "unexpected lw_web_line applied_rotation_degrees offset");
_Static_assert(offsetof(lw_web_line, text_offset) == 52u,
               "unexpected lw_web_line text_offset offset");
_Static_assert(offsetof(lw_web_line, text_length) == 56u,
               "unexpected lw_web_line text_length offset");

static lw_ocr* g_ocr;
static lw_ocr_info g_ocr_info;
static lw_ocr_line* g_native_lines;
static lw_error g_error;
static uint32_t g_reading_order = LW_READING_ORDER_HORIZONTAL_LTR;

static int allocation_fits(uint64_t count, size_t element_size) {
    return element_size != 0u && count <= (uint64_t)(SIZE_MAX / element_size);
}

static int web_fail(lw_status status, const char* message) {
    size_t length;
    lw_error_init(&g_error);
    g_error.code = (int32_t)status;
    if (message == NULL) {
        return (int)status;
    }
    length = strlen(message);
    if (length >= sizeof(g_error.message)) {
        length = sizeof(g_error.message) - 1u;
    }
    memcpy(g_error.message, message, length);
    g_error.message[length] = '\0';
    return (int)status;
}

static void release_ocr(void) {
    free(g_native_lines);
    g_native_lines = NULL;
    lw_ocr_free(g_ocr);
    g_ocr = NULL;
    memset(&g_ocr_info, 0, sizeof(g_ocr_info));
    g_reading_order = LW_READING_ORDER_HORIZONTAL_LTR;
}

LW_WEB_API int lw_web_init(int use_classifier) {
    lw_ocr_options options;
    lw_status status;

    release_ocr();
    lw_error_init(&g_error);
    lw_ocr_options_init(&options);
    options.use_direction_classification = use_classifier ? 1u : 0u;
    options.recognizer.target_width = LW_WEB_REC_TARGET_WIDTH;
    status = lw_ocr_create(web_model_path("/models/det.onnx", "/models/det.lwm"),
                           use_classifier ? web_model_path("/models/cls.onnx", "/models/cls.lwm") : NULL,
                           web_model_path("/models/rec.onnx", "/models/rec.lwm"),
                           "/models/ppocr_keys.txt", &options, &g_ocr,
                           &g_error);
    if (status != LW_STATUS_OK) {
        return (int)status;
    }
    status = lw_ocr_set_reading_order(g_ocr, g_reading_order, &g_error);
    if (status != LW_STATUS_OK) {
        release_ocr();
        return (int)status;
    }

    lw_ocr_info_init(&g_ocr_info);
    status = lw_ocr_get_info(g_ocr, &g_ocr_info);
    if (status != LW_STATUS_OK || g_ocr_info.max_line_capacity == 0u ||
        g_ocr_info.max_text_capacity == 0u) {
        release_ocr();
        return web_fail(status == LW_STATUS_OK ? LW_STATUS_INVALID_SHAPE : status,
                        "unable to query OCR output capacities");
    }
    if (!allocation_fits(g_ocr_info.max_line_capacity, sizeof(*g_native_lines)) ||
        g_ocr_info.max_text_capacity > UINT32_MAX) {
        release_ocr();
        return web_fail(LW_STATUS_OUT_OF_BOUNDS,
                        "OCR output capacities exceed the WASM32 address space");
    }

    g_native_lines =
        (lw_ocr_line*)calloc(g_ocr_info.max_line_capacity, sizeof(*g_native_lines));
    if (g_native_lines == NULL) {
        release_ocr();
        return web_fail(LW_STATUS_OUT_OF_MEMORY, "unable to allocate native OCR line buffer");
    }
    return (int)LW_STATUS_OK;
}

LW_WEB_API void lw_web_shutdown(void) {
    release_ocr();
}

LW_WEB_API uint32_t lw_web_info_size(void) {
    return (uint32_t)sizeof(lw_web_info);
}

LW_WEB_API int lw_web_get_info(lw_web_info* info) {
    if (info == NULL || g_ocr == NULL || g_native_lines == NULL) {
        return web_fail(LW_STATUS_INVALID_ARGUMENT, "OCR engine is not initialized");
    }
    memset(info, 0, sizeof(*info));
    info->abi_version = LW_WEB_ABI_VERSION;
    info->max_line_capacity = g_ocr_info.max_line_capacity;
    info->max_text_capacity = (uint32_t)g_ocr_info.max_text_capacity;
    info->line_size = (uint32_t)sizeof(lw_web_line);
    info->result_size = (uint32_t)sizeof(lw_web_result);
    return (int)LW_STATUS_OK;
}

LW_WEB_API int lw_web_set_reading_order(uint32_t reading_order) {
    lw_status status;
    if (reading_order > (uint32_t)LW_READING_ORDER_VERTICAL_LTR)
        return web_fail(LW_STATUS_INVALID_ARGUMENT, "invalid reading order");
    g_reading_order = reading_order;
    if (g_ocr == NULL)
        return (int)LW_STATUS_OK;
    status = lw_ocr_set_reading_order(g_ocr, reading_order, &g_error);
    return (int)status;
}

LW_WEB_API int lw_web_run(const uint8_t* source, uint32_t source_byte_count,
                          uint32_t source_width, uint32_t source_height,
                          uint32_t source_stride, lw_web_line* lines,
                          uint32_t line_capacity, char* text, uint32_t text_capacity,
                          lw_web_result* result) {
    lw_ocr_result native_result;
    lw_status status;
    uint32_t index;
#if defined(LW_NODE_WASM_PROFILE)
    lw_ocr_execution_profile* profile = NULL;
#endif

    if (g_ocr == NULL || g_native_lines == NULL || lines == NULL || text == NULL ||
        result == NULL || line_capacity == 0u ||
        line_capacity > g_ocr_info.max_line_capacity || text_capacity == 0u) {
        return web_fail(LW_STATUS_INVALID_ARGUMENT, "invalid WASM OCR arguments");
    }

    memset(result, 0, sizeof(*result));
    lw_error_init(&g_error);
    lw_ocr_result_init(&native_result);
#if defined(LW_NODE_WASM_PROFILE)
    if (lw_profile_env_present("LW_WASM_OCR_PROFILE")) {
        profile = (lw_ocr_execution_profile*)malloc(sizeof(*profile));
        if (profile == NULL)
            return web_fail(LW_STATUS_OUT_OF_MEMORY, "unable to allocate OCR profile");
        lw_ocr_execution_profile_init(profile, web_profile_now, NULL);
        status = lw_ocr_run_bgr_u8_profiled(
            g_ocr, source, source_byte_count, source_width, source_height,
            source_stride, g_native_lines, line_capacity, text, text_capacity,
            &native_result, profile, &g_error);
        if (status == LW_STATUS_OK) web_print_ocr_profile(profile);
        free(profile);
    } else
#endif
    status = lw_ocr_run_bgr_u8(g_ocr, source, source_byte_count, source_width, source_height,
                               source_stride, g_native_lines, line_capacity, text, text_capacity,
                               &native_result, &g_error);
    if (status != LW_STATUS_OK) {
        return (int)status;
    }

    for (index = 0u; index < native_result.line_count; ++index) {
        const lw_ocr_line* native = &g_native_lines[index];
        lw_web_line* web = &lines[index];
        if (native->text_offset > UINT32_MAX || native->text_length > UINT32_MAX) {
            return web_fail(LW_STATUS_OUT_OF_BOUNDS,
                            "OCR text range exceeds the Web ABI capacity");
        }
        web->x1 = native->box.x1;
        web->y1 = native->box.y1;
        web->x2 = native->box.x2;
        web->y2 = native->box.y2;
        web->x3 = native->box.x3;
        web->y3 = native->box.y3;
        web->x4 = native->box.x4;
        web->y4 = native->box.y4;
        web->detection_score = native->box.score;
        web->recognition_score = native->recognition_score;
        web->classification_score = native->classification_score;
        web->classification_label = native->classification_label;
        web->applied_rotation_degrees = native->applied_rotation_degrees;
        web->text_offset = (uint32_t)native->text_offset;
        web->text_length = (uint32_t)native->text_length;
    }

    result->line_count = native_result.line_count;
    result->detected_count = native_result.detected_count;
    result->detector_resized_width = native_result.detector_resized_width;
    result->detector_resized_height = native_result.detector_resized_height;
    return (int)LW_STATUS_OK;
}

LW_WEB_API const lw_error* lw_web_last_error(void) {
    return &g_error;
}

LW_WEB_API void* lw_web_malloc(size_t size) {
    return malloc(size);
}

LW_WEB_API void lw_web_free(void* pointer) {
    free(pointer);
}
