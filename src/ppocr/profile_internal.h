#ifndef LW_PPOCR_PROFILE_INTERNAL_H
#define LW_PPOCR_PROFILE_INTERNAL_H

#include "executor_internal.h"
#include "lw_infer.h"

#include <stdint.h>

/*
 * Private profiling contract used by benchmarks and regression tests. It is
 * deliberately kept out of the public C ABI until the measurement fields and
 * their wall-time/work-time semantics are proven stable.
 */
typedef struct lw_pipeline_component_profile {
    uint64_t preprocess_nanoseconds;
    uint64_t graph_nanoseconds;
    uint64_t postprocess_nanoseconds;
    lw_execution_profile execution;
} lw_pipeline_component_profile;

typedef struct lw_ocr_execution_profile {
    uint32_t struct_size;
    uint32_t reserved;
    lw_execution_profile_clock clock;
    void* clock_context;
    uint64_t total_nanoseconds;
    uint64_t crop_nanoseconds;
    uint64_t line_workers_nanoseconds;
    uint64_t line_worker_critical_nanoseconds;
    uint64_t line_dispatch_overhead_nanoseconds;
    uint64_t output_nanoseconds;
    lw_pipeline_component_profile detector;
    lw_pipeline_component_profile classifier;
    lw_pipeline_component_profile recognizer;
} lw_ocr_execution_profile;

void lw_ocr_execution_profile_init(lw_ocr_execution_profile* profile,
                                   lw_execution_profile_clock clock, void* clock_context);
void lw_pipeline_component_profile_reset(lw_pipeline_component_profile* profile,
                                         lw_execution_profile_clock clock, void* clock_context);
void lw_pipeline_component_profile_accumulate(lw_pipeline_component_profile* destination,
                                              const lw_pipeline_component_profile* source);
uint64_t lw_pipeline_profile_now(const lw_pipeline_component_profile* profile);
void lw_pipeline_profile_add_elapsed(uint64_t* destination, uint64_t started,
                                     const lw_pipeline_component_profile* profile);
uint64_t lw_ocr_profile_now(const lw_ocr_execution_profile* profile);
void lw_ocr_profile_add_elapsed(uint64_t* destination, uint64_t started,
                                const lw_ocr_execution_profile* profile);
void lw_profile_add_value(uint64_t* destination, uint64_t value);

lw_status lw_detector_detect_bgr_u8_profiled(
    lw_detector* detector, const uint8_t* source, uint64_t source_byte_count, uint32_t source_width,
    uint32_t source_height, uint32_t source_stride, lw_detection_box* boxes, uint32_t box_capacity,
    lw_detection_result* result, lw_pipeline_component_profile* profile, lw_error* error);

lw_status lw_classifier_classify_bgr_u8_profiled(lw_classifier* classifier, const uint8_t* source,
                                                 uint64_t source_byte_count, uint32_t source_width,
                                                 uint32_t source_height, uint32_t source_stride,
                                                 lw_classification_result* result,
                                                 lw_pipeline_component_profile* profile,
                                                 lw_error* error);

lw_status lw_recognizer_recognize_bgr_u8_profiled(lw_recognizer* recognizer, const uint8_t* source,
                                                  uint64_t source_byte_count, uint32_t source_width,
                                                  uint32_t source_height, uint32_t source_stride,
                                                  char* text_utf8, uint64_t text_capacity,
                                                  lw_recognition_result* result,
                                                  lw_pipeline_component_profile* profile,
                                                  lw_error* error);

lw_status lw_ocr_run_bgr_u8_profiled(lw_ocr* ocr, const uint8_t* source, uint64_t source_byte_count,
                                     uint32_t source_width, uint32_t source_height,
                                     uint32_t source_stride, lw_ocr_line* lines,
                                     uint32_t line_capacity, char* text_utf8,
                                     uint64_t text_capacity, lw_ocr_result* result,
                                     lw_ocr_execution_profile* profile, lw_error* error);

#endif
