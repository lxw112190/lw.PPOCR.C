#ifndef LW_DET_INTERNAL_H
#define LW_DET_INTERNAL_H

/* Private boundary between DET preprocessing, inference and postprocessing. */

#include "lw_infer.h"
#include "reading_order.h"

#include <stdint.h>

typedef struct lw_pipeline_component_profile lw_pipeline_component_profile;

lw_status lw_det_compute_size(uint32_t source_width, uint32_t source_height,
                              uint32_t limit_side_length, uint32_t* resized_width,
                              uint32_t* resized_height, float* width_ratio, float* height_ratio);

lw_status lw_det_preprocess_bgr_u8(const uint8_t* source, uint64_t source_byte_count,
                                   uint32_t source_width, uint32_t source_height,
                                   uint32_t source_stride, uint32_t resized_width,
                                   uint32_t resized_height, float* output,
                                   uint64_t output_element_count);

/* OpenCV-parity fixed-point resize + LUT normalization, with optional row
 * sharding over the detector's intra-op pool. */
typedef struct lw_det_preprocess_workspace {
    int32_t* x_offsets;
    int16_t* x_coefficients;
    int32_t* rows; /* (workers + 1) x 2 x width x 3 row scratch */
    uint32_t width;
    uint32_t row_slot_count;
} lw_det_preprocess_workspace;
typedef struct lw_thread_pool lw_thread_pool;

lw_status lw_det_preprocess_bgr_u8_fixed(
    const uint8_t* source, uint64_t source_byte_count, uint32_t source_width,
    uint32_t source_height, uint32_t source_stride, uint32_t resized_width,
    uint32_t resized_height, float* output, uint64_t output_element_count,
    lw_det_preprocess_workspace* workspace, lw_thread_pool* pool,
    uint32_t intra_op_thread_count);
lw_status lw_det_preprocess_bgr_u8_fixed_nhwc(
    const uint8_t* source, uint64_t source_byte_count, uint32_t source_width,
    uint32_t source_height, uint32_t source_stride, uint32_t resized_width,
    uint32_t resized_height, float* output, uint64_t output_element_count,
    lw_det_preprocess_workspace* workspace, lw_thread_pool* pool,
    uint32_t intra_op_thread_count);

void lw_det_preprocess_workspace_init(lw_det_preprocess_workspace* workspace);
void lw_det_preprocess_workspace_free(lw_det_preprocess_workspace* workspace);

lw_status lw_db_postprocess_f32(const float* prediction, uint32_t map_width, uint32_t map_height,
                                float bitmap_threshold, float box_threshold, float unclip_ratio,
                                uint32_t use_dilation, uint32_t max_candidates,
                                uint32_t source_width, uint32_t source_height, float width_ratio,
                                float height_ratio, lw_detection_box* boxes, uint32_t box_capacity,
                                uint32_t* box_count);

/* Reusable grow-only postprocess scratch owned by the detector handle; kills
 * the per-image malloc/free churn of the standalone entry. */
typedef struct lw_db_postprocess_workspace {
    uint8_t* bitmap;
    uint8_t* visited;
    uint8_t* background_visited;
    uint8_t* dilated;
    uint32_t* queue;
    void* contour_rows;
    void* points;
    void* hull;
    void* results;
    void* unclip_pts;
    void* unclip_normals;
    void* unclip_result;
    void* score_thresholds;
    void* score_hit_low;
    void* score_hit_high;
    void* score_poly_x;
    void* score_poly_y;
    uint32_t contour_row_capacity;
    uint32_t hull_capacity;
    uint32_t pixel_capacity;
    uint32_t result_capacity;
    uint32_t unclip_capacity;
    uint32_t score_capacity;
} lw_db_postprocess_workspace;

lw_status lw_db_postprocess_f32_ws(
    const float* prediction, uint32_t map_width, uint32_t map_height,
    float bitmap_threshold, float box_threshold, float unclip_ratio,
    uint32_t use_dilation, uint32_t max_candidates, uint32_t source_width,
    uint32_t source_height, float width_ratio, float height_ratio,
    lw_detection_box* boxes, uint32_t box_capacity, uint32_t* box_count,
    lw_db_postprocess_workspace* workspace, lw_pipeline_component_profile* profile);

void lw_db_postprocess_workspace_free(lw_db_postprocess_workspace* workspace);

/* Full OCR gives DET an independent phase-local intra-op budget. */
void lw_detector_set_intra_op_thread_count(lw_detector* detector, uint32_t thread_count);
uint32_t lw_detector_get_intra_op_thread_count(const lw_detector* detector);

#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
/* Test-only internal hooks: opt the detector into the compiled x64 DET
 * backend (the production default stays canonical until the promotion gate
 * flips it). */
void lw_detector_test_enable_x64_backend(lw_detector* detector);
void lw_detector_test_disable_x64_backend(lw_detector* detector);
#endif

#endif
