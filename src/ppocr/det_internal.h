#ifndef LW_DET_INTERNAL_H
#define LW_DET_INTERNAL_H

#include "lw_infer.h"

#include <stdint.h>

lw_status lw_det_compute_size(
    uint32_t source_width,
    uint32_t source_height,
    uint32_t limit_side_length,
    uint32_t* resized_width,
    uint32_t* resized_height,
    float* width_ratio,
    float* height_ratio);

lw_status lw_det_preprocess_bgr_u8(
    const uint8_t* source,
    uint64_t source_byte_count,
    uint32_t source_width,
    uint32_t source_height,
    uint32_t source_stride,
    uint32_t resized_width,
    uint32_t resized_height,
    float* output,
    uint64_t output_element_count);

lw_status lw_db_postprocess_f32(
    const float* prediction,
    uint32_t map_width,
    uint32_t map_height,
    float bitmap_threshold,
    float box_threshold,
    float unclip_ratio,
    uint32_t use_dilation,
    uint32_t max_candidates,
    uint32_t source_width,
    uint32_t source_height,
    float width_ratio,
    float height_ratio,
    lw_detection_box* boxes,
    uint32_t box_capacity,
    uint32_t* box_count);

#endif
