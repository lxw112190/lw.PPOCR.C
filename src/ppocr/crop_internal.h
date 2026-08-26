#ifndef LW_CROP_INTERNAL_H
#define LW_CROP_INTERNAL_H

#include "lw_infer.h"

#include <stdint.h>

lw_status lw_crop_quad_size(
    const lw_detection_box* box,
    uint32_t* crop_width,
    uint32_t* crop_height,
    uint64_t* crop_byte_count);

lw_status lw_crop_quad_bgr_u8(
    const uint8_t* source,
    uint64_t source_byte_count,
    uint32_t source_width,
    uint32_t source_height,
    uint32_t source_stride,
    const lw_detection_box* box,
    uint8_t* crop,
    uint64_t crop_capacity,
    uint32_t* crop_width,
    uint32_t* crop_height,
    uint64_t* crop_byte_count);

void lw_rotate_bgr_u8_180(
    uint8_t* pixels,
    uint32_t width,
    uint32_t height);

#endif
