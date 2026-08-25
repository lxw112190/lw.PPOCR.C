#include "det_internal.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static uint32_t clamp_coordinate(int64_t coordinate, uint32_t limit) {
    if (coordinate < 0) {
        return 0u;
    }
    if ((uint64_t)coordinate >= limit) {
        return limit - 1u;
    }
    return (uint32_t)coordinate;
}

lw_status lw_det_compute_size(
    uint32_t source_width,
    uint32_t source_height,
    uint32_t limit_side_length,
    uint32_t* resized_width,
    uint32_t* resized_height,
    float* width_ratio,
    float* height_ratio) {
    uint32_t maximum_side;
    double ratio;
    double scaled_width;
    double scaled_height;
    uint64_t rounded_width;
    uint64_t rounded_height;
    if (resized_width == NULL || resized_height == NULL || width_ratio == NULL ||
        height_ratio == NULL || source_width == 0u || source_height == 0u ||
        limit_side_length < 32u || limit_side_length > INT32_MAX) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    maximum_side = source_width > source_height ? source_width : source_height;
    ratio = maximum_side > limit_side_length ?
        (double)limit_side_length / maximum_side : 1.0;
    scaled_width = (double)source_width * ratio / 32.0;
    scaled_height = (double)source_height * ratio / 32.0;
    rounded_width = (uint64_t)floor(scaled_width + 0.5) * 32u;
    rounded_height = (uint64_t)floor(scaled_height + 0.5) * 32u;
    if (rounded_width < 32u) {
        rounded_width = 32u;
    }
    if (rounded_height < 32u) {
        rounded_height = 32u;
    }
    if (rounded_width > INT32_MAX || rounded_height > INT32_MAX) {
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    *resized_width = (uint32_t)rounded_width;
    *resized_height = (uint32_t)rounded_height;
    *width_ratio = (float)((double)rounded_width / source_width);
    *height_ratio = (float)((double)rounded_height / source_height);
    return LW_STATUS_OK;
}

lw_status lw_det_preprocess_bgr_u8(
    const uint8_t* source,
    uint64_t source_byte_count,
    uint32_t source_width,
    uint32_t source_height,
    uint32_t source_stride,
    uint32_t resized_width,
    uint32_t resized_height,
    float* output,
    uint64_t output_element_count) {
    static const double mean[3] = {0.485, 0.456, 0.406};
    static const double inverse_std[3] = {
        1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225};
    uint64_t row_bytes;
    uint64_t required_source_bytes;
    uint64_t plane;
    uint64_t required_output_elements;
    uint32_t output_y;
    if (source == NULL || output == NULL || source_width == 0u ||
        source_height == 0u || resized_width == 0u || resized_height == 0u) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    row_bytes = (uint64_t)source_width * 3u;
    if (row_bytes > UINT32_MAX || source_stride < row_bytes ||
        (uint64_t)(source_height - 1u) * source_stride > UINT64_MAX - row_bytes) {
        return LW_STATUS_INVALID_SHAPE;
    }
    required_source_bytes = (uint64_t)(source_height - 1u) * source_stride + row_bytes;
    plane = (uint64_t)resized_width * resized_height;
    if (plane > UINT64_MAX / 3u) {
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    required_output_elements = plane * 3u;
    if (required_source_bytes > SIZE_MAX || source_byte_count < required_source_bytes ||
        output_element_count != required_output_elements ||
        required_output_elements > SIZE_MAX / sizeof(float)) {
        return LW_STATUS_INVALID_SHAPE;
    }
    for (output_y = 0u; output_y < resized_height; ++output_y) {
        double source_y = ((double)output_y + 0.5) * source_height /
            resized_height - 0.5;
        int64_t source_y0_raw = (int64_t)floor(source_y);
        int64_t source_y1_raw = source_y0_raw + 1;
        uint32_t source_y0 = clamp_coordinate(source_y0_raw, source_height);
        uint32_t source_y1 = clamp_coordinate(source_y1_raw, source_height);
        double weight_y = source_y - source_y0_raw;
        uint32_t output_x;
        for (output_x = 0u; output_x < resized_width; ++output_x) {
            double source_x = ((double)output_x + 0.5) * source_width /
                resized_width - 0.5;
            int64_t source_x0_raw = (int64_t)floor(source_x);
            int64_t source_x1_raw = source_x0_raw + 1;
            uint32_t source_x0 = clamp_coordinate(source_x0_raw, source_width);
            uint32_t source_x1 = clamp_coordinate(source_x1_raw, source_width);
            double weight_x = source_x - source_x0_raw;
            uint32_t channel;
            for (channel = 0u; channel < 3u; ++channel) {
                double top_left = source[(size_t)((uint64_t)source_y0 * source_stride +
                    (uint64_t)source_x0 * 3u + channel)];
                double top_right = source[(size_t)((uint64_t)source_y0 * source_stride +
                    (uint64_t)source_x1 * 3u + channel)];
                double bottom_left = source[(size_t)((uint64_t)source_y1 * source_stride +
                    (uint64_t)source_x0 * 3u + channel)];
                double bottom_right = source[(size_t)((uint64_t)source_y1 * source_stride +
                    (uint64_t)source_x1 * 3u + channel)];
                double top = top_left + (top_right - top_left) * weight_x;
                double bottom = bottom_left + (bottom_right - bottom_left) * weight_x;
                double value = (top + (bottom - top) * weight_y) / 255.0;
                output[(size_t)((uint64_t)channel * plane +
                    (uint64_t)output_y * resized_width + output_x)] =
                    (float)((value - mean[channel]) * inverse_std[channel]);
            }
        }
    }
    return LW_STATUS_OK;
}
