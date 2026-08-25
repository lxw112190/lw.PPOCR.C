#include "rec_internal.h"

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

lw_status lw_rec_preprocess_bgr_u8(
    const uint8_t* source,
    uint64_t source_byte_count,
    uint32_t source_width,
    uint32_t source_height,
    uint32_t source_stride,
    uint32_t target_width,
    float* output,
    uint64_t output_element_count,
    uint32_t* resized_width) {
    uint64_t row_bytes;
    uint64_t required_source_bytes;
    uint64_t required_output_elements;
    uint64_t scaled_width_numerator;
    uint64_t computed_width;
    uint32_t actual_width;
    uint64_t channel_plane;
    uint32_t channel;
    uint32_t output_y;
    const double normalize_scale = 2.0 / 255.0;

    if (resized_width != NULL) {
        *resized_width = 0u;
    }
    if (source == NULL || output == NULL || source_width == 0u ||
        source_height == 0u || target_width == 0u) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    row_bytes = (uint64_t)source_width * 3u;
    if (row_bytes > UINT32_MAX || source_stride < row_bytes ||
        (uint64_t)(source_height - 1u) * source_stride > UINT64_MAX - row_bytes) {
        return LW_STATUS_INVALID_SHAPE;
    }
    required_source_bytes = (uint64_t)(source_height - 1u) * source_stride + row_bytes;
    required_output_elements = (uint64_t)3u * LW_REC_INPUT_HEIGHT * target_width;
    if (source_byte_count < required_source_bytes ||
        output_element_count != required_output_elements ||
        required_output_elements > SIZE_MAX / sizeof(float)) {
        return LW_STATUS_INVALID_SHAPE;
    }
    if ((uint64_t)source_width > (UINT64_MAX - (source_height - 1u)) / LW_REC_INPUT_HEIGHT) {
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    scaled_width_numerator = (uint64_t)LW_REC_INPUT_HEIGHT * source_width;
    computed_width = (scaled_width_numerator + source_height - 1u) / source_height;
    actual_width = computed_width > target_width ? target_width : (uint32_t)computed_width;
    channel_plane = (uint64_t)LW_REC_INPUT_HEIGHT * target_width;
    for (channel = 0u; channel < 3u; ++channel) {
        uint64_t index;
        float padding = (float)(128.0 * normalize_scale - 1.0);
        for (index = 0u; index < channel_plane; ++index) {
            output[(size_t)((uint64_t)channel * channel_plane + index)] = padding;
        }
    }

    for (output_y = 0u; output_y < LW_REC_INPUT_HEIGHT; ++output_y) {
        double source_y = ((double)output_y + 0.5) * source_height /
            LW_REC_INPUT_HEIGHT - 0.5;
        int64_t source_y0_raw = (int64_t)floor(source_y);
        int64_t source_y1_raw = source_y0_raw + 1;
        uint32_t source_y0 = clamp_coordinate(source_y0_raw, source_height);
        uint32_t source_y1 = clamp_coordinate(source_y1_raw, source_height);
        double weight_y = source_y - (double)source_y0_raw;
        uint32_t output_x;
        for (output_x = 0u; output_x < actual_width; ++output_x) {
            double source_x = ((double)output_x + 0.5) * source_width /
                actual_width - 0.5;
            int64_t source_x0_raw = (int64_t)floor(source_x);
            int64_t source_x1_raw = source_x0_raw + 1;
            uint32_t source_x0 = clamp_coordinate(source_x0_raw, source_width);
            uint32_t source_x1 = clamp_coordinate(source_x1_raw, source_width);
            double weight_x = source_x - (double)source_x0_raw;
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
                double value = top + (bottom - top) * weight_y;
                uint64_t output_index = (uint64_t)channel * channel_plane +
                    (uint64_t)output_y * target_width + output_x;
                output[(size_t)output_index] = (float)(value * normalize_scale - 1.0);
            }
        }
    }
    if (resized_width != NULL) {
        *resized_width = actual_width;
    }
    return LW_STATUS_OK;
}
