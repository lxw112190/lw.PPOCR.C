#include "crop_internal.h"

/* Perspective extraction and optional 180-degree rotation of detected lines. */

#include <math.h>
#include <stddef.h>
#include <stdint.h>

typedef struct crop_point {
    double x;
    double y;
} crop_point;

static int load_points(const lw_detection_box* box, crop_point points[4]) {
    const float values[8] = {box->x1, box->y1, box->x2, box->y2,
                             box->x3, box->y3, box->x4, box->y4};
    uint32_t index;
    for (index = 0u; index < 4u; ++index) {
        if (!isfinite(values[index * 2u]) || !isfinite(values[index * 2u + 1u]))
            return 0;
        points[index].x = values[index * 2u];
        points[index].y = values[index * 2u + 1u];
    }
    return 1;
}

static double point_distance(crop_point left, crop_point right) {
    return hypot(right.x - left.x, right.y - left.y);
}

lw_status lw_crop_quad_size(const lw_detection_box* box, uint32_t* crop_width,
                            uint32_t* crop_height, uint64_t* crop_byte_count) {
    crop_point points[4];
    double width_value;
    double height_value;
    uint64_t width;
    uint64_t height;
    int rotate_vertical;
    if (crop_width != NULL)
        *crop_width = 0u;
    if (crop_height != NULL)
        *crop_height = 0u;
    if (crop_byte_count != NULL)
        *crop_byte_count = 0u;
    if (box == NULL || crop_width == NULL || crop_height == NULL || crop_byte_count == NULL ||
        !load_points(box, points))
        return LW_STATUS_INVALID_ARGUMENT;
    width_value = point_distance(points[0], points[1]);
    height_value = point_distance(points[0], points[3]);
    if (!isfinite(width_value) || !isfinite(height_value) || width_value < 1.0 ||
        height_value < 1.0 || width_value > UINT32_MAX || height_value > UINT32_MAX)
        return LW_STATUS_INVALID_SHAPE;
    width = (uint64_t)floor(width_value + 0.5);
    height = (uint64_t)floor(height_value + 0.5);
    if (width == 0u)
        width = 1u;
    if (height == 0u)
        height = 1u;
    rotate_vertical = (double)height >= (double)width * 1.5;
    if (width > UINT64_MAX / height || width * height > UINT64_MAX / 3u)
        return LW_STATUS_OUT_OF_BOUNDS;
    *crop_byte_count = width * height * 3u;
    *crop_width = (uint32_t)(rotate_vertical ? height : width);
    *crop_height = (uint32_t)(rotate_vertical ? width : height);
    return LW_STATUS_OK;
}

static uint32_t clamp_coordinate(int64_t value, uint32_t limit) {
    if (value < 0)
        return 0u;
    if ((uint64_t)value >= limit)
        return limit - 1u;
    return (uint32_t)value;
}

static uint8_t sample_channel(const uint8_t* source, uint32_t source_width, uint32_t source_height,
                              uint32_t source_stride, double x, double y, uint32_t channel) {
    int64_t x0_raw = (int64_t)floor(x);
    int64_t y0_raw = (int64_t)floor(y);
    uint32_t x0 = clamp_coordinate(x0_raw, source_width);
    uint32_t x1 = clamp_coordinate(x0_raw + 1, source_width);
    uint32_t y0 = clamp_coordinate(y0_raw, source_height);
    uint32_t y1 = clamp_coordinate(y0_raw + 1, source_height);
    double wx = x - x0_raw;
    double wy = y - y0_raw;
    double top_left = source[(size_t)((uint64_t)y0 * source_stride + (uint64_t)x0 * 3u + channel)];
    double top_right = source[(size_t)((uint64_t)y0 * source_stride + (uint64_t)x1 * 3u + channel)];
    double bottom_left =
        source[(size_t)((uint64_t)y1 * source_stride + (uint64_t)x0 * 3u + channel)];
    double bottom_right =
        source[(size_t)((uint64_t)y1 * source_stride + (uint64_t)x1 * 3u + channel)];
    double top = top_left + (top_right - top_left) * wx;
    double bottom = bottom_left + (bottom_right - bottom_left) * wx;
    double value = top + (bottom - top) * wy;
    if (value <= 0.0)
        return 0u;
    if (value >= 255.0)
        return 255u;
    return (uint8_t)floor(value + 0.5);
}

lw_status lw_crop_quad_bgr_u8(const uint8_t* source, uint64_t source_byte_count,
                              uint32_t source_width, uint32_t source_height, uint32_t source_stride,
                              const lw_detection_box* box, uint8_t* crop, uint64_t crop_capacity,
                              uint32_t* crop_width, uint32_t* crop_height,
                              uint64_t* crop_byte_count) {
    crop_point points[4];
    uint32_t output_width;
    uint32_t output_height;
    uint64_t required_bytes;
    uint32_t unrotated_width;
    uint32_t unrotated_height;
    int rotate_vertical;
    double dx1;
    double dx2;
    double dx3;
    double dy1;
    double dy2;
    double dy3;
    double a;
    double b;
    double c;
    double d;
    double e;
    double f;
    double g = 0.0;
    double h = 0.0;
    uint64_t row_bytes;
    uint64_t required_source_bytes;
    uint32_t y;
    lw_status status;
    if (crop_width == NULL || crop_height == NULL || crop_byte_count == NULL)
        return LW_STATUS_INVALID_ARGUMENT;
    status = lw_crop_quad_size(box, &output_width, &output_height, &required_bytes);
    if (status != LW_STATUS_OK)
        return status;
    if (source == NULL || crop == NULL || source_width == 0u || source_height == 0u ||
        !load_points(box, points))
        return LW_STATUS_INVALID_ARGUMENT;
    row_bytes = (uint64_t)source_width * 3u;
    if (row_bytes > UINT32_MAX || source_stride < row_bytes ||
        (uint64_t)(source_height - 1u) * source_stride > UINT64_MAX - row_bytes)
        return LW_STATUS_INVALID_SHAPE;
    required_source_bytes = (uint64_t)(source_height - 1u) * source_stride + row_bytes;
    if (source_byte_count < required_source_bytes || required_source_bytes > SIZE_MAX)
        return LW_STATUS_INVALID_SHAPE;
    if (crop_capacity < required_bytes || required_bytes > SIZE_MAX)
        return LW_STATUS_OUT_OF_BOUNDS;
    unrotated_width = (uint32_t)floor(point_distance(points[0], points[1]) + 0.5);
    unrotated_height = (uint32_t)floor(point_distance(points[0], points[3]) + 0.5);
    rotate_vertical = (double)unrotated_height >= (double)unrotated_width * 1.5;

    /* Solve the projective mapping from normalized crop coordinates (u, v) to
     * the source quadrilateral. g/h are zero for an affine parallelogram. */
    dx1 = points[1].x - points[2].x;
    dx2 = points[3].x - points[2].x;
    dx3 = points[0].x - points[1].x + points[2].x - points[3].x;
    dy1 = points[1].y - points[2].y;
    dy2 = points[3].y - points[2].y;
    dy3 = points[0].y - points[1].y + points[2].y - points[3].y;
    if (fabs(dx3) > 1.0e-12 || fabs(dy3) > 1.0e-12) {
        double denominator = dx1 * dy2 - dx2 * dy1;
        if (!isfinite(denominator) || fabs(denominator) <= 1.0e-12)
            return LW_STATUS_INVALID_SHAPE;
        g = (dx3 * dy2 - dx2 * dy3) / denominator;
        h = (dx1 * dy3 - dx3 * dy1) / denominator;
    }
    a = points[1].x - points[0].x + g * points[1].x;
    b = points[3].x - points[0].x + h * points[3].x;
    c = points[0].x;
    d = points[1].y - points[0].y + g * points[1].y;
    e = points[3].y - points[0].y + h * points[3].y;
    f = points[0].y;
    /* Sample backwards from destination to source. Backward mapping avoids
     * holes, while bilinear sampling gives smoother character edges. */
    for (y = 0u; y < unrotated_height; ++y) {
        uint32_t x;
        double v = (double)y / unrotated_height;
        for (x = 0u; x < unrotated_width; ++x) {
            double u = (double)x / unrotated_width;
            double denominator = g * u + h * v + 1.0;
            double source_x;
            double source_y;
            /* Very tall regions are rotated into the horizontal orientation
             * expected by PP-OCR REC before optional CLS correction. */
            uint32_t destination_x = rotate_vertical ? unrotated_height - 1u - y : x;
            uint32_t destination_y = rotate_vertical ? x : y;
            uint64_t destination = ((uint64_t)destination_y * output_width + destination_x) * 3u;
            uint32_t channel;
            if (!isfinite(denominator) || fabs(denominator) <= 1.0e-12)
                return LW_STATUS_INVALID_SHAPE;
            source_x = (a * u + b * v + c) / denominator;
            source_y = (d * u + e * v + f) / denominator;
            if (!isfinite(source_x) || !isfinite(source_y))
                return LW_STATUS_INVALID_SHAPE;
            for (channel = 0u; channel < 3u; ++channel) {
                crop[(size_t)(destination + channel)] =
                    sample_channel(source, source_width, source_height, source_stride, source_x,
                                   source_y, channel);
            }
        }
    }
    *crop_width = output_width;
    *crop_height = output_height;
    *crop_byte_count = required_bytes;
    return LW_STATUS_OK;
}

void lw_rotate_bgr_u8_180(uint8_t* pixels, uint32_t width, uint32_t height) {
    uint64_t pixel_count;
    uint64_t left;
    if (pixels == NULL)
        return;
    pixel_count = (uint64_t)width * height;
    for (left = 0u; left < pixel_count / 2u; ++left) {
        uint64_t right = pixel_count - 1u - left;
        uint32_t channel;
        for (channel = 0u; channel < 3u; ++channel) {
            uint8_t value = pixels[(size_t)(left * 3u + channel)];
            pixels[(size_t)(left * 3u + channel)] = pixels[(size_t)(right * 3u + channel)];
            pixels[(size_t)(right * 3u + channel)] = value;
        }
    }
}
