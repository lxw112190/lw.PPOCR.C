#include "det_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct det_point {
    float x;
    float y;
} det_point;

typedef struct det_rectangle {
    float ux;
    float uy;
    float vx;
    float vy;
    float min_u;
    float max_u;
    float min_v;
    float max_v;
} det_rectangle;

static int compare_points(const void* left_value, const void* right_value) {
    const det_point* left = (const det_point*)left_value;
    const det_point* right = (const det_point*)right_value;
    if (left->x < right->x) return -1;
    if (left->x > right->x) return 1;
    if (left->y < right->y) return -1;
    if (left->y > right->y) return 1;
    return 0;
}

static float cross(det_point origin, det_point first, det_point second) {
    return (first.x - origin.x) * (second.y - origin.y) -
        (first.y - origin.y) * (second.x - origin.x);
}

static uint32_t convex_hull(
    det_point* points,
    uint32_t point_count,
    det_point* hull) {
    uint32_t unique_count = 0u;
    uint32_t index;
    uint32_t count = 0u;
    if (point_count < 3u) {
        return 0u;
    }
    qsort(points, point_count, sizeof(*points), compare_points);
    for (index = 0u; index < point_count; ++index) {
        if (unique_count == 0u ||
            points[index].x != points[unique_count - 1u].x ||
            points[index].y != points[unique_count - 1u].y) {
            points[unique_count++] = points[index];
        }
    }
    if (unique_count < 3u) {
        return 0u;
    }
    for (index = 0u; index < unique_count; ++index) {
        while (count >= 2u && cross(hull[count - 2u], hull[count - 1u],
                                    points[index]) <= 0.0f) {
            --count;
        }
        hull[count++] = points[index];
    }
    {
        uint32_t lower_count = count;
        for (index = unique_count - 1u; index > 0u; --index) {
            while (count > lower_count &&
                   cross(hull[count - 2u], hull[count - 1u],
                         points[index - 1u]) <= 0.0f) {
                --count;
            }
            hull[count++] = points[index - 1u];
        }
    }
    return count > 1u ? count - 1u : 0u;
}

static int minimum_rectangle(
    const det_point* hull,
    uint32_t hull_count,
    det_rectangle* rectangle) {
    float best_area = INFINITY;
    uint32_t edge;
    if (hull == NULL || rectangle == NULL || hull_count < 3u) {
        return 0;
    }
    for (edge = 0u; edge < hull_count; ++edge) {
        det_point current = hull[edge];
        det_point next = hull[(edge + 1u) % hull_count];
        float dx = next.x - current.x;
        float dy = next.y - current.y;
        float length = sqrtf(dx * dx + dy * dy);
        float ux;
        float uy;
        float vx;
        float vy;
        float min_u = INFINITY;
        float max_u = -INFINITY;
        float min_v = INFINITY;
        float max_v = -INFINITY;
        uint32_t point;
        float area;
        if (length <= 0.0f) {
            continue;
        }
        ux = dx / length;
        uy = dy / length;
        vx = -uy;
        vy = ux;
        for (point = 0u; point < hull_count; ++point) {
            float projection_u = hull[point].x * ux + hull[point].y * uy;
            float projection_v = hull[point].x * vx + hull[point].y * vy;
            if (projection_u < min_u) min_u = projection_u;
            if (projection_u > max_u) max_u = projection_u;
            if (projection_v < min_v) min_v = projection_v;
            if (projection_v > max_v) max_v = projection_v;
        }
        area = (max_u - min_u) * (max_v - min_v);
        if (area < best_area) {
            best_area = area;
            rectangle->ux = ux;
            rectangle->uy = uy;
            rectangle->vx = vx;
            rectangle->vy = vy;
            rectangle->min_u = min_u;
            rectangle->max_u = max_u;
            rectangle->min_v = min_v;
            rectangle->max_v = max_v;
        }
    }
    return isfinite(best_area) && best_area > 0.0f;
}

static det_point from_projection(
    const det_rectangle* rectangle,
    float projection_u,
    float projection_v) {
    det_point point;
    point.x = projection_u * rectangle->ux + projection_v * rectangle->vx;
    point.y = projection_u * rectangle->uy + projection_v * rectangle->vy;
    return point;
}

static void rectangle_points(
    const det_rectangle* rectangle,
    float expansion,
    det_point points[4]) {
    points[0] = from_projection(rectangle,
        rectangle->min_u - expansion, rectangle->min_v - expansion);
    points[1] = from_projection(rectangle,
        rectangle->max_u + expansion, rectangle->min_v - expansion);
    points[2] = from_projection(rectangle,
        rectangle->max_u + expansion, rectangle->max_v + expansion);
    points[3] = from_projection(rectangle,
        rectangle->min_u - expansion, rectangle->max_v + expansion);
}

static void order_clockwise(det_point points[4]) {
    det_point sorted[4];
    det_point left_top;
    det_point left_bottom;
    det_point right_top;
    det_point right_bottom;
    memcpy(sorted, points, sizeof(sorted));
    qsort(sorted, 4u, sizeof(sorted[0]), compare_points);
    left_top = sorted[0].y < sorted[1].y ? sorted[0] : sorted[1];
    left_bottom = sorted[0].y < sorted[1].y ? sorted[1] : sorted[0];
    right_top = sorted[2].y < sorted[3].y ? sorted[2] : sorted[3];
    right_bottom = sorted[2].y < sorted[3].y ? sorted[3] : sorted[2];
    points[0] = left_top;
    points[1] = right_top;
    points[2] = right_bottom;
    points[3] = left_bottom;
}

static float rectangle_score(
    const float* prediction,
    uint32_t width,
    uint32_t height,
    const det_rectangle* rectangle) {
    det_point corners[4];
    float minimum_x;
    float maximum_x;
    float minimum_y;
    float maximum_y;
    int32_t left;
    int32_t right;
    int32_t top;
    int32_t bottom;
    double sum = 0.0;
    uint64_t count = 0u;
    int32_t y;
    rectangle_points(rectangle, 0.0f, corners);
    minimum_x = maximum_x = corners[0].x;
    minimum_y = maximum_y = corners[0].y;
    for (y = 1; y < 4; ++y) {
        if (corners[y].x < minimum_x) minimum_x = corners[y].x;
        if (corners[y].x > maximum_x) maximum_x = corners[y].x;
        if (corners[y].y < minimum_y) minimum_y = corners[y].y;
        if (corners[y].y > maximum_y) maximum_y = corners[y].y;
    }
    left = (int32_t)floorf(minimum_x);
    right = (int32_t)ceilf(maximum_x);
    top = (int32_t)floorf(minimum_y);
    bottom = (int32_t)ceilf(maximum_y);
    if (left < 0) left = 0;
    if (top < 0) top = 0;
    if (right >= (int32_t)width) right = (int32_t)width - 1;
    if (bottom >= (int32_t)height) bottom = (int32_t)height - 1;
    for (y = top; y <= bottom; ++y) {
        int32_t x;
        for (x = left; x <= right; ++x) {
            float projection_u = x * rectangle->ux + y * rectangle->uy;
            float projection_v = x * rectangle->vx + y * rectangle->vy;
            if (projection_u >= rectangle->min_u && projection_u <= rectangle->max_u &&
                projection_v >= rectangle->min_v && projection_v <= rectangle->max_v) {
                sum += prediction[(size_t)((uint64_t)(uint32_t)y * width +
                    (uint32_t)x)];
                ++count;
            }
        }
    }
    return count == 0u ? 0.0f : (float)(sum / count);
}

static float clamp_float(float value, float maximum) {
    if (value < 0.0f) return 0.0f;
    if (value > maximum) return maximum;
    return value;
}

static void sort_reading_order(lw_detection_box* boxes, uint32_t count) {
    uint32_t index;
    for (index = 1u; index < count; ++index) {
        lw_detection_box value = boxes[index];
        uint32_t position = index;
        while (position > 0u) {
            lw_detection_box* previous = &boxes[position - 1u];
            float delta_y = previous->y1 - value.y1;
            int comes_after = fabsf(delta_y) < 10.0f ?
                previous->x1 > value.x1 : previous->y1 > value.y1;
            if (!comes_after) break;
            boxes[position] = *previous;
            --position;
        }
        boxes[position] = value;
    }
}

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
    uint32_t* box_count) {
    uint64_t pixel_count;
    uint8_t* bitmap = NULL;
    uint8_t* visited = NULL;
    uint32_t* queue = NULL;
    det_point* points = NULL;
    det_point* hull = NULL;
    lw_detection_box* results = NULL;
    uint32_t result_count = 0u;
    uint32_t candidate_count = 0u;
    uint64_t start;
    lw_status status = LW_STATUS_OK;
    if (box_count != NULL) *box_count = 0u;
    if (prediction == NULL || box_count == NULL || map_width == 0u ||
        map_height == 0u || source_width == 0u || source_height == 0u ||
        max_candidates == 0u || use_dilation > 1u ||
        !isfinite(bitmap_threshold) || bitmap_threshold < 0.0f || bitmap_threshold > 1.0f ||
        !isfinite(box_threshold) || box_threshold < 0.0f || box_threshold > 1.0f ||
        !isfinite(unclip_ratio) || unclip_ratio <= 0.0f ||
        !isfinite(width_ratio) || width_ratio <= 0.0f ||
        !isfinite(height_ratio) || height_ratio <= 0.0f ||
        (boxes == NULL && box_capacity != 0u)) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    if (map_width > INT32_MAX || map_height > INT32_MAX) {
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    pixel_count = (uint64_t)map_width * map_height;
    if (pixel_count > UINT32_MAX ||
        pixel_count > SIZE_MAX / sizeof(*queue) ||
        pixel_count > SIZE_MAX / sizeof(*points) ||
        pixel_count > SIZE_MAX / (2u * sizeof(*hull)) ||
        max_candidates > SIZE_MAX / sizeof(*results)) {
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    bitmap = (uint8_t*)malloc((size_t)pixel_count);
    visited = (uint8_t*)calloc((size_t)pixel_count, 1u);
    queue = (uint32_t*)malloc((size_t)pixel_count * sizeof(*queue));
    points = (det_point*)malloc((size_t)pixel_count * sizeof(*points));
    hull = (det_point*)malloc((size_t)pixel_count * 2u * sizeof(*hull));
    results = (lw_detection_box*)calloc(max_candidates, sizeof(*results));
    if (bitmap == NULL || visited == NULL || queue == NULL || points == NULL ||
        hull == NULL || results == NULL) {
        status = LW_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    for (start = 0u; start < pixel_count; ++start) {
        if (!isfinite(prediction[(size_t)start])) {
            status = LW_STATUS_INVALID_ARGUMENT;
            goto cleanup;
        }
        bitmap[(size_t)start] = prediction[(size_t)start] > bitmap_threshold ? 1u : 0u;
    }
    if (use_dilation != 0u) {
        uint8_t* dilated = (uint8_t*)calloc((size_t)pixel_count, 1u);
        if (dilated == NULL) {
            status = LW_STATUS_OUT_OF_MEMORY;
            goto cleanup;
        }
        for (start = 0u; start < pixel_count; ++start) {
            if (bitmap[(size_t)start] != 0u) {
                uint32_t x = (uint32_t)(start % map_width);
                uint32_t y = (uint32_t)(start / map_width);
                dilated[(size_t)start] = 1u;
                if (x + 1u < map_width) dilated[(size_t)(start + 1u)] = 1u;
                if (y + 1u < map_height) dilated[(size_t)(start + map_width)] = 1u;
                if (x + 1u < map_width && y + 1u < map_height)
                    dilated[(size_t)(start + map_width + 1u)] = 1u;
            }
        }
        free(bitmap);
        bitmap = dilated;
    }
    for (start = 0u; start < pixel_count && candidate_count < max_candidates; ++start) {
        uint32_t head = 0u;
        uint32_t tail = 0u;
        uint32_t point_count = 0u;
        uint32_t hull_count;
        det_rectangle rectangle;
        float rectangle_width;
        float rectangle_height;
        float shortest_side;
        float score;
        float area;
        float perimeter;
        float expansion;
        det_point corners[4];
        uint32_t point_index;
        if (bitmap[(size_t)start] == 0u || visited[(size_t)start] != 0u) continue;
        ++candidate_count;
        visited[(size_t)start] = 1u;
        queue[tail++] = (uint32_t)start;
        while (head < tail) {
            uint32_t position = queue[head++];
            uint32_t x = position % map_width;
            uint32_t y = position / map_width;
            int boundary = x == 0u || y == 0u || x + 1u == map_width ||
                y + 1u == map_height;
            int32_t dy;
            if (!boundary && bitmap[position - 1u] == 0u) boundary = 1;
            if (!boundary && x + 1u < map_width && bitmap[position + 1u] == 0u) boundary = 1;
            if (!boundary && y > 0u && bitmap[position - map_width] == 0u) boundary = 1;
            if (!boundary && y + 1u < map_height && bitmap[position + map_width] == 0u) boundary = 1;
            if (boundary) {
                points[point_count].x = (float)x;
                points[point_count].y = (float)y;
                ++point_count;
            }
            for (dy = -1; dy <= 1; ++dy) {
                int32_t dx;
                for (dx = -1; dx <= 1; ++dx) {
                    int64_t nx;
                    int64_t ny;
                    uint32_t neighbor;
                    if (dx == 0 && dy == 0) continue;
                    nx = (int64_t)x + dx;
                    ny = (int64_t)y + dy;
                    if (nx < 0 || nx >= map_width || ny < 0 || ny >= map_height) continue;
                    neighbor = (uint32_t)ny * map_width + (uint32_t)nx;
                    if (bitmap[neighbor] != 0u && visited[neighbor] == 0u) {
                        visited[neighbor] = 1u;
                        queue[tail++] = neighbor;
                    }
                }
            }
        }
        if (point_count <= 2u) continue;
        hull_count = convex_hull(points, point_count, hull);
        if (!minimum_rectangle(hull, hull_count, &rectangle)) continue;
        rectangle_width = rectangle.max_u - rectangle.min_u;
        rectangle_height = rectangle.max_v - rectangle.min_v;
        shortest_side = rectangle_width < rectangle_height ? rectangle_width : rectangle_height;
        if (shortest_side < 3.0f) continue;
        score = rectangle_score(prediction, map_width, map_height, &rectangle);
        if (!isfinite(score) || score < box_threshold) continue;
        area = rectangle_width * rectangle_height;
        perimeter = 2.0f * (rectangle_width + rectangle_height);
        if (perimeter <= 0.0f) continue;
        expansion = area * unclip_ratio / perimeter;
        if (shortest_side + 2.0f * expansion < 5.0f) continue;
        rectangle_points(&rectangle, expansion, corners);
        order_clockwise(corners);
        for (point_index = 0u; point_index < 4u; ++point_index) {
            corners[point_index].x = clamp_float(corners[point_index].x / width_ratio,
                (float)(source_width - 1u));
            corners[point_index].y = clamp_float(corners[point_index].y / height_ratio,
                (float)(source_height - 1u));
        }
        if (hypotf(corners[0].x - corners[1].x, corners[0].y - corners[1].y) <= 4.0f ||
            hypotf(corners[0].x - corners[3].x, corners[0].y - corners[3].y) <= 4.0f) {
            continue;
        }
        results[result_count].x1 = corners[0].x;
        results[result_count].y1 = corners[0].y;
        results[result_count].x2 = corners[1].x;
        results[result_count].y2 = corners[1].y;
        results[result_count].x3 = corners[2].x;
        results[result_count].y3 = corners[2].y;
        results[result_count].x4 = corners[3].x;
        results[result_count].y4 = corners[3].y;
        results[result_count].score = score;
        ++result_count;
    }
    sort_reading_order(results, result_count);
    *box_count = result_count;
    if (boxes != NULL) {
        if (box_capacity < result_count) {
            status = LW_STATUS_OUT_OF_BOUNDS;
            goto cleanup;
        }
        memcpy(boxes, results, (size_t)result_count * sizeof(*boxes));
    }
cleanup:
    free(results);
    free(hull);
    free(points);
    free(queue);
    free(visited);
    free(bitmap);
    return status;
}
