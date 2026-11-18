#include "det_internal.h"
#include "profile_internal.h"
#include "../simd/simd_kernels.h"

/*
 * Bounded DB-style postprocessing: threshold the probability map, collect
 * connected components, fit/expand quadrilaterals and restore source-image
 * coordinates. Candidate limits prevent hostile images from growing memory.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct det_point {
    float x;
    float y;
} det_point;

enum {
    DET_UNCLIP_POINTS_PER_CORNER = 228,
    DET_UNCLIP_EXTRA_POINTS = 8,
    DET_UNCLIP_QUAD_CAPACITY = 4 * DET_UNCLIP_POINTS_PER_CORNER + DET_UNCLIP_EXTRA_POINTS
};

typedef struct det_contour_row {
    uint32_t left;
    uint32_t end; /* Exclusive right edge. Zero denotes an untouched row. */
} det_contour_row;

typedef struct det_contour {
    det_contour_row* rows;
    uint32_t first;
    uint32_t end;
} det_contour;

static void contour_include(det_contour* contour, uint32_t y, uint32_t left, uint32_t right) {
    det_contour_row* row = &contour->rows[y];
    if (row->end == 0u || left < row->left) row->left = left;
    if (right >= row->end) row->end = right + 1u;
    if (y < contour->first) contour->first = y;
    if (y >= contour->end) contour->end = y + 1u;
}

static uint32_t contour_collect(det_contour* contour, det_point* points) {
    uint32_t count = 0u;
    uint32_t y;
    for (y = contour->first; y < contour->end; ++y) {
        det_contour_row* row = &contour->rows[y];
        if (row->end == 0u) continue;
        points[count++] = (det_point){(float)row->left, (float)y};
        if (row->left != row->end - 1u) {
            points[count++] = (det_point){(float)(row->end - 1u), (float)y};
        }
        row->end = 0u;
    }
    contour->first = UINT32_MAX;
    contour->end = 0u;
    return count;
}

/* Keep allocation overflow checks meaningful on both 32-bit and 64-bit
 * targets. Passing the count through uint64_t avoids comparisons that GCC can
 * prove are always false when the original count is only uint32_t. */
static int allocation_fits(uint64_t count, size_t element_size) {
    return element_size != 0u && count <= (uint64_t)(SIZE_MAX / element_size);
}

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
    if (left->x < right->x)
        return -1;
    if (left->x > right->x)
        return 1;
    if (left->y < right->y)
        return -1;
    if (left->y > right->y)
        return 1;
    return 0;
}

static float cross(det_point origin, det_point first, det_point second) {
    return (first.x - origin.x) * (second.y - origin.y) -
           (first.y - origin.y) * (second.x - origin.x);
}

static uint32_t convex_hull(det_point* points, uint32_t point_count, det_point* hull) {
    uint32_t unique_count = 0u;
    uint32_t index;
    uint32_t count = 0u;
    if (point_count < 3u) {
        return 0u;
    }
    qsort(points, point_count, sizeof(*points), compare_points);
    for (index = 0u; index < point_count; ++index) {
        if (unique_count == 0u || points[index].x != points[unique_count - 1u].x ||
            points[index].y != points[unique_count - 1u].y) {
            points[unique_count++] = points[index];
        }
    }
    if (unique_count < 3u) {
        return 0u;
    }
    for (index = 0u; index < unique_count; ++index) {
        while (count >= 2u && cross(hull[count - 2u], hull[count - 1u], points[index]) <= 0.0f) {
            --count;
        }
        hull[count++] = points[index];
    }
    {
        uint32_t lower_count = count;
        for (index = unique_count - 1u; index > 0u; --index) {
            while (count > lower_count &&
                   cross(hull[count - 2u], hull[count - 1u], points[index - 1u]) <= 0.0f) {
                --count;
            }
            hull[count++] = points[index - 1u];
        }
    }
    return count > 1u ? count - 1u : 0u;
}

static int minimum_rectangle(const det_point* hull, uint32_t hull_count, det_rectangle* rectangle) {
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
            if (projection_u < min_u)
                min_u = projection_u;
            if (projection_u > max_u)
                max_u = projection_u;
            if (projection_v < min_v)
                min_v = projection_v;
            if (projection_v > max_v)
                max_v = projection_v;
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

static det_point from_projection(const det_rectangle* rectangle, float projection_u,
                                 float projection_v) {
    det_point point;
    point.x = projection_u * rectangle->ux + projection_v * rectangle->vx;
    point.y = projection_u * rectangle->uy + projection_v * rectangle->vy;
    return point;
}

static void rectangle_points(const det_rectangle* rectangle, float expansion, det_point points[4]) {
    points[0] =
        from_projection(rectangle, rectangle->min_u - expansion, rectangle->min_v - expansion);
    points[1] =
        from_projection(rectangle, rectangle->max_u + expansion, rectangle->min_v - expansion);
    points[2] =
        from_projection(rectangle, rectangle->max_u + expansion, rectangle->max_v + expansion);
    points[3] =
        from_projection(rectangle, rectangle->min_u - expansion, rectangle->max_v + expansion);
}

static float cross_points(det_point origin, det_point first, det_point second) {
    return (first.x - origin.x) * (second.y - origin.y) -
           (first.y - origin.y) * (second.x - origin.x);
}

/* PaddleOCR order_points_clockwise: x+y minimal is top-left, maximal is
 * bottom-right, and the y-x difference splits the remaining two corners. A
 * 45-degree tie falls back to centroid-angle ordering. */
static int is_positive_convex_cycle(const det_point points[4]) {
    uint32_t i;
    for (i = 0u; i < 4u; ++i) {
        if (cross_points(points[i], points[(i + 1u) & 3u], points[(i + 2u) & 3u]) <= 0.0f) {
            return 0;
        }
    }
    return 1;
}

static void order_by_centroid_angle(det_point points[4], const det_point input[4]) {
    float center_x = 0.0f;
    float center_y = 0.0f;
    float angles[4];
    uint32_t order[4] = {0u, 1u, 2u, 3u};
    uint32_t start = 0u;
    uint32_t i;
    for (i = 0u; i < 4u; ++i) {
        center_x += input[i].x;
        center_y += input[i].y;
    }
    center_x /= 4.0f;
    center_y /= 4.0f;
    for (i = 0u; i < 4u; ++i) {
        angles[i] = atan2f(input[i].y - center_y, input[i].x - center_x);
    }
    for (i = 1u; i < 4u; ++i) {
        int32_t j = (int32_t)i - 1;
        uint32_t index = order[i];
        while (j >= 0 && angles[order[j]] > angles[index]) {
            order[(uint32_t)j + 1u] = order[(uint32_t)j];
            --j;
        }
        order[(uint32_t)j + 1u] = index;
    }
    for (i = 1u; i < 4u; ++i) {
        det_point candidate = input[order[i]];
        det_point best = input[order[start]];
        float candidate_sum = candidate.x + candidate.y;
        float best_sum = best.x + best.y;
        if (candidate_sum < best_sum ||
            (candidate_sum == best_sum &&
             (candidate.y < best.y || (candidate.y == best.y && candidate.x < best.x)))) {
            start = i;
        }
    }
    for (i = 0u; i < 4u; ++i) {
        points[i] = input[order[(start + i) & 3u]];
    }
}

static void order_clockwise(det_point points[4]) {
    det_point input[4];
    uint32_t min_sum_index = 0u;
    uint32_t max_sum_index = 0u;
    float min_sum;
    float max_sum;
    uint32_t first_remaining = UINT32_MAX;
    uint32_t second_remaining = UINT32_MAX;
    det_point first;
    det_point second;
    float first_diff;
    float second_diff;
    det_point top_right;
    det_point bottom_left;
    uint32_t i;
    memcpy(input, points, sizeof(input));
    min_sum = max_sum = input[0].x + input[0].y;
    for (i = 1u; i < 4u; ++i) {
        float sum = input[i].x + input[i].y;
        if (sum < min_sum) {
            min_sum = sum;
            min_sum_index = i;
        }
        if (sum > max_sum) {
            max_sum = sum;
            max_sum_index = i;
        }
    }
    for (i = 0u; i < 4u; ++i) {
        if (i == min_sum_index || i == max_sum_index) continue;
        if (first_remaining == UINT32_MAX) {
            first_remaining = i;
        } else {
            second_remaining = i;
        }
    }
    first = input[first_remaining];
    second = input[second_remaining];
    first_diff = first.y - first.x;
    second_diff = second.y - second.x;
    top_right = first_diff < second_diff ? first : second;
    bottom_left = first_diff < second_diff ? second : first;
    points[0] = input[min_sum_index];
    points[1] = top_right;
    points[2] = input[max_sum_index];
    points[3] = bottom_left;
    if (!is_positive_convex_cycle(points)) {
        order_by_centroid_angle(points, input);
    }
}

static double signed_area(const det_point* points, uint32_t count) {
    double area = 0.0;
    uint32_t i;
    for (i = 0u; i < count; ++i) {
        det_point a = points[i];
        det_point b = points[(i + 1u) % count];
        area += (double)a.x * b.y - (double)b.x * a.y;
    }
    return area * 0.5;
}

static float round_half_away(double value) {
    return (float)(value < 0.0 ? (int64_t)(value - 0.5) : (int64_t)(value + 0.5));
}

static int64_t round_half_even_i64(double value) {
    double rounded = floor(value + 0.5);
    if (rounded - value == 0.5 && ((int64_t)rounded & 1) != 0) {
        rounded -= 1.0;
    }
    return (int64_t)rounded;
}

static float round_half_even_f(float value) {
    float rounded = floorf(value + 0.5f);
    if (rounded - value == 0.5f && ((int64_t)rounded & 1) != 0) {
        rounded -= 1.0f;
    }
    return rounded;
}

static float clamp_coord(float value, float limit) {
    if (value < 0.0f) return 0.0f;
    if (value > limit) return limit;
    return value;
}

static void add_rounded(det_point* destination, uint32_t* count, uint32_t capacity,
                        double x, double y) {
    if (*count >= capacity) return;
    destination[*count].x = round_half_away(x);
    destination[*count].y = round_half_away(y);
    ++*count;
}

static void add_arc(det_point* destination, uint32_t* count, uint32_t capacity,
                    double center_x, double center_y, double start, double end, double radius) {
    const double pi = 3.14159265358979323846;
    double radius_abs = fabs(radius);
    double chord = radius_abs < 0.25 ? radius_abs : 0.25;
    double fraction = fabs(end - start) / (2.0 * pi);
    int64_t steps;
    int64_t cap;
    double x;
    double y;
    double step;
    double cos_step;
    double sin_step;
    int64_t i;
    if (fraction <= 0.0 || radius_abs <= 0.0) return;
    steps = (int64_t)(fraction * pi / acos(1.0 - chord / radius_abs));
    if (steps < 2) steps = 2;
    cap = (int64_t)(222.0 * fraction);
    if (steps > cap) steps = cap > 2 ? cap : 2;
    x = cos(start);
    y = sin(start);
    step = (end - start) / (double)steps;
    cos_step = cos(step);
    sin_step = sin(step);
    for (i = 0; i <= steps; ++i) {
        double next_x;
        add_rounded(destination, count, capacity, center_x + x * radius,
                    center_y + y * radius);
        next_x = x * cos_step - y * sin_step;
        y = x * sin_step + y * cos_step;
        x = next_x;
    }
}

/* Convex outward offset with round joins, the same construction as Clipper
 * 5.1.5 PolyOffsetBuilder for a single positive-delta polygon (the trailing
 * boolean union is unnecessary because the caller takes a min-area rect). */
static uint32_t offset_round(const det_point* box, uint32_t count, double delta,
                             lw_db_postprocess_workspace* workspace) {
    int64_t* pts = (int64_t*)workspace->unclip_pts;
    double* normals = (double*)workspace->unclip_normals;
    det_point* result = (det_point*)workspace->unclip_result;
    uint32_t result_count = 0u;
    uint32_t n = 0u;
    double winding = 0.0;
    uint32_t prev;
    uint32_t i;
    for (i = 0u; i < count; ++i) {
        int64_t x = round_half_even_i64((double)box[i].x);
        int64_t y = round_half_even_i64((double)box[i].y);
        if (n > 0u && pts[(size_t)(n - 1u) * 2u] == x && pts[(size_t)(n - 1u) * 2u + 1u] == y) {
            continue;
        }
        pts[(size_t)n * 2u] = x;
        pts[(size_t)n * 2u + 1u] = y;
        ++n;
    }
    if (n >= 2u && pts[0] == pts[(size_t)(n - 1u) * 2u] &&
        pts[1] == pts[(size_t)(n - 1u) * 2u + 1u]) {
        --n;
    }
    if (n < 3u) return 0u;
    for (i = 0u; i < n; ++i) {
        int64_t ax = pts[(size_t)i * 2u];
        int64_t ay = pts[(size_t)i * 2u + 1u];
        int64_t bx = pts[(size_t)((i + 1u) % n) * 2u];
        int64_t by = pts[(size_t)((i + 1u) % n) * 2u + 1u];
        winding += (double)ax * by - (double)bx * ay;
    }
    if (winding < 0.0) {
        for (i = 0u; i < n / 2u; ++i) {
            int64_t tx = pts[(size_t)i * 2u];
            int64_t ty = pts[(size_t)i * 2u + 1u];
            pts[(size_t)i * 2u] = pts[(size_t)(n - 1u - i) * 2u];
            pts[(size_t)i * 2u + 1u] = pts[(size_t)(n - 1u - i) * 2u + 1u];
            pts[(size_t)(n - 1u - i) * 2u] = tx;
            pts[(size_t)(n - 1u - i) * 2u + 1u] = ty;
        }
    }
    for (i = 0u; i < n; ++i) {
        double dx = (double)(pts[(size_t)((i + 1u) % n) * 2u] - pts[(size_t)i * 2u]);
        double dy = (double)(pts[(size_t)((i + 1u) % n) * 2u + 1u] - pts[(size_t)i * 2u + 1u]);
        double length = sqrt(dx * dx + dy * dy);
        if (length > 0.0) {
            dx /= length;
            dy /= length;
        }
        normals[(size_t)i * 2u] = dy;
        normals[(size_t)i * 2u + 1u] = -dx;
    }
    prev = n - 1u;
    for (i = 0u; i < n; ++i) {
        int64_t x = pts[(size_t)i * 2u];
        int64_t y = pts[(size_t)i * 2u + 1u];
        double incoming_x = normals[(size_t)prev * 2u];
        double incoming_y = normals[(size_t)prev * 2u + 1u];
        double outgoing_x = normals[(size_t)i * 2u];
        double outgoing_y = normals[(size_t)i * 2u + 1u];
        add_rounded(result, &result_count, workspace->unclip_capacity,
                    (double)x + incoming_x * delta, (double)y + incoming_y * delta);
        if ((incoming_x * outgoing_y - outgoing_x * incoming_y) * delta >= 0.0) {
            if (outgoing_x * incoming_x + outgoing_y * incoming_y < 0.985) {
                double start = atan2(incoming_y, incoming_x);
                double end = atan2(outgoing_y, outgoing_x);
                if (end < start) end += 2.0 * 3.14159265358979323846;
                add_arc(result, &result_count, workspace->unclip_capacity,
                        (double)x, (double)y, start, end, delta);
            }
        } else {
            add_rounded(result, &result_count, workspace->unclip_capacity,
                        (double)x, (double)y);
        }
        add_rounded(result, &result_count, workspace->unclip_capacity,
                    (double)x + outgoing_x * delta, (double)y + outgoing_y * delta);
        prev = i;
    }
    return result_count;
}

/* cv2.fillPoly scanline replica: per-row crossing parity against float
 * thresholds plus exact integer on-edge hits, over an int32-truncated
 * polygon, averaging the prediction map. */
static float polygon_score(const float* prediction, uint32_t width, uint32_t height,
                           const det_point* polygon, uint32_t count,
                           lw_db_postprocess_workspace* workspace) {
    float* thresholds = (float*)workspace->score_thresholds;
    int32_t* hit_low = (int32_t*)workspace->score_hit_low;
    int32_t* hit_high = (int32_t*)workspace->score_hit_high;
    int32_t* poly_x = (int32_t*)workspace->score_poly_x;
    int32_t* poly_y = (int32_t*)workspace->score_poly_y;
    float min_x = INFINITY;
    float max_x = -INFINITY;
    float min_y = INFINITY;
    float max_y = -INFINITY;
    int32_t left;
    int32_t right;
    int32_t top;
    int32_t bottom;
    double sum = 0.0;
    uint64_t pixel_count = 0u;
    int32_t y;
    uint32_t i;
    for (i = 0u; i < count; ++i) {
        if (polygon[i].x < min_x) min_x = polygon[i].x;
        if (polygon[i].x > max_x) max_x = polygon[i].x;
        if (polygon[i].y < min_y) min_y = polygon[i].y;
        if (polygon[i].y > max_y) max_y = polygon[i].y;
    }
    left = (int32_t)floorf(min_x);
    right = (int32_t)ceilf(max_x);
    top = (int32_t)floorf(min_y);
    bottom = (int32_t)ceilf(max_y);
    if (left < 0) left = 0;
    if (top < 0) top = 0;
    if (right >= (int32_t)width) right = (int32_t)width - 1;
    if (bottom >= (int32_t)height) bottom = (int32_t)height - 1;
    if (left > right || top > bottom) return 0.0f;
    for (i = 0u; i < count; ++i) {
        poly_x[i] = (int32_t)polygon[i].x - left;
        poly_y[i] = (int32_t)polygon[i].y - top;
    }
    for (y = top; y <= bottom; ++y) {
        int32_t py = y - top;
        uint32_t threshold_count = 0u;
        uint32_t hit_count = 0u;
        uint32_t threshold_index = 0u;
        int32_t x;
        uint32_t j;
        for (i = 0u, j = count - 1u; i < count; j = i++) {
            int32_t ax = poly_x[i];
            int32_t ay = poly_y[i];
            int32_t bx = poly_x[j];
            int32_t by = poly_y[j];
            int32_t dx = bx - ax;
            int32_t dy = by - ay;
            if ((ay > py) != (by > py)) {
                thresholds[threshold_count++] = (float)dx * (py - ay) / dy + ax;
            }
            if (py < (ay < by ? ay : by) || py > (ay > by ? ay : by)) continue;
            if (dy == 0) {
                hit_low[hit_count] = ax < bx ? ax : bx;
                hit_high[hit_count] = ax > bx ? ax : bx;
                ++hit_count;
            } else {
                int64_t numerator = (int64_t)(py - ay) * dx;
                if (numerator % dy == 0) {
                    int32_t hx = ax + (int32_t)(numerator / dy);
                    if (hx >= (ax < bx ? ax : bx) && hx <= (ax > bx ? ax : bx)) {
                        hit_low[hit_count] = hx;
                        hit_high[hit_count] = hx;
                        ++hit_count;
                    }
                }
            }
        }
        for (i = 1u; i < threshold_count; ++i) {
            float value = thresholds[i];
            uint32_t insert = i;
            while (insert > 0u && thresholds[insert - 1u] > value) {
                thresholds[insert] = thresholds[insert - 1u];
                --insert;
            }
            thresholds[insert] = value;
        }
        for (x = left; x <= right; ++x) {
            int32_t px = x - left;
            int inside;
            while (threshold_index < threshold_count && thresholds[threshold_index] <= px) {
                ++threshold_index;
            }
            inside = ((threshold_count - threshold_index) & 1u) != 0u;
            if (!inside) {
                for (i = 0u; i < hit_count; ++i) {
                    if (px >= hit_low[i] && px <= hit_high[i]) {
                        inside = 1;
                        break;
                    }
                }
            }
            if (inside) {
                sum += prediction[(size_t)((uint64_t)(uint32_t)y * width + (uint32_t)x)];
                ++pixel_count;
            }
        }
    }
    return pixel_count == 0u ? 0.0f : (float)(sum / (double)pixel_count);
}

/* Grow the reusable scratch buffers to cover this image. Buffers only grow;
 * a detector that processes many same-size images allocates once. */
static lw_status workspace_grow(lw_db_postprocess_workspace* workspace, uint32_t pixels,
                                uint32_t map_height, uint32_t max_candidates) {
    uint8_t* bitmap = workspace->bitmap;
    uint8_t* visited = workspace->visited;
    uint8_t* background_visited = workspace->background_visited;
    uint8_t* dilated = workspace->dilated;
    uint32_t* stack = workspace->queue;
    lw_detection_box* results = (lw_detection_box*)workspace->results;
    if (workspace->pixel_capacity < pixels) {
        bitmap = (uint8_t*)malloc((size_t)pixels);
        visited = (uint8_t*)malloc((size_t)pixels);
        background_visited = (uint8_t*)malloc((size_t)pixels);
        dilated = (uint8_t*)malloc((size_t)pixels);
        stack = (uint32_t*)malloc(3u * (size_t)pixels * sizeof(*stack));
        if (bitmap == NULL || visited == NULL || background_visited == NULL ||
            dilated == NULL || stack == NULL) {
            free(stack);
            free(dilated);
            free(background_visited);
            free(visited);
            free(bitmap);
            return LW_STATUS_OUT_OF_MEMORY;
        }
        free(workspace->queue);
        free(workspace->dilated);
        free(workspace->background_visited);
        free(workspace->visited);
        free(workspace->bitmap);
        workspace->bitmap = bitmap;
        workspace->visited = visited;
        workspace->background_visited = background_visited;
        workspace->dilated = dilated;
        workspace->queue = stack;
        workspace->pixel_capacity = pixels;
    }
    if (workspace->contour_row_capacity < map_height) {
        det_contour_row* rows = (det_contour_row*)calloc(map_height, sizeof(*rows));
        det_point* points = (det_point*)malloc((size_t)map_height * 2u * sizeof(*points));
        if (rows == NULL || points == NULL) {
            free(points);
            free(rows);
            return LW_STATUS_OUT_OF_MEMORY;
        }
        free(workspace->contour_rows);
        free(workspace->points);
        workspace->contour_rows = rows;
        workspace->points = points;
        workspace->contour_row_capacity = map_height;
    }
    if (workspace->result_capacity < max_candidates) {
        results = (lw_detection_box*)calloc(max_candidates, sizeof(*results));
        if (results == NULL) {
            return LW_STATUS_OUT_OF_MEMORY;
        }
        free(workspace->results);
        workspace->results = results;
        workspace->result_capacity = max_candidates;
    }
    return LW_STATUS_OK;
}

/* The monotone chain can write up to twice the input count. */
static lw_status workspace_grow_hull(lw_db_postprocess_workspace* workspace, uint32_t count) {
    det_point* hull;
    if (workspace->hull_capacity >= count) return LW_STATUS_OK;
    if (count > UINT32_MAX / 2u || !allocation_fits(count, 2u * sizeof(*hull))) {
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    hull = (det_point*)realloc(workspace->hull, (size_t)count * 2u * sizeof(*hull));
    if (hull == NULL) return LW_STATUS_OUT_OF_MEMORY;
    workspace->hull = hull;
    workspace->hull_capacity = count;
    return LW_STATUS_OK;
}

/* Grow the polygon scratch (unclip round-join and PolygonScore arrays). */
static lw_status workspace_grow_polygon(lw_db_postprocess_workspace* workspace, uint32_t count) {
    uint32_t capacity;
    if (count > (UINT32_MAX - DET_UNCLIP_EXTRA_POINTS) / DET_UNCLIP_POINTS_PER_CORNER) {
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    capacity = count * DET_UNCLIP_POINTS_PER_CORNER + DET_UNCLIP_EXTRA_POINTS;
    if (workspace->unclip_capacity < capacity) {
        int64_t* pts = (int64_t*)malloc((size_t)count * 2u * sizeof(*pts));
        double* normals = (double*)malloc((size_t)count * 2u * sizeof(*normals));
        det_point* result = (det_point*)malloc((size_t)capacity * sizeof(*result));
        if (pts == NULL || normals == NULL || result == NULL) {
            free(result);
            free(normals);
            free(pts);
            return LW_STATUS_OUT_OF_MEMORY;
        }
        free(workspace->unclip_result);
        free(workspace->unclip_normals);
        free(workspace->unclip_pts);
        workspace->unclip_pts = pts;
        workspace->unclip_normals = normals;
        workspace->unclip_result = result;
        workspace->unclip_capacity = capacity;
    }
    if (workspace->score_capacity < count) {
        float* thresholds = (float*)malloc((size_t)count * sizeof(*thresholds));
        int32_t* hit_low = (int32_t*)malloc((size_t)count * sizeof(*hit_low));
        int32_t* hit_high = (int32_t*)malloc((size_t)count * sizeof(*hit_high));
        int32_t* poly_x = (int32_t*)malloc((size_t)count * sizeof(*poly_x));
        int32_t* poly_y = (int32_t*)malloc((size_t)count * sizeof(*poly_y));
        if (thresholds == NULL || hit_low == NULL || hit_high == NULL ||
            poly_x == NULL || poly_y == NULL) {
            free(poly_y);
            free(poly_x);
            free(hit_high);
            free(hit_low);
            free(thresholds);
            return LW_STATUS_OUT_OF_MEMORY;
        }
        free(workspace->score_poly_y);
        free(workspace->score_poly_x);
        free(workspace->score_hit_high);
        free(workspace->score_hit_low);
        free(workspace->score_thresholds);
        workspace->score_thresholds = thresholds;
        workspace->score_hit_low = hit_low;
        workspace->score_hit_high = hit_high;
        workspace->score_poly_x = poly_x;
        workspace->score_poly_y = poly_y;
        workspace->score_capacity = count;
    }
    return LW_STATUS_OK;
}

void lw_db_postprocess_workspace_free(lw_db_postprocess_workspace* workspace) {
    if (workspace == NULL) return;
    free(workspace->score_poly_y);
    free(workspace->score_poly_x);
    free(workspace->score_hit_high);
    free(workspace->score_hit_low);
    free(workspace->score_thresholds);
    free(workspace->unclip_result);
    free(workspace->unclip_normals);
    free(workspace->unclip_pts);
    free(workspace->results);
    free(workspace->hull);
    free(workspace->points);
    free(workspace->contour_rows);
    free(workspace->queue);
    free(workspace->dilated);
    free(workspace->background_visited);
    free(workspace->visited);
    free(workspace->bitmap);
    memset(workspace, 0, sizeof(*workspace));
}

/* Each foreground run is maximal and is marked visited in its entirety.
 * Both endpoints are boundary pixels. Every other point on that row lies
 * between the component's leftmost and rightmost endpoints, so those two
 * points alone preserve its convex hull. */
static void fill_foreground(const uint8_t* bitmap, uint8_t* visited,
                            uint32_t map_width, uint32_t map_height,
                            uint32_t start, det_contour* contour, uint32_t* stack) {
    uint32_t top = 0u;
    uint32_t seed_y = start / map_width;
    uint32_t seed_x = start % map_width;
    uint32_t seed_row = seed_y * map_width;
    uint32_t left = seed_x;
    uint32_t right = seed_x;
    while (left > 0u && bitmap[seed_row + left - 1u] != 0u &&
           visited[seed_row + left - 1u] == 0u) {
        --left;
    }
    while (right + 1u < map_width && bitmap[seed_row + right + 1u] != 0u &&
           visited[seed_row + right + 1u] == 0u) {
        ++right;
    }
    memset(visited + seed_row + left, 1u, right - left + 1u);
    stack[top++] = seed_y;
    stack[top++] = left;
    stack[top++] = right;
    while (top > 0u) {
        uint32_t run_right = stack[--top];
        uint32_t run_left = stack[--top];
        uint32_t y = stack[--top];
        uint32_t x;
        int32_t direction;
        uint32_t scan_from;
        uint32_t scan_to;
        contour_include(contour, y, run_left, run_right);
        scan_from = run_left > 0u ? run_left - 1u : 0u;
        scan_to = run_right + 1u < map_width ? run_right + 1u : map_width - 1u;
        for (direction = -1; direction <= 1; direction += 2) {
            int64_t neighbor_y = (int64_t)y + direction;
            uint32_t nrow;
            if (neighbor_y < 0 || neighbor_y >= (int64_t)map_height) continue;
            nrow = (uint32_t)neighbor_y * map_width;
            x = scan_from;
            while (x <= scan_to) {
                if (bitmap[nrow + x] != 0u && visited[nrow + x] == 0u) {
                    uint32_t next_left = x;
                    uint32_t next_right = x;
                    while (next_left > 0u && bitmap[nrow + next_left - 1u] != 0u &&
                           visited[nrow + next_left - 1u] == 0u) {
                        --next_left;
                    }
                    while (next_right + 1u < map_width &&
                           bitmap[nrow + next_right + 1u] != 0u &&
                           visited[nrow + next_right + 1u] == 0u) {
                        ++next_right;
                    }
                    memset(visited + nrow + next_left, 1u, next_right - next_left + 1u);
                    stack[top++] = (uint32_t)neighbor_y;
                    stack[top++] = next_left;
                    stack[top++] = next_right;
                    x = next_right + 2u;
                } else {
                    ++x;
                }
            }
        }
    }
}

/* Scanline flood fill over the 4-connected background component containing
 * `seed`. With contour != NULL this traces the component's contour
 * (OpenCV RETR_LIST background-hole parity); with NULL it only marks the
 * visited set. */
static void fill_background(const uint8_t* bitmap, uint8_t* visited,
                            uint32_t map_width, uint32_t map_height, uint32_t seed,
                            det_contour* contour, uint32_t* stack) {
    uint32_t top = 0u;
    uint32_t seed_y = seed / map_width;
    uint32_t seed_x = seed % map_width;
    uint32_t seed_row = seed_y * map_width;
    uint32_t left = seed_x;
    uint32_t right = seed_x;
    if (bitmap[seed] != 0u || visited[seed] != 0u) return;
    while (left > 0u && bitmap[seed_row + left - 1u] == 0u &&
           visited[seed_row + left - 1u] == 0u) {
        --left;
    }
    while (right + 1u < map_width && bitmap[seed_row + right + 1u] == 0u &&
           visited[seed_row + right + 1u] == 0u) {
        ++right;
    }
    memset(visited + seed_row + left, 1u, right - left + 1u);
    stack[top++] = seed_y;
    stack[top++] = left;
    stack[top++] = right;
    while (top > 0u) {
        uint32_t run_right = stack[--top];
        uint32_t run_left = stack[--top];
        uint32_t y = stack[--top];
        uint32_t row = y * map_width;
        int32_t direction;
        if (contour != NULL) {
            if (run_left > 0u && bitmap[row + run_left - 1u] != 0u) {
                contour_include(contour, y, run_left, run_left);
            }
            if (run_right + 1u < map_width && bitmap[row + run_right + 1u] != 0u) {
                contour_include(contour, y, run_right, run_right);
            }
        }
        for (direction = -1; direction <= 1; direction += 2) {
            int64_t neighbor_y = (int64_t)y + direction;
            uint32_t nrow;
            uint32_t x;
            if (neighbor_y < 0 || neighbor_y >= (int64_t)map_height) continue;
            nrow = (uint32_t)neighbor_y * map_width;
            x = run_left;
            while (x <= run_right) {
                if (bitmap[nrow + x] != 0u) {
                    if (contour != NULL) {
                        contour_include(contour, y, x, x);
                    }
                    ++x;
                } else if (visited[nrow + x] == 0u) {
                    uint32_t next_left = x;
                    uint32_t next_right = x;
                    while (next_left > 0u && bitmap[nrow + next_left - 1u] == 0u &&
                           visited[nrow + next_left - 1u] == 0u) {
                        --next_left;
                    }
                    while (next_right + 1u < map_width &&
                           bitmap[nrow + next_right + 1u] == 0u &&
                           visited[nrow + next_right + 1u] == 0u) {
                        ++next_right;
                    }
                    memset(visited + nrow + next_left, 1u, next_right - next_left + 1u);
                    stack[top++] = (uint32_t)neighbor_y;
                    stack[top++] = next_left;
                    stack[top++] = next_right;
                    x = next_right + 1u;
                } else {
                    ++x;
                }
            }
        }
    }
}

/* Reference TryBuildDetection: min-area rect of the contour -> PolygonScore
 * on the mini-box -> Clipper-style round-join unclip -> min-area rect of the
 * expanded polygon -> round-half-even remap to source coordinates -> min-area
 * rect of the mapped corners (PaddleX CropByPolys parity). */
static int try_build_detection(const float* prediction, uint32_t map_width, uint32_t map_height,
                               float box_threshold, float unclip_ratio,
                               uint32_t source_width, uint32_t source_height,
                               det_point* boundary, uint32_t boundary_count,
                               det_point* hull, lw_db_postprocess_workspace* workspace,
                               lw_pipeline_component_profile* profile,
                               lw_detection_box* out_box) {
    det_rectangle rectangle;
    det_point mini_box[4];
    det_point corners[4];
    det_point mapped[4];
    float shortest_side;
    float score;
    double area_value;
    double perimeter;
    double distance_value;
    uint32_t unclipped_count;
    uint32_t i;
    if (!minimum_rectangle(hull, convex_hull(boundary, boundary_count, hull), &rectangle)) {
        return 0;
    }
    shortest_side = rectangle.max_u - rectangle.min_u < rectangle.max_v - rectangle.min_v
                        ? rectangle.max_u - rectangle.min_u
                        : rectangle.max_v - rectangle.min_v;
    if (shortest_side < 3.0f) return 0;
    rectangle_points(&rectangle, 0.0f, mini_box);
    order_clockwise(mini_box);
    if (workspace_grow_polygon(workspace, 4u) != LW_STATUS_OK) return 0;
    score = polygon_score(prediction, map_width, map_height, mini_box, 4u, workspace);
    if (!isfinite(score) || score < box_threshold) return 0;
    area_value = fabs(signed_area(mini_box, 4u));
    perimeter = 0.0;
    for (i = 0u; i < 4u; ++i) {
        float dx = mini_box[(i + 1u) % 4u].x - mini_box[i].x;
        float dy = mini_box[(i + 1u) % 4u].y - mini_box[i].y;
        perimeter += sqrtf(dx * dx + dy * dy);
    }
    distance_value = area_value * (double)unclip_ratio / perimeter;
    if (!isfinite(distance_value) || distance_value <= 0.0) return 0;
    {
        uint64_t unclip_started = lw_pipeline_profile_now(profile);
        unclipped_count = offset_round(mini_box, 4u, distance_value, workspace);
        lw_pipeline_profile_add_elapsed(profile == NULL ? NULL : &profile->unclip_nanoseconds,
                                        unclip_started, profile);
    }
    /* The grow(4) capacity already covers the round-join output bound
     * (4 corners x 228 points); do not grow again, the buffer must stay in
     * place because convex_hull sorts its input in place. */
    {
        det_point* unclipped = (det_point*)workspace->unclip_result;
        if (!minimum_rectangle(hull, convex_hull(unclipped, unclipped_count, hull),
                               &rectangle)) {
            return 0;
        }
    }
    shortest_side = rectangle.max_u - rectangle.min_u < rectangle.max_v - rectangle.min_v
                        ? rectangle.max_u - rectangle.min_u
                        : rectangle.max_v - rectangle.min_v;
    rectangle_points(&rectangle, 0.0f, corners);
    order_clockwise(corners);
    /* PaddleOCR's quad post-process maps coordinates back to the source with
     * np.round (half-even) and clips to [0, size]. */
    for (i = 0u; i < 4u; ++i) {
        corners[i].x = clamp_coord(
            round_half_even_f(corners[i].x / (float)map_width * (float)source_width),
            (float)source_width);
        corners[i].y = clamp_coord(
            round_half_even_f(corners[i].y / (float)map_height * (float)source_height),
            (float)source_height);
    }
    /* PaddleX CropByPolys converts the mapped polygon to int32 and calls
     * minAreaRect before perspective cropping; reproduce that geometry. */
    memcpy(mapped, corners, sizeof(mapped));
    if (!minimum_rectangle(hull, convex_hull(mapped, 4u, hull), &rectangle)) return 0;
    rectangle_points(&rectangle, 0.0f, corners);
    order_clockwise(corners);
    if (hypotf(corners[0].x - corners[1].x, corners[0].y - corners[1].y) <= 4.0f ||
        hypotf(corners[0].x - corners[3].x, corners[0].y - corners[3].y) <= 4.0f) {
        return 0;
    }
    out_box->x1 = corners[0].x;
    out_box->y1 = corners[0].y;
    out_box->x2 = corners[1].x;
    out_box->y2 = corners[1].y;
    out_box->x3 = corners[2].x;
    out_box->y3 = corners[2].y;
    out_box->x4 = corners[3].x;
    out_box->y4 = corners[3].y;
    out_box->score = score;
    return 1;
}

static lw_status finish_contour(det_contour* contour, const float* prediction,
                                uint32_t map_width, uint32_t map_height,
                                float box_threshold, float unclip_ratio,
                                uint32_t source_width, uint32_t source_height,
                                lw_db_postprocess_workspace* workspace,
                                lw_pipeline_component_profile* profile,
                                uint32_t* result_count) {
    det_point* points = (det_point*)workspace->points;
    uint32_t count = contour_collect(contour, points);
    lw_status status;
    lw_detection_box* results = (lw_detection_box*)workspace->results;
    if (count < 3u) return LW_STATUS_OK;
    /* Reuse the hull for unclip, whose round-join buffer can hold this many points. */
    status = workspace_grow_hull(workspace, count > DET_UNCLIP_QUAD_CAPACITY
                                               ? count : DET_UNCLIP_QUAD_CAPACITY);
    if (status != LW_STATUS_OK) return status;
    if (try_build_detection(prediction, map_width, map_height, box_threshold, unclip_ratio,
                            source_width, source_height, points, count,
                            (det_point*)workspace->hull, workspace, profile, &results[*result_count])) {
        ++*result_count;
    }
    return LW_STATUS_OK;
}

lw_status lw_db_postprocess_f32_ws(
    const float* prediction, uint32_t map_width, uint32_t map_height,
    float bitmap_threshold, float box_threshold, float unclip_ratio,
    uint32_t use_dilation, uint32_t max_candidates, uint32_t source_width,
    uint32_t source_height, float width_ratio, float height_ratio,
    lw_detection_box* boxes, uint32_t box_capacity, uint32_t* box_count,
    lw_db_postprocess_workspace* workspace, lw_pipeline_component_profile* profile) {
    uint64_t pixel_count;
    uint8_t* bitmap;
    uint8_t* visited;
    uint32_t* stack;
    det_contour contour;
    lw_detection_box* results;
    uint32_t result_count = 0u;
    uint32_t candidate_count = 0u;
    uint32_t start;
    lw_status status;
    if (box_count != NULL)
        *box_count = 0u;
    if (prediction == NULL || box_count == NULL || map_width == 0u || map_height == 0u ||
        source_width == 0u || source_height == 0u || max_candidates == 0u || use_dilation > 1u ||
        !isfinite(bitmap_threshold) || bitmap_threshold < 0.0f || bitmap_threshold > 1.0f ||
        !isfinite(box_threshold) || box_threshold < 0.0f || box_threshold > 1.0f ||
        !isfinite(unclip_ratio) || unclip_ratio <= 0.0f || !isfinite(width_ratio) ||
        width_ratio <= 0.0f || !isfinite(height_ratio) || height_ratio <= 0.0f ||
        (boxes == NULL && box_capacity != 0u) || workspace == NULL) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    if (map_width > INT32_MAX || map_height > INT32_MAX) {
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    pixel_count = (uint64_t)map_width * map_height;
    if (pixel_count > UINT32_MAX ||
        pixel_count > SIZE_MAX / (3u * sizeof(*stack)) ||
        !allocation_fits(map_height, 2u * sizeof(det_point)) ||
        !allocation_fits(map_height, sizeof(det_contour_row)) ||
        !allocation_fits(max_candidates, sizeof(*results))) {
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    status = workspace_grow(workspace, (uint32_t)pixel_count, map_height, max_candidates);
    if (status != LW_STATUS_OK)
        return status;
    bitmap = workspace->bitmap;
    visited = workspace->visited;
    stack = workspace->queue;
    contour = (det_contour){(det_contour_row*)workspace->contour_rows, UINT32_MAX, 0u};
    results = (lw_detection_box*)workspace->results;
    memset(visited, 0, (size_t)pixel_count);
    /* Convert model probabilities to a compact bitmap after rejecting NaN/Inf. */
#if defined(__EMSCRIPTEN__) && defined(LW_WASM_DB_BITMAP_SIMD)
    if (lw_wasm128_threshold_bitmap_f32(prediction, bitmap, pixel_count,
                                        bitmap_threshold) != 0) {
#else
    if (lw_avx2_threshold_bitmap_f32(prediction, bitmap, pixel_count,
                                     bitmap_threshold) != 0) {
#endif
        return LW_STATUS_INVALID_ARGUMENT;
    }
    if (use_dilation != 0u) {
        /* OpenCV's default anchor for a 2x2 kernel is (0, 0): the source
         * footprint for dst(x,y) is (x,y), (x+1,y), (x,y+1), (x+1,y+1). */
        uint8_t* dilated = workspace->dilated;
        memset(dilated, 0, (size_t)pixel_count);
        for (start = 0u; start < pixel_count; ++start) {
            if (bitmap[(size_t)start] != 0u) {
                uint32_t x = start % map_width;
                uint32_t y = start / map_width;
                dilated[(size_t)start] = 1u;
                if (x + 1u < map_width)
                    dilated[(size_t)(start + 1u)] = 1u;
                if (y + 1u < map_height)
                    dilated[(size_t)(start + map_width)] = 1u;
                if (x + 1u < map_width && y + 1u < map_height)
                    dilated[(size_t)(start + map_width + 1u)] = 1u;
            }
        }
        bitmap = dilated;
    }
    /* Flood-fill each 8-connected foreground component. */
    for (start = 0u; start < pixel_count && candidate_count < max_candidates; ++start) {
        if (bitmap[(size_t)start] == 0u || visited[(size_t)start] != 0u)
            continue;
        ++candidate_count;
        fill_foreground(bitmap, visited, map_width, map_height, start, &contour, stack);
        status = finish_contour(&contour, prediction, map_width, map_height, box_threshold,
                                unclip_ratio, source_width, source_height, workspace, profile, &result_count);
        if (status != LW_STATUS_OK) return status;
    }
    /* OpenCV RETR_LIST also returns contours for enclosed background regions:
     * flood the exterior background from the image border, then trace the
     * remaining zero components as holes with 4-connectivity (the complement
     * of the 8-connected foreground contours). */
    memset(workspace->background_visited, 0, (size_t)pixel_count);
    for (start = 0u; start < map_width; ++start) {
        (void)fill_background(bitmap, workspace->background_visited, map_width, map_height,
                              start, NULL, stack);
        if (map_height > 1u) {
            (void)fill_background(bitmap, workspace->background_visited, map_width, map_height,
                                  (map_height - 1u) * map_width + start, NULL, stack);
        }
    }
    for (start = 1u; start + 1u < map_height; ++start) {
        (void)fill_background(bitmap, workspace->background_visited, map_width, map_height,
                              start * map_width, NULL, stack);
        if (map_width > 1u) {
            (void)fill_background(bitmap, workspace->background_visited, map_width, map_height,
                                  start * map_width + map_width - 1u, NULL, stack);
        }
    }
    for (start = 0u; start < pixel_count && candidate_count < max_candidates; ++start) {
        if (bitmap[(size_t)start] != 0u || workspace->background_visited[(size_t)start] != 0u) {
            continue;
        }
        ++candidate_count;
        fill_background(bitmap, workspace->background_visited, map_width, map_height, start, &contour, stack);
        status = finish_contour(&contour, prediction, map_width, map_height, box_threshold,
                                unclip_ratio, source_width, source_height, workspace, profile, &result_count);
        if (status != LW_STATUS_OK) return status;
    }
    *box_count = result_count;
    if (boxes != NULL) {
        if (box_capacity < result_count) {
            return LW_STATUS_OUT_OF_BOUNDS;
        }
        memcpy(boxes, results, (size_t)result_count * sizeof(*boxes));
    }
    return LW_STATUS_OK;
}

lw_status lw_db_postprocess_f32(const float* prediction, uint32_t map_width, uint32_t map_height,
                                float bitmap_threshold, float box_threshold, float unclip_ratio,
                                uint32_t use_dilation, uint32_t max_candidates,
                                uint32_t source_width, uint32_t source_height, float width_ratio,
                                float height_ratio, lw_detection_box* boxes, uint32_t box_capacity,
                                uint32_t* box_count) {
    lw_db_postprocess_workspace workspace;
    lw_status status;
    memset(&workspace, 0, sizeof(workspace));
    status = lw_db_postprocess_f32_ws(prediction, map_width, map_height, bitmap_threshold,
                                      box_threshold, unclip_ratio, use_dilation, max_candidates,
                                      source_width, source_height, width_ratio, height_ratio, boxes,
                                      box_capacity, box_count, &workspace, NULL);
    lw_db_postprocess_workspace_free(&workspace);
    return status;
}
