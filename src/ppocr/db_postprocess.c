#include "db_geometry_internal.h"
#include "../simd/simd_kernels.h"

/*
 * Scan the thresholded DB probability map and pass foreground contours and
 * enclosed holes to the candidate fitter. The scan order defines which
 * contours count toward the candidate limit.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct det_contour_row {
    uint32_t left;
    uint32_t end; /* Exclusive right edge. Zero denotes an untouched row. */
};

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

/* Grow the reusable scratch buffers to cover this image. Buffers only grow;
 * a detector that processes many same-size images allocates once. */
static lw_status workspace_grow(lw_db_postprocess_workspace* workspace, uint32_t pixels,
                                uint32_t map_height, uint32_t max_candidates) {
    uint8_t* bitmap = workspace->bitmap;
    uint8_t* visited = workspace->visited;
    uint8_t* background_visited = workspace->background_visited;
    uint8_t* dilated = workspace->dilated;
    uint32_t* stack = workspace->queue;
    lw_detection_box* results = workspace->results;
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

static lw_status finish_contour(det_contour* contour, const float* prediction,
                                uint32_t map_width, uint32_t map_height,
                                float box_threshold, float unclip_ratio,
                                uint32_t source_width, uint32_t source_height,
                                lw_db_postprocess_workspace* workspace,
                                lw_pipeline_component_profile* profile,
                                uint32_t* result_count) {
    uint32_t point_count = contour_collect(contour, workspace->points);
    if (point_count < 3u) return LW_STATUS_OK;
    return lw_db_fit_candidate(point_count, prediction, map_width, map_height, box_threshold,
                               unclip_ratio, source_width, source_height, workspace, profile,
                               result_count);
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
        !lw_db_allocation_fits(map_height, 2u * sizeof(det_point)) ||
        !lw_db_allocation_fits(map_height, sizeof(det_contour_row)) ||
        !lw_db_allocation_fits(max_candidates, sizeof(*results))) {
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    status = workspace_grow(workspace, (uint32_t)pixel_count, map_height, max_candidates);
    if (status != LW_STATUS_OK)
        return status;
    bitmap = workspace->bitmap;
    visited = workspace->visited;
    stack = workspace->queue;
    contour = (det_contour){workspace->contour_rows, UINT32_MAX, 0u};
    results = workspace->results;
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
                                unclip_ratio, source_width, source_height, workspace, profile,
                                &result_count);
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
                                unclip_ratio, source_width, source_height, workspace, profile,
                                &result_count);
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
