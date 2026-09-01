#include "reading_order.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct reading_item {
    lw_detection_box box;
    uint32_t original_index;
    uint32_t column;
    float min_x;
    float max_x;
    float center_x;
    float min_y;
    float center_y;
    float width;
} reading_item;

typedef struct reading_column {
    float min_x;
    float max_x;
    float center_x;
    float width;
    uint32_t count;
} reading_column;

int lw_reading_order_is_valid(uint32_t reading_order) {
    return reading_order <= (uint32_t)LW_READING_ORDER_VERTICAL_LTR;
}

static void legacy_horizontal_sort(lw_detection_box* boxes, uint32_t count) {
    uint32_t index;
    for (index = 1u; index < count; ++index) {
        lw_detection_box value = boxes[index];
        uint32_t position = index;
        while (position > 0u) {
            lw_detection_box* previous = &boxes[position - 1u];
            float delta_y = previous->y1 - value.y1;
            int comes_after = fabsf(delta_y) < 10.0f ? previous->x1 > value.x1
                                                     : previous->y1 > value.y1;
            if (!comes_after)
                break;
            boxes[position] = *previous;
            --position;
        }
        boxes[position] = value;
    }
}

static void item_geometry(reading_item* item) {
    const lw_detection_box* box = &item->box;
    item->min_x = item->max_x = box->x1;
    item->min_y = box->y1;
    if (box->x2 < item->min_x) item->min_x = box->x2;
    if (box->x3 < item->min_x) item->min_x = box->x3;
    if (box->x4 < item->min_x) item->min_x = box->x4;
    if (box->x2 > item->max_x) item->max_x = box->x2;
    if (box->x3 > item->max_x) item->max_x = box->x3;
    if (box->x4 > item->max_x) item->max_x = box->x4;
    item->min_y = box->y1;
    if (box->y2 < item->min_y) item->min_y = box->y2;
    if (box->y3 < item->min_y) item->min_y = box->y3;
    if (box->y4 < item->min_y) item->min_y = box->y4;
    item->center_x = (item->min_x + item->max_x) * 0.5f;
    item->center_y = (box->y1 + box->y2 + box->y3 + box->y4) * 0.25f;
    item->width = item->max_x - item->min_x;
    if (!(item->width > 0.0f) || !isfinite(item->width)) item->width = 1.0f;
}

static int item_x_after(const reading_item* left, const reading_item* right) {
    if (left->center_x > right->center_x) return 1;
    if (left->center_x < right->center_x) return 0;
    return left->original_index > right->original_index;
}

static float interval_overlap(const reading_item* item, const reading_column* column) {
    float overlap = fminf(item->max_x, column->max_x) - fmaxf(item->min_x, column->min_x);
    float denominator = fmaxf(item->width, column->width);
    return overlap > 0.0f && denominator > 0.0f ? overlap / denominator : 0.0f;
}

static int item_before(const reading_item* left, const reading_item* right,
                       const reading_column* columns, uint32_t reverse_columns) {
    const reading_column* left_column = &columns[left->column];
    const reading_column* right_column = &columns[right->column];
    if (left->column != right->column) {
        if (left_column->center_x != right_column->center_x)
            return reverse_columns ? left_column->center_x > right_column->center_x
                                   : left_column->center_x < right_column->center_x;
        return left->column < right->column;
    }
    if (left->min_y != right->min_y) return left->min_y < right->min_y;
    if (left->center_y != right->center_y) return left->center_y < right->center_y;
    return left->original_index < right->original_index;
}

static lw_status vertical_sort(lw_detection_box* boxes, uint32_t count,
                               uint32_t reading_order) {
    reading_item* items;
    reading_column* columns;
    uint32_t column_count = 0u;
    uint32_t index;
    if (count < 2u) return LW_STATUS_OK;
    if ((size_t)count > SIZE_MAX / sizeof(*items) ||
        (size_t)count > SIZE_MAX / sizeof(*columns))
        return LW_STATUS_OUT_OF_BOUNDS;
    items = (reading_item*)malloc((size_t)count * sizeof(*items));
    columns = (reading_column*)calloc(count, sizeof(*columns));
    if (items == NULL || columns == NULL) {
        free(columns);
        free(items);
        return LW_STATUS_OUT_OF_MEMORY;
    }
    for (index = 0u; index < count; ++index) {
        items[index].box = boxes[index];
        items[index].original_index = index;
        item_geometry(&items[index]);
    }
    for (index = 1u; index < count; ++index) {
        reading_item value = items[index];
        uint32_t position = index;
        while (position > 0u && item_x_after(&items[position - 1u], &value)) {
            items[position] = items[position - 1u];
            --position;
        }
        items[position] = value;
    }
    for (index = 0u; index < count; ++index) {
        uint32_t column;
        uint32_t best = UINT32_MAX;
        float best_overlap = 0.0f;
        float best_distance = INFINITY;
        for (column = 0u; column < column_count; ++column) {
            float overlap = interval_overlap(&items[index], &columns[column]);
            float distance = fabsf(items[index].center_x - columns[column].center_x);
            if (overlap >= 0.45f &&
                (overlap > best_overlap ||
                 (overlap == best_overlap && distance < best_distance))) {
                best = column;
                best_overlap = overlap;
                best_distance = distance;
            }
        }
        if (best == UINT32_MAX) {
            best = column_count++;
            columns[best].min_x = items[index].min_x;
            columns[best].max_x = items[index].max_x;
            columns[best].center_x = items[index].center_x;
            columns[best].width = items[index].width;
        }
        items[index].column = best;
        columns[best].min_x = fminf(columns[best].min_x, items[index].min_x);
        columns[best].max_x = fmaxf(columns[best].max_x, items[index].max_x);
        columns[best].count += 1u;
        columns[best].center_x =
            (columns[best].center_x * (float)(columns[best].count - 1u) +
             items[index].center_x) / (float)columns[best].count;
        columns[best].width = fmaxf(columns[best].width, items[index].width);
    }
    for (index = 1u; index < count; ++index) {
        reading_item value = items[index];
        uint32_t position = index;
        while (position > 0u &&
               item_before(&value, &items[position - 1u], columns,
                           reading_order == LW_READING_ORDER_VERTICAL_RTL)) {
            items[position] = items[position - 1u];
            --position;
        }
        items[position] = value;
    }
    for (index = 0u; index < count; ++index) boxes[index] = items[index].box;
    free(columns);
    free(items);
    return LW_STATUS_OK;
}

lw_status lw_sort_detection_boxes(lw_detection_box* boxes, uint32_t count,
                                  uint32_t reading_order) {
    if ((boxes == NULL && count != 0u) || !lw_reading_order_is_valid(reading_order))
        return LW_STATUS_INVALID_ARGUMENT;
    if (count < 2u) return LW_STATUS_OK;
    if (reading_order == LW_READING_ORDER_HORIZONTAL_LTR) {
        legacy_horizontal_sort(boxes, count);
        return LW_STATUS_OK;
    }
    return vertical_sort(boxes, count, reading_order);
}
