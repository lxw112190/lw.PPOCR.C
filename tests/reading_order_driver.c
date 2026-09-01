#include "reading_order.h"

#include <stdio.h>
#include <string.h>

static lw_detection_box box(float x, float y, float score) {
    lw_detection_box value = {x, y, x + 8.0f, y, x + 8.0f, y + 12.0f, x, y + 12.0f,
                              score, 0u};
    return value;
}

static int check_order(const lw_detection_box* values, const float* expected, unsigned count) {
    unsigned index;
    for (index = 0u; index < count; ++index) {
        if (values[index].score != expected[index])
            return 0;
    }
    return 1;
}

int main(void) {
    const float rtl[] = {2.0f, 4.0f, 1.0f, 3.0f};
    const float ltr[] = {1.0f, 3.0f, 2.0f, 4.0f};
    const float horizontal[] = {1.0f, 2.0f, 3.0f, 4.0f};
    lw_detection_box values[4] = {box(20.0f, 100.0f, 1.0f), box(120.0f, 150.0f, 4.0f),
                                  box(120.0f, 100.0f, 2.0f), box(20.0f, 150.0f, 3.0f)};
    lw_detection_box original[4];
    memcpy(original, values, sizeof(values));
    if (lw_sort_detection_boxes(values, 4u, LW_READING_ORDER_VERTICAL_RTL) != LW_STATUS_OK ||
        !check_order(values, rtl, 4u))
        return 1;
    memcpy(values, original, sizeof(values));
    if (lw_sort_detection_boxes(values, 4u, LW_READING_ORDER_VERTICAL_LTR) != LW_STATUS_OK ||
        !check_order(values, ltr, 4u))
        return 1;
    memcpy(values, original, sizeof(values));
    if (lw_sort_detection_boxes(values, 4u, LW_READING_ORDER_HORIZONTAL_LTR) != LW_STATUS_OK ||
        !check_order(values, horizontal, 4u))
        return 1;
    memcpy(values, original, sizeof(values));
    if (lw_sort_detection_boxes(values, 4u, 99u) != LW_STATUS_INVALID_ARGUMENT ||
        memcmp(values, original, sizeof(values)) != 0)
        return 1;
    puts("reading order sorting: ok");
    return 0;
}
