#ifndef LW_READING_ORDER_INTERNAL_H
#define LW_READING_ORDER_INTERNAL_H

#include "lw_infer.h"
#include <stdint.h>

int lw_reading_order_is_valid(uint32_t reading_order);
lw_status lw_sort_detection_boxes(lw_detection_box* boxes, uint32_t count, uint32_t reading_order);

#endif
