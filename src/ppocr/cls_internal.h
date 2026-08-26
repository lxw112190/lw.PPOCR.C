#ifndef LW_CLS_INTERNAL_H
#define LW_CLS_INTERNAL_H

/* Private classifier preprocessing contract. */

#include "lw_infer.h"

#include <stdint.h>

#define LW_CLS_INPUT_HEIGHT 80u
#define LW_CLS_INPUT_WIDTH 160u
#define LW_CLS_CLASS_COUNT 2u

lw_status lw_cls_preprocess_bgr_u8(const uint8_t* source, uint64_t source_byte_count,
                                   uint32_t source_width, uint32_t source_height,
                                   uint32_t source_stride, float* output,
                                   uint64_t output_element_count, uint32_t* resized_width);

#endif
