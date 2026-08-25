#ifndef LW_REC_INTERNAL_H
#define LW_REC_INTERNAL_H

#include "lw_infer.h"

#include <stdint.h>

#define LW_REC_INPUT_HEIGHT 48u

typedef struct lw_rec_dictionary lw_rec_dictionary;

lw_status lw_rec_preprocess_bgr_u8(
    const uint8_t* source,
    uint64_t source_byte_count,
    uint32_t source_width,
    uint32_t source_height,
    uint32_t source_stride,
    uint32_t target_width,
    float* output,
    uint64_t output_element_count,
    uint32_t* resized_width);

lw_status lw_rec_dictionary_load(
    const char* path_utf8,
    lw_rec_dictionary** out_dictionary,
    lw_error* error);
void lw_rec_dictionary_free(lw_rec_dictionary* dictionary);
uint32_t lw_rec_dictionary_class_count(const lw_rec_dictionary* dictionary);

lw_status lw_rec_ctc_decode_f32(
    const lw_rec_dictionary* dictionary,
    const float* probabilities,
    uint64_t probability_element_count,
    uint32_t time_steps,
    uint32_t class_count,
    char* text_utf8,
    uint64_t text_capacity,
    uint64_t* required_capacity,
    float* score,
    uint32_t* emitted_count,
    lw_error* error);

#endif
