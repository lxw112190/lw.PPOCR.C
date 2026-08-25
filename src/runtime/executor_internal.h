#ifndef LW_EXECUTOR_INTERNAL_H
#define LW_EXECUTOR_INTERNAL_H

#include "lw_infer.h"

#include <stdint.h>

lw_status lw_execute_session_f32(
    lw_session* session,
    const float* input,
    uint64_t input_element_count,
    float* output,
    uint64_t output_element_count,
    lw_error* error);

#endif
