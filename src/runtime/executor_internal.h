#ifndef LW_EXECUTOR_INTERNAL_H
#define LW_EXECUTOR_INTERNAL_H

#include "lw_infer.h"

#include <stdint.h>

#define LW_EXECUTION_PROFILE_OPERATOR_CAPACITY 16u

typedef uint64_t (*lw_execution_profile_clock)(void* context);

typedef struct lw_execution_profile {
    uint32_t struct_size;
    uint32_t reserved;
    lw_execution_profile_clock clock;
    void* clock_context;
    uint64_t operator_nanoseconds[LW_EXECUTION_PROFILE_OPERATOR_CAPACITY];
    uint64_t operator_invocations[LW_EXECUTION_PROFILE_OPERATOR_CAPACITY];
} lw_execution_profile;

lw_status lw_execute_session_f32(
    lw_session* session,
    const float* input,
    uint64_t input_element_count,
    float* output,
    uint64_t output_element_count,
    lw_error* error);

lw_status lw_execute_session_f32_profiled(
    lw_session* session,
    const float* input,
    uint64_t input_element_count,
    float* output,
    uint64_t output_element_count,
    lw_execution_profile* profile,
    lw_error* error);

#endif
