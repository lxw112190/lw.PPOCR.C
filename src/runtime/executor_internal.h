#ifndef LW_EXECUTOR_INTERNAL_H
#define LW_EXECUTOR_INTERNAL_H

#include "lw_infer.h"

#include <stdint.h>

#define LW_EXECUTION_PROFILE_OPERATOR_CAPACITY 22u
#define LW_EXECUTION_PROFILE_NODE_CAPACITY 256u
#define LW_EXECUTION_PROFILE_CONV_CLASS_CAPACITY 5u
#define LW_EXECUTION_PROFILE_THREAD_HISTOGRAM_CAPACITY 17u

typedef enum lw_execution_profile_conv_class {
    LW_EXECUTION_PROFILE_CONV_1X1 = 0,
    LW_EXECUTION_PROFILE_CONV_3X3 = 1,
    LW_EXECUTION_PROFILE_CONV_DEPTHWISE_3X3 = 2,
    LW_EXECUTION_PROFILE_CONV_STRIDE2_3X3 = 3,
    LW_EXECUTION_PROFILE_CONV_OTHER = 4
} lw_execution_profile_conv_class;

typedef uint64_t (*lw_execution_profile_clock)(void* context);

typedef struct lw_execution_profile {
    uint32_t struct_size;
    uint32_t reserved;
    lw_execution_profile_clock clock;
    void* clock_context;
    uint64_t operator_nanoseconds[LW_EXECUTION_PROFILE_OPERATOR_CAPACITY];
    uint64_t operator_invocations[LW_EXECUTION_PROFILE_OPERATOR_CAPACITY];
    uint64_t node_nanoseconds[LW_EXECUTION_PROFILE_NODE_CAPACITY];
    uint64_t node_invocations[LW_EXECUTION_PROFILE_NODE_CAPACITY];
    uint64_t conv_class_nanoseconds[LW_EXECUTION_PROFILE_CONV_CLASS_CAPACITY];
    uint64_t conv_class_invocations[LW_EXECUTION_PROFILE_CONV_CLASS_CAPACITY];
    /* Index is the actual thread count selected for one Conv invocation.
     * Index zero remains unused so reports can print the policy directly. */
    uint64_t conv_thread_histogram[LW_EXECUTION_PROFILE_THREAD_HISTOGRAM_CAPACITY];
} lw_execution_profile;

lw_status lw_execute_session_f32(lw_session* session, const float* input,
                                 uint64_t input_element_count, float* output,
                                 uint64_t output_element_count, lw_error* error);

lw_status lw_execute_session_f32_profiled(lw_session* session, const float* input,
                                          uint64_t input_element_count, float* output,
                                          uint64_t output_element_count,
                                          lw_execution_profile* profile, lw_error* error);

#endif
