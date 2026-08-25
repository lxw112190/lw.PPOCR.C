#ifndef LW_MODEL_INTERNAL_H
#define LW_MODEL_INTERNAL_H

#include "error_internal.h"
#include "lw_infer.h"

#include <stddef.h>
#include <stdint.h>

#define LWM_V0_HEADER_SIZE 160u
#define LWM_V0_TENSOR_SIZE 80u
#define LWM_V0_NODE_SIZE 72u
#define LWM_V0_CHECKSUM_OFFSET 128u
#define LWM_V0_MAX_DIMS 8u
#define LWM_V0_MAX_NODE_INPUTS 8u
#define LWM_V0_MAX_NODE_OUTPUTS 4u
#define LWM_V0_NO_WORKSPACE UINT64_MAX
#define LWM_V0_HEADER_FLAG_NO_MEMORY_PLAN 1u
#define LWM_V0_TENSOR_FLAG_CONSTANT 1u
#define LWM_V0_TENSOR_FLAG_INPUT 2u
#define LWM_V0_TENSOR_FLAG_OUTPUT 4u

struct lw_model {
    uint8_t* bytes;
    size_t byte_count;
    lw_model_info info;
    uint64_t input_offset;
    uint64_t output_offset;
    uint64_t tensor_offset;
    uint64_t node_offset;
    uint64_t param_offset;
    uint64_t param_size;
    uint64_t weight_offset;
    uint64_t weight_size;
};

lw_status lw_validate_lwm_v0(lw_model* model, lw_error* error);

#endif
