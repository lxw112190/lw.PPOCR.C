#ifndef LW_SESSION_INTERNAL_H
#define LW_SESSION_INTERNAL_H

/* Runtime tensor metadata shared by shape inference, planning and execution. */

#include "model_internal.h"

#include <stddef.h>
#include <stdint.h>

#define LW_WORKSPACE_ALIGNMENT 64u

typedef struct lw_runtime_tensor {
    uint32_t dtype;
    uint32_t rank;
    int32_t dimensions[LW_MAX_DIMS];
    uint32_t flags;
    uint64_t byte_size;
    uint64_t workspace_offset;
    int32_t birth_node;
    int32_t last_use_node;
    int workspace_live;
} lw_runtime_tensor;

struct lw_session {
    const lw_model* model;
    lw_runtime_tensor* tensors;
    uint8_t* workspace;
    size_t workspace_bytes;
    lw_session_info info;
};

lw_status lw_resolve_shapes(lw_session* session, uint64_t max_tensor_size, lw_error* error);
lw_status lw_plan_workspace(lw_session* session, uint64_t max_workspace_size, lw_error* error);

#endif
