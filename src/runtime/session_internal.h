#ifndef LW_SESSION_INTERNAL_H
#define LW_SESSION_INTERNAL_H

/* Runtime tensor metadata shared by shape inference, planning and execution. */

#include "model_internal.h"
#include "parallel_internal.h"
#include "../simd/cpu_features.h"

#include <stddef.h>
#include <stdint.h>

#define LW_WORKSPACE_ALIGNMENT 64u

typedef enum lw_prepared_constant_kind {
    LW_PREPARED_CONSTANT_NONE = 0,
    LW_PREPARED_CONSTANT_CONV1X1_PACKED4 = 1,
    LW_PREPARED_CONSTANT_MATMUL_PACKED16 = 2,
    LW_PREPARED_CONSTANT_CONV3X3_STRIDE2_PACKED8 = 3
} lw_prepared_constant_kind;

typedef struct lw_prepared_constant {
    uint32_t kind;
    uint32_t reserved;
    uint64_t packed_weight_offset;
    uint64_t packed_weight_count;
} lw_prepared_constant;

typedef struct lw_shared_prepared_constants {
    uint32_t ref_count;
    lw_prepared_constant* constants;
    uint8_t* packed_weights;
    size_t packed_weight_bytes;
} lw_shared_prepared_constants;
typedef struct lw_bound_node {
    uint32_t node_index;
    uint16_t operator_type;
    uint16_t input_count;
    uint32_t input_indices[LWM_V0_MAX_NODE_INPUTS];
    uint32_t output_index;
    uint32_t implementation;
    const lw_prepared_constant* prepared_constant;
} lw_bound_node;

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
    lw_cpu_capabilities cpu;
    lw_runtime_tensor* tensors;
    uint8_t* workspace;
    size_t workspace_bytes;
    lw_prepared_constant* prepared_constants;
    uint8_t* packed_weights;
    size_t packed_weight_bytes;
    lw_shared_prepared_constants* shared_prepared_constants;
    lw_bound_node* execution_nodes;
    uint32_t execution_node_count;
    lw_thread_pool* thread_pool;
    uint32_t intra_op_thread_count;
    lw_session_info info;
};

lw_status lw_resolve_shapes(lw_session* session, uint64_t max_tensor_size, lw_error* error);
lw_status lw_plan_workspace(lw_session* session, uint64_t max_workspace_size, lw_error* error);
void lw_session_set_intra_op_thread_count(lw_session* session, uint32_t thread_count);
lw_status lw_session_share_prepared_constants(lw_session* destination,
                                               const lw_session* source,
                                               lw_error* error);
lw_status lw_prepare_execution_nodes(lw_session* session, lw_error* error);
void lw_free_execution_nodes(lw_session* session);

#endif
