#include "session_internal.h"

#include "error_internal.h"
#include "lwm_read.h"

#include <stdint.h>
#include <stdlib.h>

void lw_free_execution_nodes(lw_session* session) {
    if (session == NULL) {
        return;
    }
    free(session->execution_nodes);
    session->execution_nodes = NULL;
    session->execution_node_count = 0u;
}

lw_status lw_prepare_execution_nodes(lw_session* session, lw_error* error) {
    uint32_t node_index;

    if (session == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "session is required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    lw_free_execution_nodes(session);
#if defined(LW_EXPERIMENTAL_PREPARED_EXECUTION)
    if (session->model->info.node_count == 0u) {
        lw_set_error(error, LW_STATUS_OK, "");
        return LW_STATUS_OK;
    }
    if ((size_t)session->model->info.node_count >
        SIZE_MAX / sizeof(*session->execution_nodes)) {
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS,
                     "execution node table size overflows");
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    session->execution_nodes = (lw_bound_node*)calloc(
        session->model->info.node_count, sizeof(*session->execution_nodes));
    if (session->execution_nodes == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY,
                     "unable to allocate execution node table");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    session->execution_node_count = session->model->info.node_count;
    for (node_index = 0u; node_index < session->execution_node_count; ++node_index) {
        const uint8_t* node = session->model->bytes +
            (size_t)session->model->node_offset +
            (size_t)node_index * LWM_V0_NODE_SIZE;
        lw_bound_node* bound = &session->execution_nodes[node_index];
        uint16_t input_count = lwm_read_u16(node + 2u);
        uint32_t input_index;

        if (input_count > LWM_V0_MAX_NODE_INPUTS) {
            lw_free_execution_nodes(session);
            lw_set_error(error, LW_STATUS_INVALID_FORMAT,
                         "execution node input count exceeds the LWM limit");
            return LW_STATUS_INVALID_FORMAT;
        }
        bound->node_index = node_index;
        bound->operator_type = lwm_read_u16(node);
        bound->input_count = input_count;
        bound->output_index = lwm_read_u32(node + 40u);
        for (input_index = 0u; input_index < input_count; ++input_index) {
            bound->input_indices[input_index] = lwm_read_u32(
                node + 8u + (size_t)input_index * sizeof(uint32_t));
        }
        if (session->prepared_constants != NULL &&
            session->prepared_constants[node_index].kind != LW_PREPARED_CONSTANT_NONE) {
            bound->implementation = session->prepared_constants[node_index].kind;
            bound->prepared_constant = &session->prepared_constants[node_index];
        }
    }
#else
    (void)node_index;
#endif
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}
