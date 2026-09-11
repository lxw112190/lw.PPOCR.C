#include "session_internal.h"

#include "error_internal.h"
#include "lwm_read.h"
#include "packed_conv_internal.h"
#include "../simd/simd_kernels.h"

#include <stdint.h>
#include <stdlib.h>

static lw_packed_conv1x1_kernel_fn select_conv1x1_kernel(const lw_session* session,
                                                          uint16_t* kernel_id) {
    if (lw_simd_level_is_avx2(session->cpu.simd)) {
        *kernel_id = LW_CONV1X1_KERNEL_AVX2;
        return lw_avx2_packed_conv1x1_f32;
    }
    if (lw_simd_level_is_neon(session->cpu.simd)) {
        *kernel_id = LW_CONV1X1_KERNEL_NEON;
        return lw_neon_packed_conv1x1_f32;
    }
    if (lw_simd_level_is_lsx(session->cpu.simd)) {
        *kernel_id = LW_CONV1X1_KERNEL_LSX;
        return lw_lsx_packed_conv1x1_f32;
    }
    if (lw_simd_level_is_sse2(session->cpu.simd)) {
        *kernel_id = LW_CONV1X1_KERNEL_SSE2;
        return lw_sse2_packed_conv1x1_f32;
    }
    *kernel_id = LW_CONV1X1_KERNEL_SCALAR;
    return lw_scalar_packed_conv1x1_f32;
}

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
        bound->execution_kind = LW_BOUND_EXEC_GENERIC;
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
            if (bound->operator_type == 1u &&
                bound->implementation == LW_PREPARED_CONSTANT_CONV1X1_PACKED4 &&
                input_count >= 2u) {
                bound->execution_kind = LW_BOUND_EXEC_CONV1X1_PACKED;
                bound->data.conv1x1.kernel = select_conv1x1_kernel(
                    session, &bound->data.conv1x1.kernel_id);
                bound->data.conv1x1.packed_weights =
                    (const float*)(const void*)(session->packed_weights +
                        (size_t)session->prepared_constants[node_index].packed_weight_offset);
                bound->data.conv1x1.input_index = bound->input_indices[0];
                bound->data.conv1x1.bias_index =
                    input_count == 3u ? bound->input_indices[2] : UINT32_MAX;
                bound->data.conv1x1.output_index = bound->output_index;
                bound->data.conv1x1.output_tile = LW_PACKED_CONV1X1_OUTPUT_TILE;
            }
        }
    }
#else
    (void)node_index;
#endif
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}
