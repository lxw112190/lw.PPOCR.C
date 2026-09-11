#include "lw_infer.h"
#include "lwm_read.h"
#include "session_internal.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

static void make_rec_input(lw_tensor_desc* input, int32_t batch, int32_t width) {
    lw_tensor_desc_init(input);
    input->dtype = LW_DTYPE_F32;
    input->rank = 4u;
    input->dimensions[0] = batch;
    input->dimensions[1] = 3;
    input->dimensions[2] = 48;
    input->dimensions[3] = width;
}

static int check_execution_table(const lw_model* model, const lw_session* session) {
    uint32_t node_index;
#if defined(LW_EXPERIMENTAL_PREPARED_EXECUTION)
    if (session->execution_nodes == NULL ||
        session->execution_node_count != model->info.node_count) {
        return 0;
    }
    for (node_index = 0u; node_index < session->execution_node_count; ++node_index) {
        const uint8_t* disk = model->bytes + (size_t)model->node_offset +
                              (size_t)node_index * LWM_V0_NODE_SIZE;
        const lw_bound_node* bound = &session->execution_nodes[node_index];
        if (bound->node_index != node_index ||
            bound->operator_type != lwm_read_u16(disk) ||
            bound->input_count != lwm_read_u16(disk + 2u) ||
            bound->output_index != lwm_read_u32(disk + 40u)) {
            return 0;
        }
        for (uint32_t input_index = 0u; input_index < bound->input_count; ++input_index) {
            if (bound->input_indices[input_index] !=
                lwm_read_u32(disk + 8u + (size_t)input_index * sizeof(uint32_t))) {
                return 0;
            }
        }
        if (bound->prepared_constant != NULL &&
            bound->prepared_constant != &session->prepared_constants[node_index]) {
            return 0;
        }
    }
#else
    (void)model;
    (void)node_index;
    if (session->execution_nodes != NULL || session->execution_node_count != 0u) {
        return 0;
    }
#endif
    return 1;
}
static int create_and_check(const lw_model* model, int32_t batch, int32_t width,
                            int32_t expected_steps, uint64_t* workspace_size) {
    lw_tensor_desc input;
    lw_tensor_desc output;
    lw_session_info info;
    lw_session* session = NULL;
    lw_error error;
    lw_status status;
    make_rec_input(&input, batch, width);
    lw_error_init(&error);
    status = lw_session_create(model, &input, 1u, NULL, &session, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "session create failed for width %" PRId32 ": %s: %s\n", width,
                lw_status_string(status), error.message);
        return 0;
    }
    if (!check_execution_table(model, session)) {
        fprintf(stderr, "unexpected prepared execution table for width %" PRId32 "\n", width);
        lw_session_free(session);
        return 0;
    }
    lw_session_info_init(&info);
    lw_tensor_desc_init(&output);
    if (lw_session_get_info(session, &info) != LW_STATUS_OK ||
        lw_session_get_output_desc(session, 0u, &output) != LW_STATUS_OK ||
        info.tensor_count != 274u || info.input_count != 1u || info.output_count != 1u ||
        info.workspace_size == 0u || (info.workspace_size & 63u) != 0u ||
        output.dtype != LW_DTYPE_F32 || output.rank != 3u || output.dimensions[0] != batch ||
        output.dimensions[1] != expected_steps || output.dimensions[2] != 6906) {
        fprintf(stderr, "unexpected session plan or output shape for width %" PRId32 "\n", width);
        lw_session_free(session);
        return 0;
    }
    *workspace_size = info.workspace_size;
    lw_session_free(session);
    return 1;
}

int main(int argc, char** argv) {
    lw_model* model = NULL;
    lw_error error;
    lw_status status;
    uint64_t workspace_320 = 0u;
    uint64_t workspace_odd = 0u;
    uint64_t workspace_minimum = 0u;
    uint64_t workspace_640 = 0u;
    lw_tensor_desc input;
    lw_session_options options;
    lw_session* session = NULL;
    uint32_t i;
    if (argc != 2) {
        fprintf(stderr, "expected path to rec.lwm\n");
        return 2;
    }
    lw_error_init(&error);
    status = lw_model_load(argv[1], NULL, &model, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "%s: %s\n", lw_status_string(status), error.message);
        return 1;
    }
    if (!create_and_check(model, 1, 7, 1, &workspace_minimum) ||
        !create_and_check(model, 1, 320, 40, &workspace_320) ||
        !create_and_check(model, 1, 321, 40, &workspace_odd) ||
        !create_and_check(model, 2, 640, 80, &workspace_640) || workspace_odd < workspace_320 ||
        workspace_640 <= workspace_320) {
        lw_model_free(model);
        return 1;
    }

    make_rec_input(&input, 1, 320);
    lw_session_options_init(&options);
    options.max_workspace_size = workspace_320 - 1u;
    lw_error_init(&error);
    status = lw_session_create(model, &input, 1u, &options, &session, &error);
    if (status != LW_STATUS_MEMORY_LIMIT || session != NULL) {
        fprintf(stderr, "workspace memory limit was not enforced\n");
        lw_model_free(model);
        return 1;
    }

    lw_session_options_init(&options);
    options.max_tensor_size = 1024u;
    lw_error_init(&error);
    status = lw_session_create(model, &input, 1u, &options, &session, &error);
    if (status != LW_STATUS_MEMORY_LIMIT || session != NULL) {
        fprintf(stderr, "single tensor memory limit was not enforced\n");
        lw_model_free(model);
        return 1;
    }

    make_rec_input(&input, 1, 320);
    input.dimensions[2] = 47;
    lw_error_init(&error);
    status = lw_session_create(model, &input, 1u, NULL, &session, &error);
    if (status != LW_STATUS_INVALID_SHAPE || session != NULL) {
        fprintf(stderr, "invalid REC input height was not rejected\n");
        lw_model_free(model);
        return 1;
    }

    make_rec_input(&input, 1, 1);
    lw_error_init(&error);
    status = lw_session_create(model, &input, 1u, NULL, &session, &error);
    if (status != LW_STATUS_INVALID_SHAPE || session != NULL) {
        fprintf(stderr, "too-small REC input width was not rejected\n");
        lw_model_free(model);
        return 1;
    }

    make_rec_input(&input, 1, 64);
    for (i = 0u; i < 100u; ++i) {
        lw_error_init(&error);
        status = lw_session_create(model, &input, 1u, NULL, &session, &error);
        if (status != LW_STATUS_OK) {
            fprintf(stderr, "repeated session create failed at iteration %" PRIu32 ": %s\n", i,
                    error.message);
            lw_model_free(model);
            return 1;
        }
        lw_session_free(session);
        session = NULL;
    }
    printf("workspace_320=%" PRIu64 " workspace_batch2_640=%" PRIu64 "\n", workspace_320,
           workspace_640);
    lw_model_free(model);
    return 0;
}
