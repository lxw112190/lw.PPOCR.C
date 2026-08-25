#include "session_internal.h"
#include "lwm_read.h"

#include <stdlib.h>

typedef struct free_block {
    uint64_t offset;
    uint64_t size;
} free_block;

static uint64_t align_workspace(uint64_t value) {
    return (value + (LW_WORKSPACE_ALIGNMENT - 1u)) & ~(uint64_t)(LW_WORKSPACE_ALIGNMENT - 1u);
}

static int add_free_block(free_block* blocks, uint32_t* count, uint32_t capacity, uint64_t offset, uint64_t size) {
    uint32_t position = 0u;
    uint32_t i;
    if (*count >= capacity) {
        return 0;
    }
    while (position < *count && blocks[position].offset < offset) {
        ++position;
    }
    for (i = *count; i > position; --i) {
        blocks[i] = blocks[i - 1u];
    }
    blocks[position].offset = offset;
    blocks[position].size = size;
    ++(*count);
    if (position > 0u && blocks[position - 1u].offset + blocks[position - 1u].size == blocks[position].offset) {
        blocks[position - 1u].size += blocks[position].size;
        for (i = position; i + 1u < *count; ++i) {
            blocks[i] = blocks[i + 1u];
        }
        --(*count);
        --position;
    }
    if (position + 1u < *count && blocks[position].offset + blocks[position].size == blocks[position + 1u].offset) {
        blocks[position].size += blocks[position + 1u].size;
        for (i = position + 1u; i + 1u < *count; ++i) {
            blocks[i] = blocks[i + 1u];
        }
        --(*count);
    }
    return 1;
}

static int allocate_block(free_block* blocks, uint32_t* count, uint64_t size, uint64_t* offset) {
    uint32_t i;
    uint32_t j;
    for (i = 0u; i < *count; ++i) {
        if (blocks[i].size >= size) {
            *offset = blocks[i].offset;
            blocks[i].offset += size;
            blocks[i].size -= size;
            if (blocks[i].size == 0u) {
                for (j = i; j + 1u < *count; ++j) {
                    blocks[j] = blocks[j + 1u];
                }
                --(*count);
            }
            return 1;
        }
    }
    return 0;
}

lw_status lw_plan_workspace(lw_session* session, uint64_t max_workspace_size, lw_error* error) {
    const lw_model* model = session->model;
    free_block* blocks;
    uint32_t free_count = 0u;
    uint64_t workspace_end = 0u;
    uint32_t i;
    uint32_t tensor_index;

    blocks = (free_block*)calloc((size_t)model->info.tensor_count + 1u, sizeof(*blocks));
    if (blocks == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate workspace planner state");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    for (tensor_index = 0u; tensor_index < model->info.tensor_count; ++tensor_index) {
        lw_runtime_tensor* tensor = &session->tensors[tensor_index];
        tensor->workspace_offset = UINT64_MAX;
        tensor->birth_node = -1;
        tensor->last_use_node = -1;
        tensor->workspace_live = 0;
    }
    for (i = 0u; i < model->info.node_count; ++i) {
        const uint8_t* node = model->bytes + (size_t)model->node_offset + (size_t)i * LWM_V0_NODE_SIZE;
        uint16_t input_count = lwm_read_u16(node + 2);
        uint16_t output_count = lwm_read_u16(node + 4);
        uint32_t j;
        for (j = 0u; j < input_count; ++j) {
            lw_runtime_tensor* tensor = &session->tensors[lwm_read_u32(node + 8u + j * 4u)];
            tensor->last_use_node = (int32_t)i;
        }
        for (j = 0u; j < output_count; ++j) {
            lw_runtime_tensor* tensor = &session->tensors[lwm_read_u32(node + 40u + j * 4u)];
            if (tensor->birth_node != -1) {
                free(blocks);
                lw_set_error(error, LW_STATUS_INVALID_FORMAT, "multiple nodes produce the same tensor");
                return LW_STATUS_INVALID_FORMAT;
            }
            tensor->birth_node = (int32_t)i;
            tensor->last_use_node = (int32_t)i;
        }
    }
    for (i = 0u; i < model->info.output_count; ++i) {
        uint32_t index = lwm_read_u32(model->bytes + (size_t)model->output_offset + (size_t)i * 4u);
        session->tensors[index].last_use_node = (int32_t)model->info.node_count;
    }

    for (i = 0u; i < model->info.node_count; ++i) {
        const uint8_t* node = model->bytes + (size_t)model->node_offset + (size_t)i * LWM_V0_NODE_SIZE;
        uint16_t output_count = lwm_read_u16(node + 4);
        uint32_t j;
        for (tensor_index = 0u; tensor_index < model->info.tensor_count; ++tensor_index) {
            lw_runtime_tensor* tensor = &session->tensors[tensor_index];
            if (tensor->workspace_live && tensor->last_use_node < (int32_t)i) {
                uint64_t size = align_workspace(tensor->byte_size);
                if (!add_free_block(blocks, &free_count, model->info.tensor_count + 1u,
                                    tensor->workspace_offset, size)) {
                    free(blocks);
                    lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "workspace free-list capacity exceeded");
                    return LW_STATUS_OUT_OF_MEMORY;
                }
                tensor->workspace_live = 0;
            }
        }
        for (j = 0u; j < output_count; ++j) {
            lw_runtime_tensor* tensor = &session->tensors[lwm_read_u32(node + 40u + j * 4u)];
            uint64_t size;
            uint64_t offset;
            if (tensor->byte_size > UINT64_MAX - (LW_WORKSPACE_ALIGNMENT - 1u)) {
                free(blocks);
                lw_set_error(error, LW_STATUS_MEMORY_LIMIT, "aligned tensor workspace size overflows");
                return LW_STATUS_MEMORY_LIMIT;
            }
            size = align_workspace(tensor->byte_size);
            if (!allocate_block(blocks, &free_count, size, &offset)) {
                if (workspace_end > UINT64_MAX - size) {
                    free(blocks);
                    lw_set_error(error, LW_STATUS_MEMORY_LIMIT, "workspace size overflows");
                    return LW_STATUS_MEMORY_LIMIT;
                }
                offset = workspace_end;
                workspace_end += size;
            }
            if (workspace_end > max_workspace_size || workspace_end > SIZE_MAX) {
                free(blocks);
                lw_set_error(error, LW_STATUS_MEMORY_LIMIT, "planned workspace exceeds max_workspace_size");
                return LW_STATUS_MEMORY_LIMIT;
            }
            tensor->workspace_offset = offset;
            tensor->workspace_live = 1;
        }
    }
    free(blocks);
    session->workspace_bytes = (size_t)workspace_end;
    return LW_STATUS_OK;
}
