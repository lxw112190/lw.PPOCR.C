#include "session_internal.h"
#include "lwm_read.h"

#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  include <malloc.h>
#endif

#define LW_DEFAULT_MAX_WORKSPACE_SIZE (UINT64_C(512) * UINT64_C(1024) * UINT64_C(1024))
#define LW_DEFAULT_MAX_TENSOR_SIZE (UINT64_C(256) * UINT64_C(1024) * UINT64_C(1024))

static void* workspace_allocate(size_t size) {
#if defined(_WIN32)
    return _aligned_malloc(size, LW_WORKSPACE_ALIGNMENT);
#else
    return aligned_alloc(LW_WORKSPACE_ALIGNMENT, size);
#endif
}

static void workspace_release(void* pointer) {
#if defined(_WIN32)
    _aligned_free(pointer);
#else
    free(pointer);
#endif
}

static uint32_t runtime_dtype_size(uint32_t dtype) {
    switch (dtype) {
        case LW_DTYPE_F32: return 4u;
        case LW_DTYPE_I32: return 4u;
        case LW_DTYPE_I64: return 8u;
        case LW_DTYPE_U8: return 1u;
        default: return 0u;
    }
}

static lw_status resolve_existing_tensor_size(
    lw_runtime_tensor* tensor,
    uint64_t max_tensor_size,
    lw_error* error) {
    uint64_t elements = 1u;
    uint32_t item_size = runtime_dtype_size(tensor->dtype);
    uint32_t i;
    if (item_size == 0u || tensor->rank > LW_MAX_DIMS) {
        lw_set_error(error, LW_STATUS_INVALID_SHAPE, "input or constant tensor type is invalid");
        return LW_STATUS_INVALID_SHAPE;
    }
    for (i = 0u; i < tensor->rank; ++i) {
        int32_t dimension = tensor->dimensions[i];
        if (dimension <= 0 || elements > UINT64_MAX / (uint32_t)dimension) {
            lw_set_error(error, LW_STATUS_INVALID_SHAPE, "input or constant tensor dimensions are invalid");
            return LW_STATUS_INVALID_SHAPE;
        }
        elements *= (uint32_t)dimension;
    }
    if (elements > UINT64_MAX / item_size) {
        lw_set_error(error, LW_STATUS_INVALID_SHAPE, "input or constant tensor byte size overflows");
        return LW_STATUS_INVALID_SHAPE;
    }
    tensor->byte_size = elements * item_size;
    if (tensor->byte_size > max_tensor_size) {
        lw_set_error(error, LW_STATUS_MEMORY_LIMIT, "input or constant tensor exceeds max_tensor_size");
        return LW_STATUS_MEMORY_LIMIT;
    }
    return LW_STATUS_OK;
}

void lw_tensor_desc_init(lw_tensor_desc* tensor) {
    if (tensor == NULL) {
        return;
    }
    memset(tensor, 0, sizeof(*tensor));
    tensor->struct_size = (uint32_t)sizeof(*tensor);
}

void lw_session_options_init(lw_session_options* options) {
    if (options == NULL) {
        return;
    }
    memset(options, 0, sizeof(*options));
    options->struct_size = (uint32_t)sizeof(*options);
    options->max_workspace_size = LW_DEFAULT_MAX_WORKSPACE_SIZE;
    options->max_tensor_size = LW_DEFAULT_MAX_TENSOR_SIZE;
}

void lw_session_info_init(lw_session_info* info) {
    if (info == NULL) {
        return;
    }
    memset(info, 0, sizeof(*info));
    info->struct_size = (uint32_t)sizeof(*info);
}

lw_status lw_session_create(
    const lw_model* model,
    const lw_tensor_desc* inputs,
    uint32_t input_count,
    const lw_session_options* options,
    lw_session** out_session,
    lw_error* error) {
    uint64_t max_workspace_size = LW_DEFAULT_MAX_WORKSPACE_SIZE;
    uint64_t max_tensor_size = LW_DEFAULT_MAX_TENSOR_SIZE;
    lw_session* session;
    uint32_t i;
    lw_status status;
    if (out_session != NULL) {
        *out_session = NULL;
    }
    if (model == NULL || inputs == NULL || out_session == NULL || input_count != model->info.input_count) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "model, inputs, matching input_count, and out_session are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    if (options != NULL) {
        if (options->struct_size != sizeof(*options) || options->reserved != 0u) {
            lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "invalid session options structure");
            return LW_STATUS_INVALID_ARGUMENT;
        }
        if (options->max_workspace_size != 0u) {
            max_workspace_size = options->max_workspace_size;
        }
        if (options->max_tensor_size != 0u) {
            max_tensor_size = options->max_tensor_size;
        }
    }
    session = (lw_session*)calloc(1u, sizeof(*session));
    if (session == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate session handle");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    session->model = model;
    session->tensors = (lw_runtime_tensor*)calloc(model->info.tensor_count, sizeof(*session->tensors));
    if (session->tensors == NULL) {
        lw_session_free(session);
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate resolved tensor table");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    for (i = 0u; i < model->info.tensor_count; ++i) {
        const uint8_t* disk = model->bytes + (size_t)model->tensor_offset + (size_t)i * LWM_V0_TENSOR_SIZE;
        lw_runtime_tensor* tensor = &session->tensors[i];
        uint32_t j;
        tensor->dtype = lwm_read_u32(disk);
        tensor->rank = lwm_read_u32(disk + 4);
        tensor->flags = lwm_read_u32(disk + 40);
        tensor->workspace_offset = UINT64_MAX;
        for (j = 0u; j < LW_MAX_DIMS; ++j) {
            tensor->dimensions[j] = lwm_read_i32(disk + 8u + j * 4u);
        }
    }
    for (i = 0u; i < input_count; ++i) {
        uint32_t tensor_index = lwm_read_u32(model->bytes + (size_t)model->input_offset + (size_t)i * 4u);
        lw_runtime_tensor* tensor = &session->tensors[tensor_index];
        const lw_tensor_desc* input = &inputs[i];
        uint32_t j;
        if (input->struct_size != sizeof(*input) || input->reserved != 0u ||
            input->dtype != tensor->dtype || input->rank != tensor->rank) {
            lw_session_free(session);
            lw_set_error(error, LW_STATUS_INVALID_SHAPE, "input tensor descriptor type or rank is invalid");
            return LW_STATUS_INVALID_SHAPE;
        }
        for (j = 0u; j < input->rank; ++j) {
            if (input->dimensions[j] <= 0 ||
                (tensor->dimensions[j] != -1 && tensor->dimensions[j] != input->dimensions[j])) {
                lw_session_free(session);
                lw_set_error(error, LW_STATUS_INVALID_SHAPE, "input tensor dimensions disagree with the model");
                return LW_STATUS_INVALID_SHAPE;
            }
        }
        for (j = input->rank; j < LW_MAX_DIMS; ++j) {
            if (input->dimensions[j] != 0) {
                lw_session_free(session);
                lw_set_error(error, LW_STATUS_INVALID_SHAPE, "unused input tensor dimensions must be zero");
                return LW_STATUS_INVALID_SHAPE;
            }
        }
        memcpy(tensor->dimensions, input->dimensions, sizeof(tensor->dimensions));
    }
    for (i = 0u; i < model->info.tensor_count; ++i) {
        lw_runtime_tensor* tensor = &session->tensors[i];
        if ((tensor->flags & (LWM_V0_TENSOR_FLAG_CONSTANT | LWM_V0_TENSOR_FLAG_INPUT)) != 0u) {
            status = resolve_existing_tensor_size(tensor, max_tensor_size, error);
            if (status != LW_STATUS_OK) {
                lw_session_free(session);
                return status;
            }
        }
    }
    status = lw_resolve_shapes(session, max_tensor_size, error);
    if (status != LW_STATUS_OK) {
        lw_session_free(session);
        return status;
    }
    status = lw_plan_workspace(session, max_workspace_size, error);
    if (status != LW_STATUS_OK) {
        lw_session_free(session);
        return status;
    }
    if (session->workspace_bytes != 0u) {
        session->workspace = (uint8_t*)workspace_allocate(session->workspace_bytes);
        if (session->workspace == NULL) {
            lw_session_free(session);
            lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate planned session workspace");
            return LW_STATUS_OUT_OF_MEMORY;
        }
    }
    memset(&session->info, 0, sizeof(session->info));
    session->info.struct_size = (uint32_t)sizeof(session->info);
    session->info.tensor_count = model->info.tensor_count;
    session->info.input_count = model->info.input_count;
    session->info.output_count = model->info.output_count;
    session->info.workspace_size = session->workspace_bytes;
    *out_session = session;
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}

void lw_session_free(lw_session* session) {
    if (session == NULL) {
        return;
    }
    workspace_release(session->workspace);
    session->workspace = NULL;
    free(session->tensors);
    session->tensors = NULL;
    free(session);
}

lw_status lw_session_get_info(const lw_session* session, lw_session_info* info) {
    if (session == NULL || info == NULL || info->struct_size != sizeof(*info)) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    *info = session->info;
    return LW_STATUS_OK;
}

lw_status lw_session_get_output_desc(
    const lw_session* session,
    uint32_t output_index,
    lw_tensor_desc* output) {
    uint32_t tensor_index;
    const lw_runtime_tensor* tensor;
    if (session == NULL || output == NULL || output->struct_size != sizeof(*output) ||
        output_index >= session->model->info.output_count) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    tensor_index = lwm_read_u32(
        session->model->bytes + (size_t)session->model->output_offset + (size_t)output_index * 4u);
    tensor = &session->tensors[tensor_index];
    memset(output, 0, sizeof(*output));
    output->struct_size = (uint32_t)sizeof(*output);
    output->dtype = tensor->dtype;
    output->rank = tensor->rank;
    memcpy(output->dimensions, tensor->dimensions, sizeof(output->dimensions));
    return LW_STATUS_OK;
}
