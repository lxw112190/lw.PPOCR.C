#include "session_internal.h"

/*
 * Session construction joins model metadata, concrete input shapes and the
 * workspace plan. The model bytes remain owned by lw_model; applications must
 * therefore keep the model alive until every session created from it is freed.
 */
#include "lwm_read.h"
#include "packed_conv_internal.h"
#include "packed_matmul_internal.h"
#include "parallel_internal.h"
#include "cpu_features.h"

#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  include <malloc.h>
#endif

#define LW_DEFAULT_MAX_WORKSPACE_SIZE (UINT64_C(512) * UINT64_C(1024) * UINT64_C(1024))
#define LW_DEFAULT_MAX_TENSOR_SIZE (UINT64_C(256) * UINT64_C(1024) * UINT64_C(1024))
#define LW_OP_CONV 1u
#define LW_OP_MATMUL 14u

static void* workspace_allocate(size_t size) {
#if defined(_WIN32)
    return _aligned_malloc(size, LW_WORKSPACE_ALIGNMENT);
#else
    void* pointer = NULL;
    if (posix_memalign(&pointer, LW_WORKSPACE_ALIGNMENT, size) != 0) {
        return NULL;
    }
    return pointer;
#endif
}

static void workspace_release(void* pointer) {
#if defined(_WIN32)
    _aligned_free(pointer);
#else
    free(pointer);
#endif
}

static int align_up_64(uint64_t value, uint64_t* aligned_value) {
    const uint64_t mask = LW_WORKSPACE_ALIGNMENT - 1u;
    if (aligned_value == NULL || value > UINT64_MAX - mask) {
        return 0;
    }
    *aligned_value = (value + mask) & ~mask;
    return 1;
}

static int uint64_fits_size_t(uint64_t value) {
    return (uint64_t)(size_t)value == value;
}

static int prepared_pointwise_geometry(lw_simd_level simd_level,
                                       const lw_runtime_tensor* input) {
    const uint64_t height = (uint32_t)input->dimensions[2];
    const uint64_t width = (uint32_t)input->dimensions[3];
    if (width >= UINT64_C(2) * height) {
        return lw_simd_level_has_packed_conv1x1(simd_level);
    }
#if defined(_M_X64) || defined(__x86_64__)
    /* The 4x16 microkernel needs the x64 register file. Keep square feature
     * maps on the canonical kernel for x86, SSE2-only hosts and tiny tensors. */
    return lw_simd_level_is_avx2(simd_level) && height * width >= 256u;
#else
    /* NEON and LSX both have enough vector registers for the portable packed
     * four-output microkernel on large square detector feature maps. */
    return (lw_simd_level_is_neon(simd_level) || lw_simd_level_is_lsx(simd_level)) &&
           height * width >= 256u;
#endif
}

static int prepared_pointwise_node(const lw_session* session, lw_simd_level simd_level,
                                   uint32_t node_index,
                                   uint32_t* weight_tensor_index, uint64_t* packed_weight_count) {
    const lw_model* model = session->model;
    const uint8_t* node =
        model->bytes + (size_t)model->node_offset + (size_t)node_index * LWM_V0_NODE_SIZE;
    uint16_t input_count = lwm_read_u16(node + 2u);
    uint64_t param_offset;
    const uint8_t* params;
    uint32_t input_index;
    uint32_t weights_index;
    uint32_t output_index;
    const lw_runtime_tensor* input;
    const lw_runtime_tensor* weights;
    const lw_runtime_tensor* output;
    if (lwm_read_u16(node) != LW_OP_CONV || (input_count != 2u && input_count != 3u)) {
        return 0;
    }
    param_offset = lwm_read_u64(node + 56u);
    params = model->bytes + (size_t)param_offset;
    input_index = lwm_read_u32(node + 8u);
    weights_index = lwm_read_u32(node + 12u);
    output_index = lwm_read_u32(node + 40u);
    input = &session->tensors[input_index];
    weights = &session->tensors[weights_index];
    output = &session->tensors[output_index];
    /* Long OCR feature maps use the portable packed path. Large square maps
     * additionally require the x64 AVX2 4x16 microkernel. */
    if (input->dtype != LW_DTYPE_F32 || input->rank != 4u || weights->dtype != LW_DTYPE_F32 ||
        weights->rank != 4u || output->dtype != LW_DTYPE_F32 || output->rank != 4u ||
        (weights->flags & LWM_V0_TENSOR_FLAG_CONSTANT) == 0u || lwm_read_u32(params + 4u) != 1u ||
        lwm_read_i32(params + 8u) != 1 || lwm_read_i32(params + 12u) != 1 ||
        lwm_read_i32(params + 16u) != 1 || lwm_read_i32(params + 20u) != 1 ||
        lwm_read_i32(params + 24u) != 1 || lwm_read_i32(params + 28u) != 1 ||
        lwm_read_i32(params + 32u) != 0 || lwm_read_i32(params + 36u) != 0 ||
        lwm_read_i32(params + 40u) != 0 || lwm_read_i32(params + 44u) != 0 ||
        weights->dimensions[0] != output->dimensions[1] ||
        weights->dimensions[1] != input->dimensions[1] || weights->dimensions[2] != 1 ||
        weights->dimensions[3] != 1 || input->dimensions[0] != output->dimensions[0] ||
        output->dimensions[1] < (int32_t)LW_PACKED_CONV1X1_OUTPUT_TILE ||
        output->dimensions[1] % (int32_t)LW_PACKED_CONV1X1_OUTPUT_TILE != 0 ||
        !prepared_pointwise_geometry(simd_level, input) ||
        input->dimensions[2] != output->dimensions[2] ||
        input->dimensions[3] != output->dimensions[3] ||
        !lw_packed_conv1x1_weight_count((uint32_t)input->dimensions[1],
                                        (uint32_t)output->dimensions[1], packed_weight_count)) {
        return 0;
    }
    *weight_tensor_index = weights_index;
    return 1;
}

static int prepared_matmul_node(const lw_session* session, lw_simd_level simd_level,
                                uint32_t node_index, uint32_t* weight_tensor_index,
                                uint64_t* packed_weight_count) {
#if defined(_M_X64) || defined(__x86_64__)
    const lw_model* model = session->model;
    const uint8_t* node =
        model->bytes + (size_t)model->node_offset + (size_t)node_index * LWM_V0_NODE_SIZE;
    uint32_t input_index;
    uint32_t weights_index;
    uint32_t output_index;
    const lw_runtime_tensor* input;
    const lw_runtime_tensor* weights;
    const lw_runtime_tensor* output;
    uint32_t rows;
    uint32_t inner_dimension;
    uint32_t columns;
    if (!lw_simd_level_is_avx2(simd_level) || lwm_read_u16(node) != LW_OP_MATMUL ||
        lwm_read_u16(node + 2u) != 2u) {
        return 0;
    }
    input_index = lwm_read_u32(node + 8u);
    weights_index = lwm_read_u32(node + 12u);
    output_index = lwm_read_u32(node + 40u);
    input = &session->tensors[input_index];
    weights = &session->tensors[weights_index];
    output = &session->tensors[output_index];
    if (input->dtype != LW_DTYPE_F32 || input->rank < 2u ||
        weights->dtype != LW_DTYPE_F32 || weights->rank != 2u ||
        (weights->flags & LWM_V0_TENSOR_FLAG_CONSTANT) == 0u || output->dtype != LW_DTYPE_F32 ||
        output->rank != input->rank) {
        return 0;
    }
    rows = (uint32_t)input->dimensions[input->rank - 2u];
    inner_dimension = (uint32_t)input->dimensions[input->rank - 1u];
    columns = (uint32_t)weights->dimensions[1];
    if (rows < 4u || rows % 4u != 0u || inner_dimension < 64u || columns < 1024u ||
        weights->dimensions[0] != (int32_t)inner_dimension ||
        output->dimensions[output->rank - 2u] != (int32_t)rows ||
        output->dimensions[output->rank - 1u] != (int32_t)columns ||
        !lw_packed_matmul_weight_count(inner_dimension, columns, packed_weight_count)) {
        return 0;
    }
    *weight_tensor_index = weights_index;
    return 1;
#else
    (void)session;
    (void)simd_level;
    (void)node_index;
    (void)weight_tensor_index;
    (void)packed_weight_count;
    return 0;
#endif
}

static const float* constant_f32_data(const lw_session* session, uint32_t tensor_index) {
    const lw_model* model = session->model;
    const uint8_t* disk =
        model->bytes + (size_t)model->tensor_offset + (size_t)tensor_index * LWM_V0_TENSOR_SIZE;
    uint64_t data_offset = lwm_read_u64(disk + 48u);
    return (const float*)(const void*)(model->bytes + (size_t)data_offset);
}

static lw_status prepare_constant_weights(lw_session* session, lw_error* error) {
    const lw_model* model = session->model;
    const lw_simd_level simd_level = lw_detect_simd_level();
    uint64_t total_bytes = 0u;
    uint32_t node_index;
    if (model->info.node_count == 0u ||
        !lw_simd_level_has_packed_conv1x1(simd_level)) {
        return LW_STATUS_OK;
    }
    session->prepared_nodes =
        (lw_prepared_node*)calloc(model->info.node_count, sizeof(*session->prepared_nodes));
    if (session->prepared_nodes == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate prepared node table");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    for (node_index = 0u; node_index < model->info.node_count; ++node_index) {
        uint32_t weight_tensor_index;
        uint64_t packed_weight_count;
        uint64_t packed_bytes;
        lw_prepared_node* prepared = &session->prepared_nodes[node_index];
        if (prepared_pointwise_node(session, simd_level, node_index, &weight_tensor_index,
                                    &packed_weight_count)) {
            prepared->kind = LW_PREPARED_NODE_CONV1X1_PACKED4;
        } else if (prepared_matmul_node(session, simd_level, node_index, &weight_tensor_index,
                                        &packed_weight_count)) {
            prepared->kind = LW_PREPARED_NODE_MATMUL_PACKED16;
        } else {
            continue;
        }
        (void)weight_tensor_index;
        packed_bytes = packed_weight_count * sizeof(float);
        if (!align_up_64(total_bytes, &total_bytes)) {
            lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "packed weight alignment overflows");
            return LW_STATUS_OUT_OF_BOUNDS;
        }
        if (total_bytes > UINT64_MAX - packed_bytes ||
            !uint64_fits_size_t(total_bytes + packed_bytes)) {
            lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "packed weights overflow");
            return LW_STATUS_OUT_OF_BOUNDS;
        }
        prepared->packed_weight_offset = total_bytes;
        prepared->packed_weight_count = packed_weight_count;
        total_bytes += packed_bytes;
    }
    if (total_bytes == 0u) {
        return LW_STATUS_OK;
    }
    if (!align_up_64(total_bytes, &total_bytes) || !uint64_fits_size_t(total_bytes)) {
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "packed weight allocation overflows");
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    session->packed_weights = (uint8_t*)workspace_allocate((size_t)total_bytes);
    if (session->packed_weights == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate packed weights");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    session->packed_weight_bytes = (size_t)total_bytes;
    for (node_index = 0u; node_index < model->info.node_count; ++node_index) {
        lw_prepared_node* prepared = &session->prepared_nodes[node_index];
        uint32_t weight_tensor_index;
        uint64_t packed_weight_count;
        const uint8_t* node;
        uint32_t input_index;
        uint32_t output_index;
        node = model->bytes + (size_t)model->node_offset + (size_t)node_index * LWM_V0_NODE_SIZE;
        if (prepared->kind == LW_PREPARED_NODE_CONV1X1_PACKED4 &&
            prepared_pointwise_node(session, simd_level, node_index, &weight_tensor_index,
                                    &packed_weight_count) &&
            packed_weight_count == prepared->packed_weight_count) {
            input_index = lwm_read_u32(node + 8u);
            output_index = lwm_read_u32(node + 40u);
            lw_pack_conv1x1_weights_f32(
                constant_f32_data(session, weight_tensor_index),
                (uint32_t)session->tensors[input_index].dimensions[1],
                (uint32_t)session->tensors[output_index].dimensions[1],
                (float*)(void*)(session->packed_weights +
                                (size_t)prepared->packed_weight_offset));
        } else if (prepared->kind == LW_PREPARED_NODE_MATMUL_PACKED16 &&
                   prepared_matmul_node(session, simd_level, node_index, &weight_tensor_index,
                                        &packed_weight_count) &&
                   packed_weight_count == prepared->packed_weight_count) {
            const lw_runtime_tensor* weights = &session->tensors[weight_tensor_index];
            lw_pack_matmul_weights_f32(
                constant_f32_data(session, weight_tensor_index),
                (uint32_t)weights->dimensions[0], (uint32_t)weights->dimensions[1],
                (float*)(void*)(session->packed_weights +
                                (size_t)prepared->packed_weight_offset));
        }
    }
    return LW_STATUS_OK;
}

static uint32_t runtime_dtype_size(uint32_t dtype) {
    switch (dtype) {
    case LW_DTYPE_F32:
        return 4u;
    case LW_DTYPE_I32:
        return 4u;
    case LW_DTYPE_I64:
        return 8u;
    case LW_DTYPE_U8:
        return 1u;
    default:
        return 0u;
    }
}

static lw_status resolve_existing_tensor_size(lw_runtime_tensor* tensor, uint64_t max_tensor_size,
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
            lw_set_error(error, LW_STATUS_INVALID_SHAPE,
                         "input or constant tensor dimensions are invalid");
            return LW_STATUS_INVALID_SHAPE;
        }
        elements *= (uint32_t)dimension;
    }
    if (elements > UINT64_MAX / item_size) {
        lw_set_error(error, LW_STATUS_INVALID_SHAPE,
                     "input or constant tensor byte size overflows");
        return LW_STATUS_INVALID_SHAPE;
    }
    tensor->byte_size = elements * item_size;
    if (tensor->byte_size > max_tensor_size) {
        lw_set_error(error, LW_STATUS_MEMORY_LIMIT,
                     "input or constant tensor exceeds max_tensor_size");
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

lw_status lw_session_create(const lw_model* model, const lw_tensor_desc* inputs,
                            uint32_t input_count, const lw_session_options* options,
                            lw_session** out_session, lw_error* error) {
    uint64_t max_workspace_size = LW_DEFAULT_MAX_WORKSPACE_SIZE;
    uint64_t max_tensor_size = LW_DEFAULT_MAX_TENSOR_SIZE;
    lw_session* session;
    uint32_t i;
    lw_status status;
    if (out_session != NULL) {
        *out_session = NULL;
    }
    if (model == NULL || inputs == NULL || out_session == NULL ||
        input_count != model->info.input_count) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "model, inputs, matching input_count, and out_session are required");
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
    session->intra_op_thread_count = 1u;
    session->model = model;
    session->tensors =
        (lw_runtime_tensor*)calloc(model->info.tensor_count, sizeof(*session->tensors));
    if (session->tensors == NULL) {
        lw_session_free(session);
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate resolved tensor table");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    for (i = 0u; i < model->info.tensor_count; ++i) {
        const uint8_t* disk =
            model->bytes + (size_t)model->tensor_offset + (size_t)i * LWM_V0_TENSOR_SIZE;
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
    /* Replace model -1 dimensions with the caller's concrete input shape while
     * requiring every static dimension to match exactly. */
    for (i = 0u; i < input_count; ++i) {
        uint32_t tensor_index =
            lwm_read_u32(model->bytes + (size_t)model->input_offset + (size_t)i * 4u);
        lw_runtime_tensor* tensor = &session->tensors[tensor_index];
        const lw_tensor_desc* input = &inputs[i];
        uint32_t j;
        if (input->struct_size != sizeof(*input) || input->reserved != 0u ||
            input->dtype != tensor->dtype || input->rank != tensor->rank) {
            lw_session_free(session);
            lw_set_error(error, LW_STATUS_INVALID_SHAPE,
                         "input tensor descriptor type or rank is invalid");
            return LW_STATUS_INVALID_SHAPE;
        }
        for (j = 0u; j < input->rank; ++j) {
            if (input->dimensions[j] <= 0 ||
                (tensor->dimensions[j] != -1 && tensor->dimensions[j] != input->dimensions[j])) {
                lw_session_free(session);
                lw_set_error(error, LW_STATUS_INVALID_SHAPE,
                             "input tensor dimensions disagree with the model");
                return LW_STATUS_INVALID_SHAPE;
            }
        }
        for (j = input->rank; j < LW_MAX_DIMS; ++j) {
            if (input->dimensions[j] != 0) {
                lw_session_free(session);
                lw_set_error(error, LW_STATUS_INVALID_SHAPE,
                             "unused input tensor dimensions must be zero");
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
    /* Shape resolution precedes planning because every workspace allocation
     * needs a concrete, overflow-checked byte size. */
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
            lw_set_error(error, LW_STATUS_OUT_OF_MEMORY,
                         "unable to allocate planned session workspace");
            return LW_STATUS_OUT_OF_MEMORY;
        }
    }
    status = prepare_constant_weights(session, error);
    if (status != LW_STATUS_OK) {
        lw_session_free(session);
        return status;
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

void lw_session_set_intra_op_thread_count(lw_session* session, uint32_t thread_count) {
    if (session == NULL) {
        return;
    }
    if (thread_count == 0u) {
        thread_count = 1u;
    }
    if (thread_count > LW_PARALLEL_MAX_WORKERS) {
        thread_count = LW_PARALLEL_MAX_WORKERS;
    }
    if (session->intra_op_thread_count == thread_count) {
        return;
    }
    lw_thread_pool_free(session->thread_pool);
    session->thread_pool = thread_count > 1u ? lw_thread_pool_create(thread_count) : NULL;
    session->intra_op_thread_count = lw_thread_pool_worker_count(session->thread_pool);
}

void lw_session_free(lw_session* session) {
    if (session == NULL) {
        return;
    }
    lw_thread_pool_free(session->thread_pool);
    session->thread_pool = NULL;
    workspace_release(session->workspace);
    session->workspace = NULL;
    workspace_release(session->packed_weights);
    session->packed_weights = NULL;
    session->packed_weight_bytes = 0u;
    free(session->prepared_nodes);
    session->prepared_nodes = NULL;
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

lw_status lw_session_get_output_desc(const lw_session* session, uint32_t output_index,
                                     lw_tensor_desc* output) {
    uint32_t tensor_index;
    const lw_runtime_tensor* tensor;
    if (session == NULL || output == NULL || output->struct_size != sizeof(*output) ||
        output_index >= session->model->info.output_count) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    tensor_index = lwm_read_u32(session->model->bytes + (size_t)session->model->output_offset +
                                (size_t)output_index * 4u);
    tensor = &session->tensors[tensor_index];
    memset(output, 0, sizeof(*output));
    output->struct_size = (uint32_t)sizeof(*output);
    output->dtype = tensor->dtype;
    output->rank = tensor->rank;
    memcpy(output->dimensions, tensor->dimensions, sizeof(output->dimensions));
    return LW_STATUS_OK;
}
