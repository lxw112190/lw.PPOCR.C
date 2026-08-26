#include "executor_internal.h"

/*
 * Small graph interpreter for the operator subset emitted by the converter.
 * Constants point into validated model bytes; intermediate tensors point into
 * the preplanned workspace. Dispatch must stay allocation-free during a run.
 */

#include "lwm_read.h"
#include "scalar_kernels.h"
#include "session_internal.h"

#include <stdio.h>
#include <string.h>

enum {
    LW_OP_CONV = 1,
    LW_OP_ADD = 2,
    LW_OP_MUL = 3,
    LW_OP_DIV = 4,
    LW_OP_ERF = 5,
    LW_OP_HARD_SIGMOID = 6,
    LW_OP_BATCH_NORMALIZATION = 7,
    LW_OP_REDUCE_MEAN = 8,
    LW_OP_RELU = 9,
    LW_OP_AVERAGE_POOL = 10,
    LW_OP_SQUEEZE = 11,
    LW_OP_TRANSPOSE = 12,
    LW_OP_UNSQUEEZE = 13,
    LW_OP_MATMUL = 14,
    LW_OP_SOFTMAX = 15,
    LW_OP_RESHAPE = 16,
    LW_OP_CONCAT = 17,
    LW_OP_CONV_TRANSPOSE = 18,
    LW_OP_MAX_POOL = 19,
    LW_OP_RESIZE = 20,
    LW_OP_SIGMOID = 21
};

static float read_f32(const uint8_t* bytes) {
    uint32_t bits = lwm_read_u32(bytes);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint64_t tensor_element_count(const lw_runtime_tensor* tensor) {
    return tensor->byte_size / sizeof(float);
}

static const float* tensor_input_data(const lw_session* session, uint32_t tensor_index,
                                      uint32_t graph_input_index, const float* graph_input) {
    const lw_runtime_tensor* tensor = &session->tensors[tensor_index];
    const lw_model* model = session->model;
    if (tensor_index == graph_input_index) {
        return graph_input;
    }
    if ((tensor->flags & LWM_V0_TENSOR_FLAG_CONSTANT) != 0u) {
        /* Constant payloads are zero-copy views into immutable model bytes. */
        const uint8_t* disk =
            model->bytes + (size_t)model->tensor_offset + (size_t)tensor_index * LWM_V0_TENSOR_SIZE;
        uint64_t data_offset = lwm_read_u64(disk + 48);
        return (const float*)(const void*)(model->bytes + (size_t)data_offset);
    }
    if (tensor->workspace_offset == UINT64_MAX) {
        return NULL;
    }
    return (const float*)(const void*)(session->workspace + (size_t)tensor->workspace_offset);
}

static float* tensor_output_data(lw_session* session, uint32_t tensor_index) {
    const lw_runtime_tensor* tensor = &session->tensors[tensor_index];
    if (tensor->workspace_offset == UINT64_MAX) {
        return NULL;
    }
    return (float*)(void*)(session->workspace + (size_t)tensor->workspace_offset);
}

static lw_status dispatch_node(lw_session* session, const uint8_t* node, uint32_t graph_input_index,
                               const float* graph_input) {
    const lw_model* model = session->model;
    const uint16_t op = lwm_read_u16(node);
    const uint16_t input_count = lwm_read_u16(node + 2);
    const uint16_t output_count = lwm_read_u16(node + 4);
    const uint64_t param_offset = lwm_read_u64(node + 56);
    const uint8_t* params = param_offset == 0u ? NULL : model->bytes + (size_t)param_offset;
    const lw_runtime_tensor* input_tensors[LWM_V0_MAX_NODE_INPUTS];
    const float* inputs[LWM_V0_MAX_NODE_INPUTS];
    lw_runtime_tensor* output_tensor;
    float* output;
    uint32_t index;

    if (output_count != 1u) {
        return LW_STATUS_UNSUPPORTED;
    }
    for (index = 0u; index < input_count; ++index) {
        uint32_t tensor_index = lwm_read_u32(node + 8u + index * 4u);
        input_tensors[index] = &session->tensors[tensor_index];
        inputs[index] = tensor_input_data(session, tensor_index, graph_input_index, graph_input);
        if (input_tensors[index]->dtype != LW_DTYPE_F32 || inputs[index] == NULL) {
            return LW_STATUS_UNSUPPORTED;
        }
    }
    index = lwm_read_u32(node + 40);
    output_tensor = &session->tensors[index];
    output = tensor_output_data(session, index);
    if (output_tensor->dtype != LW_DTYPE_F32 || output == NULL) {
        return LW_STATUS_UNSUPPORTED;
    }

    /* Parameters were structurally validated at model-load time. Each kernel
     * still validates runtime-dependent shapes before reading tensor data. */
    switch (op) {
    case LW_OP_CONV: {
        int32_t kernel[2] = {lwm_read_i32(params + 8), lwm_read_i32(params + 12)};
        int32_t strides[2] = {lwm_read_i32(params + 16), lwm_read_i32(params + 20)};
        int32_t dilations[2] = {lwm_read_i32(params + 24), lwm_read_i32(params + 28)};
        int32_t pads[4] = {lwm_read_i32(params + 32), lwm_read_i32(params + 36),
                           lwm_read_i32(params + 40), lwm_read_i32(params + 44)};
        if (input_count != 2u && input_count != 3u) {
            return LW_STATUS_INVALID_SHAPE;
        }
        return lw_scalar_conv2d_f32(
            inputs[0], inputs[1], input_count == 3u ? inputs[2] : NULL,
            input_count == 3u ? (uint32_t)tensor_element_count(input_tensors[2]) : 0u, output,
            input_tensors[0]->dimensions, input_tensors[1]->dimensions, output_tensor->dimensions,
            kernel, strides, dilations, pads, lwm_read_u32(params + 4));
    }
    case LW_OP_ADD:
    case LW_OP_MUL:
    case LW_OP_DIV: {
        lw_scalar_binary_op operation =
            op == LW_OP_ADD ? LW_SCALAR_BINARY_ADD
                            : (op == LW_OP_MUL ? LW_SCALAR_BINARY_MUL : LW_SCALAR_BINARY_DIV);
        if (input_count != 2u) {
            return LW_STATUS_INVALID_SHAPE;
        }
        return lw_scalar_binary_f32(operation, inputs[0], input_tensors[0]->rank,
                                    input_tensors[0]->dimensions, inputs[1], input_tensors[1]->rank,
                                    input_tensors[1]->dimensions, output, output_tensor->rank,
                                    output_tensor->dimensions);
    }
    case LW_OP_ERF:
        return input_count == 1u
                   ? lw_scalar_erf_f32(inputs[0], output, tensor_element_count(input_tensors[0]))
                   : LW_STATUS_INVALID_SHAPE;
    case LW_OP_HARD_SIGMOID:
        return input_count == 1u
                   ? lw_scalar_hard_sigmoid_f32(inputs[0], output,
                                                tensor_element_count(input_tensors[0]),
                                                read_f32(params + 4), read_f32(params + 8))
                   : LW_STATUS_INVALID_SHAPE;
    case LW_OP_BATCH_NORMALIZATION:
        return input_count == 5u
                   ? lw_scalar_batch_normalization_f32(
                         inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                         (uint32_t)tensor_element_count(input_tensors[1]), read_f32(params + 4),
                         output, input_tensors[0]->rank, input_tensors[0]->dimensions)
                   : LW_STATUS_INVALID_SHAPE;
    case LW_OP_REDUCE_MEAN: {
        int32_t axes[LW_MAX_DIMS];
        uint32_t axes_count = lwm_read_u16(params + 2);
        if (input_count != 1u) {
            return LW_STATUS_INVALID_SHAPE;
        }
        for (index = 0u; index < axes_count; ++index) {
            axes[index] = lwm_read_i32(params + 12u + index * 4u);
        }
        return lw_scalar_reduce_mean_f32(inputs[0], output, input_tensors[0]->rank,
                                         input_tensors[0]->dimensions, axes_count, axes,
                                         lwm_read_u32(params + 4), lwm_read_u32(params + 8),
                                         output_tensor->rank, output_tensor->dimensions);
    }
    case LW_OP_RELU:
        return input_count == 1u
                   ? lw_scalar_relu_f32(inputs[0], output, tensor_element_count(input_tensors[0]))
                   : LW_STATUS_INVALID_SHAPE;
    case LW_OP_AVERAGE_POOL: {
        int32_t kernel[2] = {lwm_read_i32(params + 8), lwm_read_i32(params + 12)};
        int32_t strides[2] = {lwm_read_i32(params + 16), lwm_read_i32(params + 20)};
        int32_t pads[4] = {lwm_read_i32(params + 24), lwm_read_i32(params + 28),
                           lwm_read_i32(params + 32), lwm_read_i32(params + 36)};
        if (input_count != 1u) {
            return LW_STATUS_INVALID_SHAPE;
        }
        return lw_scalar_average_pool2d_f32(inputs[0], output, input_tensors[0]->dimensions,
                                            output_tensor->dimensions, kernel, strides, pads,
                                            lwm_read_u32(params + 40), lwm_read_u32(params + 44));
    }
    case LW_OP_SQUEEZE:
    case LW_OP_TRANSPOSE:
    case LW_OP_UNSQUEEZE: {
        int32_t axes[LW_MAX_DIMS];
        uint32_t axes_count = lwm_read_u16(params + 2);
        if (input_count != 1u) {
            return LW_STATUS_INVALID_SHAPE;
        }
        for (index = 0u; index < axes_count; ++index) {
            axes[index] = lwm_read_i32(params + 4u + index * 4u);
        }
        if (op == LW_OP_SQUEEZE) {
            return lw_scalar_squeeze_f32(inputs[0], output, input_tensors[0]->rank,
                                         input_tensors[0]->dimensions, axes_count, axes,
                                         output_tensor->rank, output_tensor->dimensions);
        }
        if (op == LW_OP_UNSQUEEZE) {
            return lw_scalar_unsqueeze_f32(inputs[0], output, input_tensors[0]->rank,
                                           input_tensors[0]->dimensions, axes_count, axes,
                                           output_tensor->rank, output_tensor->dimensions);
        }
        return lw_scalar_transpose_f32(inputs[0], output, input_tensors[0]->rank,
                                       input_tensors[0]->dimensions, axes_count, axes,
                                       output_tensor->dimensions);
    }
    case LW_OP_MATMUL: {
        uint64_t batch_count = 1u;
        uint32_t rank = input_tensors[0]->rank;
        if (input_count != 2u || rank < 2u || input_tensors[1]->rank != 2u) {
            return LW_STATUS_UNSUPPORTED;
        }
        for (index = 0u; index + 2u < rank; ++index) {
            batch_count *= (uint32_t)input_tensors[0]->dimensions[index];
        }
        if (batch_count > UINT32_MAX) {
            return LW_STATUS_OUT_OF_BOUNDS;
        }
        return lw_matmul_shared_f32(inputs[0], inputs[1], output, (uint32_t)batch_count,
                                    (uint32_t)input_tensors[0]->dimensions[rank - 2u],
                                    (uint32_t)input_tensors[0]->dimensions[rank - 1u],
                                    (uint32_t)input_tensors[1]->dimensions[1]);
    }
    case LW_OP_SOFTMAX:
        return input_count == 1u
                   ? lw_scalar_softmax_f32(inputs[0], output, input_tensors[0]->rank,
                                           input_tensors[0]->dimensions, lwm_read_i32(params + 4))
                   : LW_STATUS_INVALID_SHAPE;
    case LW_OP_RESHAPE:
        return input_count == 1u
                   ? lw_scalar_reshape_f32(inputs[0], output, input_tensors[0]->rank,
                                           input_tensors[0]->dimensions, output_tensor->rank,
                                           output_tensor->dimensions)
                   : LW_STATUS_INVALID_SHAPE;
    case LW_OP_CONCAT: {
        uint32_t ranks[LWM_V0_MAX_NODE_INPUTS];
        const int32_t* dimensions[LWM_V0_MAX_NODE_INPUTS];
        for (index = 0u; index < input_count; ++index) {
            ranks[index] = input_tensors[index]->rank;
            dimensions[index] = input_tensors[index]->dimensions;
        }
        return lw_scalar_concat_f32(inputs, input_count, ranks, dimensions, output,
                                    output_tensor->rank, output_tensor->dimensions,
                                    lwm_read_i32(params + 4));
    }
    case LW_OP_CONV_TRANSPOSE: {
        int32_t kernel[2] = {lwm_read_i32(params + 8), lwm_read_i32(params + 12)};
        int32_t strides[2] = {lwm_read_i32(params + 16), lwm_read_i32(params + 20)};
        int32_t dilations[2] = {lwm_read_i32(params + 24), lwm_read_i32(params + 28)};
        int32_t pads[4] = {lwm_read_i32(params + 32), lwm_read_i32(params + 36),
                           lwm_read_i32(params + 40), lwm_read_i32(params + 44)};
        if (input_count != 2u && input_count != 3u) {
            return LW_STATUS_INVALID_SHAPE;
        }
        return lw_scalar_conv_transpose2d_f32(
            inputs[0], inputs[1], input_count == 3u ? inputs[2] : NULL,
            input_count == 3u ? (uint32_t)tensor_element_count(input_tensors[2]) : 0u, output,
            input_tensors[0]->dimensions, input_tensors[1]->dimensions, output_tensor->dimensions,
            kernel, strides, dilations, pads, lwm_read_u32(params + 4));
    }
    case LW_OP_MAX_POOL: {
        int32_t kernel[2] = {lwm_read_i32(params + 8), lwm_read_i32(params + 12)};
        int32_t strides[2] = {lwm_read_i32(params + 16), lwm_read_i32(params + 20)};
        int32_t pads[4] = {lwm_read_i32(params + 24), lwm_read_i32(params + 28),
                           lwm_read_i32(params + 32), lwm_read_i32(params + 36)};
        return input_count == 1u
                   ? lw_scalar_max_pool2d_f32(inputs[0], output, input_tensors[0]->dimensions,
                                              output_tensor->dimensions, kernel, strides, pads,
                                              lwm_read_u32(params + 40))
                   : LW_STATUS_INVALID_SHAPE;
    }
    case LW_OP_RESIZE: {
        float scales[LW_MAX_DIMS];
        uint32_t scale_count = lwm_read_u16(params + 2);
        if (input_count != 1u || scale_count != input_tensors[0]->rank) {
            return LW_STATUS_INVALID_SHAPE;
        }
        for (index = 0u; index < scale_count; ++index) {
            scales[index] = read_f32(params + 4u + index * 4u);
        }
        return lw_scalar_resize_nearest_f32(inputs[0], output, input_tensors[0]->rank,
                                            input_tensors[0]->dimensions, output_tensor->dimensions,
                                            scales);
    }
    case LW_OP_SIGMOID:
        return input_count == 1u ? lw_scalar_sigmoid_f32(inputs[0], output,
                                                         tensor_element_count(input_tensors[0]))
                                 : LW_STATUS_INVALID_SHAPE;
    default:
        return LW_STATUS_UNSUPPORTED;
    }
}

static uint32_t profile_conv_class(const lw_session* session, const uint8_t* node) {
    const lw_model* model = session->model;
    uint64_t param_offset = lwm_read_u64(node + 56);
    const uint8_t* params = model->bytes + (size_t)param_offset;
    const lw_runtime_tensor* input = &session->tensors[lwm_read_u32(node + 8u)];
    const lw_runtime_tensor* weights = &session->tensors[lwm_read_u32(node + 12u)];
    int32_t kernel_height = lwm_read_i32(params + 8u);
    int32_t kernel_width = lwm_read_i32(params + 12u);
    int32_t stride_height = lwm_read_i32(params + 16u);
    int32_t stride_width = lwm_read_i32(params + 20u);
    uint32_t group = lwm_read_u32(params + 4u);

    if (kernel_height == 1 && kernel_width == 1) {
        return LW_EXECUTION_PROFILE_CONV_1X1;
    }
    if (kernel_height == 3 && kernel_width == 3 && input->rank == 4u && weights->rank == 4u &&
        input->dimensions[1] > 0 && group == (uint32_t)input->dimensions[1] &&
        weights->dimensions[0] == input->dimensions[1]) {
        return LW_EXECUTION_PROFILE_CONV_DEPTHWISE_3X3;
    }
    if (kernel_height == 3 && kernel_width == 3 && stride_height == 2 && stride_width == 2) {
        return LW_EXECUTION_PROFILE_CONV_STRIDE2_3X3;
    }
    if (kernel_height == 3 && kernel_width == 3) {
        return LW_EXECUTION_PROFILE_CONV_3X3;
    }
    return LW_EXECUTION_PROFILE_CONV_OTHER;
}

static lw_status execute_session_f32(lw_session* session, const float* input,
                                     uint64_t input_element_count, float* output,
                                     uint64_t output_element_count, lw_execution_profile* profile,
                                     lw_error* error) {
    const lw_model* model;
    uint32_t graph_input_index;
    uint32_t graph_output_index;
    uint32_t node_index;
    lw_status status;
    char message[LW_ERROR_MESSAGE_CAPACITY];

    if (session == NULL || input == NULL || output == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "session, input, and output are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    model = session->model;
    if (model->info.input_count != 1u || model->info.output_count != 1u) {
        lw_set_error(error, LW_STATUS_UNSUPPORTED,
                     "private executor currently requires one input and one output");
        return LW_STATUS_UNSUPPORTED;
    }
    graph_input_index = lwm_read_u32(model->bytes + (size_t)model->input_offset);
    graph_output_index = lwm_read_u32(model->bytes + (size_t)model->output_offset);
    if (session->tensors[graph_input_index].dtype != LW_DTYPE_F32 ||
        session->tensors[graph_output_index].dtype != LW_DTYPE_F32 ||
        input_element_count != tensor_element_count(&session->tensors[graph_input_index]) ||
        output_element_count != tensor_element_count(&session->tensors[graph_output_index])) {
        lw_set_error(error, LW_STATUS_INVALID_SHAPE,
                     "input or output element count does not match the session");
        return LW_STATUS_INVALID_SHAPE;
    }
    for (node_index = 0u; node_index < model->info.node_count; ++node_index) {
        const uint8_t* node =
            model->bytes + (size_t)model->node_offset + (size_t)node_index * LWM_V0_NODE_SIZE;
        uint32_t operation = (uint32_t)lwm_read_u16(node);
        uint64_t started = 0u;
        if (profile != NULL) {
            started = profile->clock(profile->clock_context);
        }
        status = dispatch_node(session, node, graph_input_index, input);
        if (profile != NULL && operation < LW_EXECUTION_PROFILE_OPERATOR_CAPACITY) {
            uint64_t finished = profile->clock(profile->clock_context);
            if (finished >= started) {
                uint64_t elapsed = finished - started;
                if (profile->operator_nanoseconds[operation] <= UINT64_MAX - elapsed &&
                    profile->operator_invocations[operation] != UINT64_MAX) {
                    profile->operator_nanoseconds[operation] += elapsed;
                    profile->operator_invocations[operation] += 1u;
                }
                if (node_index < LW_EXECUTION_PROFILE_NODE_CAPACITY &&
                    profile->node_nanoseconds[node_index] <= UINT64_MAX - elapsed &&
                    profile->node_invocations[node_index] != UINT64_MAX) {
                    profile->node_nanoseconds[node_index] += elapsed;
                    profile->node_invocations[node_index] += 1u;
                }
                if (operation == LW_OP_CONV) {
                    uint32_t conv_class = profile_conv_class(session, node);
                    if (profile->conv_class_nanoseconds[conv_class] <= UINT64_MAX - elapsed &&
                        profile->conv_class_invocations[conv_class] != UINT64_MAX) {
                        profile->conv_class_nanoseconds[conv_class] += elapsed;
                        profile->conv_class_invocations[conv_class] += 1u;
                    }
                }
            }
        }
        if (status != LW_STATUS_OK) {
#if defined(_MSC_VER)
            (void)sprintf_s(message, sizeof(message), "node %u (operator %u) failed: %s",
                            node_index, (uint32_t)lwm_read_u16(node), lw_status_string(status));
#else
            (void)snprintf(message, sizeof(message), "node %u (operator %u) failed: %s", node_index,
                           (uint32_t)lwm_read_u16(node), lw_status_string(status));
#endif
            lw_set_error(error, status, message);
            return status;
        }
    }
    memcpy(output, tensor_output_data(session, graph_output_index),
           (size_t)session->tensors[graph_output_index].byte_size);
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}

lw_status lw_execute_session_f32(lw_session* session, const float* input,
                                 uint64_t input_element_count, float* output,
                                 uint64_t output_element_count, lw_error* error) {
    return execute_session_f32(session, input, input_element_count, output, output_element_count,
                               NULL, error);
}

lw_status lw_execute_session_f32_profiled(lw_session* session, const float* input,
                                          uint64_t input_element_count, float* output,
                                          uint64_t output_element_count,
                                          lw_execution_profile* profile, lw_error* error) {
    if (profile == NULL || profile->struct_size != sizeof(*profile) || profile->reserved != 0u ||
        profile->clock == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "an initialized execution profile and clock are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    return execute_session_f32(session, input, input_element_count, output, output_element_count,
                               profile, error);
}
