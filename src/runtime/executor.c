#include "executor_internal.h"

/*
 * Small graph interpreter for the operator subset emitted by the converter.
 * Constants point into validated model bytes; intermediate tensors point into
 * the preplanned workspace. Dispatch must stay allocation-free during a run.
 */

#include "lwm_read.h"
#include "packed_conv_internal.h"
#include "packed_conv3x3_internal.h"
#include "parallel_internal.h"
#include "scalar_kernels.h"
#include "session_internal.h"
#include "cpu_features.h"
#include "simd_kernels.h"

#include <stdio.h>
#include <string.h>

#define LW_PARALLEL_CONV_MIN_MULTIPLY_ADDS UINT64_C(8000000)
#define LW_PARALLEL_CONV_TRANSPOSE_MIN_MULTIPLY_ADDS UINT64_C(8000000)

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
    LW_OP_SIGMOID = 21,
    LW_OP_SUB = 22,
    LW_OP_SQRT = 23,
    LW_OP_POW = 24,
    LW_OP_SLICE = 25
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

typedef struct lw_parallel_conv_context {
    const float* input;
    const float* weights;
    const float* bias;
    float* output;
    int32_t input_dimensions[4];
    int32_t weight_dimensions[4];
    int32_t output_dimensions[4];
    int32_t kernel[2];
    int32_t strides[2];
    int32_t dilations[2];
    int32_t pads[4];
    uint32_t groups;
    uint32_t packed;
    lw_packed_conv1x1_kernel_fn conv1x1_kernel;
    lw_packed_conv3x3_kernel_fn conv3x3_kernel;
    lw_status statuses[LW_PARALLEL_MAX_WORKERS];
} lw_parallel_conv_context;

typedef struct lw_parallel_conv_transpose_context {
    const float* input;
    const float* weights;
    const float* bias;
    float* output;
    int32_t input_dimensions[4];
    int32_t weight_dimensions[4];
    int32_t output_dimensions[4];
    int32_t kernel[2];
    int32_t strides[2];
    int32_t dilations[2];
    int32_t pads[4];
    uint32_t groups;
    uint32_t bias_count;
    lw_simd_level simd_level;
} lw_parallel_conv_transpose_context;

static uint64_t multiply_saturated(uint64_t left, uint64_t right) {
    return right != 0u && left > UINT64_MAX / right ? UINT64_MAX : left * right;
}

static uint32_t parallel_conv_worker_count(const lw_session* session,
                                           const lw_parallel_conv_context* context) {
    uint32_t output_channels = (uint32_t)context->output_dimensions[1];
    uint32_t maximum_workers;
    uint64_t work;
    int depthwise;
    if (session->intra_op_thread_count <= 1u || context->input_dimensions[0] != 1 ||
        context->output_dimensions[0] != 1 || output_channels == 0u) {
        return 1u;
    }
    depthwise = context->groups == (uint32_t)context->input_dimensions[1] &&
                context->groups == output_channels && context->weight_dimensions[1] == 1;
    if (context->groups != 1u && !depthwise) {
        return 1u;
    }
    if (context->packed != 0u) {
        const uint32_t output_tile =
            context->packed == LW_PREPARED_CONSTANT_CONV3X3_STRIDE2_PACKED8
                ? LW_PACKED_CONV3X3_STRIDE2_OUTPUT_TILE
                : LW_PACKED_CONV1X1_OUTPUT_TILE;
        if (output_channels % output_tile != 0u) {
            return 1u;
        }
        maximum_workers = output_channels / output_tile;
    } else {
        maximum_workers = output_channels;
    }
    if (maximum_workers > session->intra_op_thread_count) {
        maximum_workers = session->intra_op_thread_count;
    }
    if (maximum_workers < 2u) {
        return 1u;
    }
    work = multiply_saturated(output_channels, (uint32_t)context->output_dimensions[2]);
    work = multiply_saturated(work, (uint32_t)context->output_dimensions[3]);
    work = multiply_saturated(work, (uint32_t)context->weight_dimensions[1]);
    work = multiply_saturated(work, (uint32_t)context->kernel[0]);
    work = multiply_saturated(work, (uint32_t)context->kernel[1]);
    return work >= LW_PARALLEL_CONV_MIN_MULTIPLY_ADDS ? maximum_workers : 1u;
}

static void execute_parallel_conv_slice(void* opaque, uint32_t worker_index,
                                        uint32_t worker_count) {
    lw_parallel_conv_context* context = (lw_parallel_conv_context*)opaque;
    uint32_t output_channels = (uint32_t)context->output_dimensions[1];
    uint32_t item_size =
        context->packed == LW_PREPARED_CONSTANT_CONV3X3_STRIDE2_PACKED8
            ? LW_PACKED_CONV3X3_STRIDE2_OUTPUT_TILE
            : (context->packed != 0u ? LW_PACKED_CONV1X1_OUTPUT_TILE : 1u);
    uint32_t item_count = output_channels / item_size;
    uint32_t channel_begin =
        (uint32_t)(((uint64_t)item_count * worker_index) / worker_count) * item_size;
    uint32_t channel_end =
        (uint32_t)(((uint64_t)item_count * (worker_index + 1u)) / worker_count) * item_size;
    uint32_t channel_count = channel_end - channel_begin;
    uint64_t input_plane =
        (uint64_t)(uint32_t)context->input_dimensions[2] * (uint32_t)context->input_dimensions[3];
    uint64_t output_plane =
        (uint64_t)(uint32_t)context->output_dimensions[2] * (uint32_t)context->output_dimensions[3];
    uint64_t kernel_plane = (uint64_t)(uint32_t)context->kernel[0] * (uint32_t)context->kernel[1];
    int32_t input_dimensions[4];
    int32_t weight_dimensions[4];
    int32_t output_dimensions[4];
    const float* input = context->input;
    const float* weights;
    const float* bias = context->bias == NULL ? NULL : context->bias + channel_begin;
    float* output = context->output + (size_t)((uint64_t)channel_begin * output_plane);
    uint32_t groups = context->groups;
    memcpy(input_dimensions, context->input_dimensions, sizeof(input_dimensions));
    memcpy(weight_dimensions, context->weight_dimensions, sizeof(weight_dimensions));
    memcpy(output_dimensions, context->output_dimensions, sizeof(output_dimensions));
    output_dimensions[1] = (int32_t)channel_count;
    weight_dimensions[0] = (int32_t)channel_count;
    if (context->packed == LW_PREPARED_CONSTANT_CONV1X1_PACKED4) {
        if (context->conv1x1_kernel != NULL) {
            uint32_t input_channels = (uint32_t)input_dimensions[1];
            weights = context->weights + (size_t)((uint64_t)channel_begin * input_channels);
            context->conv1x1_kernel(input, weights, bias, output, input_dimensions, output_dimensions);
            context->statuses[worker_index] = LW_STATUS_OK;
            return;
        }
        uint32_t input_channels = (uint32_t)input_dimensions[1];
        weights = context->weights + (size_t)((uint64_t)channel_begin * input_channels);
        lw_packed_conv1x1_f32(input, weights, bias, output, input_dimensions, output_dimensions);
        context->statuses[worker_index] = LW_STATUS_OK;
        return;
    }
    if (context->packed == LW_PREPARED_CONSTANT_CONV3X3_STRIDE2_PACKED8) {
        const uint32_t input_channels = (uint32_t)input_dimensions[1];
        weights = context->weights +
                  (size_t)((uint64_t)channel_begin * input_channels * 9u);
        if (context->conv3x3_kernel != NULL) {
            context->conv3x3_kernel(input, weights, bias, output, input_dimensions,
                                    output_dimensions);
            context->statuses[worker_index] = LW_STATUS_OK;
            return;
        }
        lw_packed_conv3x3_stride2_pad1_f32(
            input, weights, bias, output, input_dimensions, output_dimensions);
        context->statuses[worker_index] = LW_STATUS_OK;
        return;
    }
    if (groups == 1u) {
        weights = context->weights +
                  (size_t)((uint64_t)channel_begin * (uint32_t)weight_dimensions[1] * kernel_plane);
    } else {
        input += (size_t)((uint64_t)channel_begin * input_plane);
        weights = context->weights + (size_t)((uint64_t)channel_begin * kernel_plane);
        input_dimensions[1] = (int32_t)channel_count;
        groups = channel_count;
    }
    context->statuses[worker_index] = lw_scalar_conv2d_f32(
        input, weights, bias, bias == NULL ? 0u : channel_count, output, input_dimensions,
        weight_dimensions, output_dimensions, context->kernel, context->strides, context->dilations,
        context->pads, groups);
}

static lw_status dispatch_parallel_conv(lw_session* session, const float* input,
                                        const float* weights, const float* bias, float* output,
                                        const int32_t input_dimensions[4],
                                        const int32_t weight_dimensions[4],
                                        const int32_t output_dimensions[4], const int32_t kernel[2],
                                        const int32_t strides[2], const int32_t dilations[2],
                                        const int32_t pads[4], uint32_t groups, uint32_t packed,
                                        lw_packed_conv1x1_kernel_fn conv1x1_kernel,
                                        lw_packed_conv3x3_kernel_fn conv3x3_kernel,
                                        lw_execution_profile* profile) {
    lw_parallel_conv_context context;
    uint32_t worker_count;
    uint32_t worker_index;
    memset(&context, 0, sizeof(context));
    context.input = input;
    context.weights = weights;
    context.bias = bias;
    context.output = output;
    memcpy(context.input_dimensions, input_dimensions, sizeof(context.input_dimensions));
    memcpy(context.weight_dimensions, weight_dimensions, sizeof(context.weight_dimensions));
    memcpy(context.output_dimensions, output_dimensions, sizeof(context.output_dimensions));
    memcpy(context.kernel, kernel, sizeof(context.kernel));
    memcpy(context.strides, strides, sizeof(context.strides));
    memcpy(context.dilations, dilations, sizeof(context.dilations));
    memcpy(context.pads, pads, sizeof(context.pads));
    context.groups = groups;
    context.packed = packed;
    context.conv1x1_kernel = conv1x1_kernel;
    context.conv3x3_kernel = conv3x3_kernel;
    worker_count = parallel_conv_worker_count(session, &context);
    if (profile != NULL && worker_count < LW_EXECUTION_PROFILE_THREAD_HISTOGRAM_CAPACITY &&
        profile->conv_thread_histogram[worker_count] != UINT64_MAX) {
        ++profile->conv_thread_histogram[worker_count];
    }
    if (worker_count <= 1u) {
        return LW_STATUS_UNSUPPORTED;
    }
    lw_thread_pool_run(session->thread_pool, worker_count, execute_parallel_conv_slice, &context);
    for (worker_index = 0u; worker_index < worker_count; ++worker_index) {
        if (context.statuses[worker_index] != LW_STATUS_OK) {
            return context.statuses[worker_index];
        }
    }
    return LW_STATUS_OK;
}

static uint32_t parallel_conv_transpose_worker_count(
    const lw_session* session, const lw_parallel_conv_transpose_context* context) {
    uint32_t maximum_workers;
    uint32_t output_channels;
    uint64_t work;
    if (session->intra_op_thread_count <= 1u || context->groups != 1u ||
        context->input_dimensions[0] != 1 || context->output_dimensions[0] != 1 ||
        context->kernel[0] != 2 || context->kernel[1] != 2 || context->strides[0] != 2 ||
        context->strides[1] != 2 || context->dilations[0] != 1 ||
        context->dilations[1] != 1 || context->pads[0] != 0 || context->pads[1] != 0 ||
        context->pads[2] != 0 || context->pads[3] != 0 ||
        context->weight_dimensions[0] != context->input_dimensions[1] ||
        context->weight_dimensions[1] != context->output_dimensions[1] ||
        context->weight_dimensions[2] != 2 || context->weight_dimensions[3] != 2 ||
        context->output_dimensions[2] != context->input_dimensions[2] * 2 ||
        context->output_dimensions[3] != context->input_dimensions[3] * 2 ||
        (context->bias == NULL ? context->bias_count != 0u
                               : context->bias_count !=
                                     (uint32_t)context->output_dimensions[1])) {
        return 1u;
    }
    output_channels = (uint32_t)context->output_dimensions[1];
    maximum_workers = output_channels;
    if (maximum_workers > session->intra_op_thread_count) {
        maximum_workers = session->intra_op_thread_count;
    }
    if (maximum_workers < 2u) {
        return 1u;
    }
    work = multiply_saturated(output_channels, (uint32_t)context->input_dimensions[1]);
    work = multiply_saturated(work, (uint32_t)context->input_dimensions[2]);
    work = multiply_saturated(work, (uint32_t)context->input_dimensions[3]);
    work = multiply_saturated(work, 4u);
    return work >= LW_PARALLEL_CONV_TRANSPOSE_MIN_MULTIPLY_ADDS ? maximum_workers : 1u;
}

static void execute_parallel_conv_transpose_slice(void* opaque, uint32_t worker_index,
                                                  uint32_t worker_count) {
    lw_parallel_conv_transpose_context* context =
        (lw_parallel_conv_transpose_context*)opaque;
    const uint32_t output_channels = (uint32_t)context->output_dimensions[1];
    const uint32_t channel_begin =
        (uint32_t)(((uint64_t)output_channels * worker_index) / worker_count);
    const uint32_t channel_end =
        (uint32_t)(((uint64_t)output_channels * (worker_index + 1u)) / worker_count);
    if (lw_simd_level_is_avx2(context->simd_level)) {
        lw_avx2_conv_transpose2x2_stride2_range_f32(
            context->input, context->weights, context->bias, context->output,
            context->input_dimensions, context->output_dimensions, channel_begin, channel_end);
    } else if (lw_simd_level_is_neon(context->simd_level)) {
        lw_neon_conv_transpose2x2_stride2_range_f32(
            context->input, context->weights, context->bias, context->output,
            context->input_dimensions, context->output_dimensions, channel_begin, channel_end);
    } else if (lw_simd_level_is_sse2(context->simd_level)) {
        lw_sse2_conv_transpose2x2_stride2_range_f32(
            context->input, context->weights, context->bias, context->output,
            context->input_dimensions, context->output_dimensions, channel_begin, channel_end);
    } else {
        lw_scalar_conv_transpose2x2_stride2_range_f32(
            context->input, context->weights, context->bias, context->output,
            context->input_dimensions, context->output_dimensions, channel_begin, channel_end);
    }
}

static lw_status dispatch_parallel_conv_transpose(
    lw_session* session, const float* input, const float* weights, const float* bias,
    uint32_t bias_count, float* output, const int32_t input_dimensions[4],
    const int32_t weight_dimensions[4], const int32_t output_dimensions[4],
    const int32_t kernel[2], const int32_t strides[2], const int32_t dilations[2],
    const int32_t pads[4], uint32_t groups, lw_simd_level simd_level,
    lw_execution_profile* profile) {
    lw_parallel_conv_transpose_context context;
    uint32_t worker_count;
    memset(&context, 0, sizeof(context));
    context.input = input;
    context.weights = weights;
    context.bias = bias;
    context.output = output;
    memcpy(context.input_dimensions, input_dimensions, sizeof(context.input_dimensions));
    memcpy(context.weight_dimensions, weight_dimensions, sizeof(context.weight_dimensions));
    memcpy(context.output_dimensions, output_dimensions, sizeof(context.output_dimensions));
    memcpy(context.kernel, kernel, sizeof(context.kernel));
    memcpy(context.strides, strides, sizeof(context.strides));
    memcpy(context.dilations, dilations, sizeof(context.dilations));
    memcpy(context.pads, pads, sizeof(context.pads));
    context.groups = groups;
    context.bias_count = bias_count;
    context.simd_level = simd_level;
    worker_count = parallel_conv_transpose_worker_count(session, &context);
    if (profile != NULL && worker_count < LW_EXECUTION_PROFILE_THREAD_HISTOGRAM_CAPACITY &&
        profile->conv_transpose_thread_histogram[worker_count] != UINT64_MAX) {
        ++profile->conv_transpose_thread_histogram[worker_count];
    }
    if (worker_count <= 1u) {
        return LW_STATUS_UNSUPPORTED;
    }
    lw_thread_pool_run(session->thread_pool, worker_count,
                       execute_parallel_conv_transpose_slice, &context);
    return LW_STATUS_OK;
}

static const lw_prepared_constant* prepared_constant_for_node(
    const lw_session* session, uint32_t node_index, lw_execution_profile* profile) {
#if !defined(LW_EXPERIMENTAL_PREPARED_EXECUTION)
    (void)profile;
#endif
#if defined(LW_EXPERIMENTAL_PREPARED_EXECUTION)
    if (session->execution_nodes != NULL && node_index < session->execution_node_count) {
        const lw_prepared_constant* prepared =
            session->execution_nodes[node_index].prepared_constant;
        if (profile != NULL && profile->prepared_binding_lookups != UINT64_MAX) {
            ++profile->prepared_binding_lookups;
            if (prepared != NULL) {
                if (profile->prepared_binding_hits != UINT64_MAX) {
                    ++profile->prepared_binding_hits;
                }
            } else if (profile->prepared_binding_fallbacks != UINT64_MAX) {
                ++profile->prepared_binding_fallbacks;
            }
        }
        return prepared;
    }
#endif
    if (session->prepared_constants == NULL) {
        return NULL;
    }
    return &session->prepared_constants[node_index];
}

static lw_status dispatch_node(lw_session* session, const uint8_t* node, uint32_t node_index,
                               uint32_t graph_input_index, const float* graph_input,
                               lw_simd_level simd_level, lw_execution_profile* profile) {
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
    const lw_prepared_constant* prepared_constant;
    lw_packed_conv1x1_kernel_fn bound_kernel = NULL;
    lw_packed_conv3x3_kernel_fn bound_conv3x3_kernel = NULL;

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
    prepared_constant = prepared_constant_for_node(session, node_index, profile);
#if defined(LW_EXPERIMENTAL_PREPARED_EXECUTION)
    if (session->execution_nodes != NULL && node_index < session->execution_node_count &&
        session->execution_nodes[node_index].execution_kind == LW_BOUND_EXEC_CONV1X1_PACKED) {
        bound_kernel = session->execution_nodes[node_index].data.conv1x1.kernel;
    } else if (session->execution_nodes != NULL &&
               node_index < session->execution_node_count &&
               session->execution_nodes[node_index].execution_kind == LW_BOUND_EXEC_CONV3X3_PACKED) {
        bound_conv3x3_kernel = session->execution_nodes[node_index].data.conv3x3.kernel;
    }
#endif

    /* Parameters were structurally validated at model-load time. Each kernel
     * still validates runtime-dependent shapes before reading tensor data. */
    switch (op) {
    case LW_OP_CONV: {
        int32_t kernel[2] = {lwm_read_i32(params + 8), lwm_read_i32(params + 12)};
        int32_t strides[2] = {lwm_read_i32(params + 16), lwm_read_i32(params + 20)};
        int32_t dilations[2] = {lwm_read_i32(params + 24), lwm_read_i32(params + 28)};
        int32_t pads[4] = {lwm_read_i32(params + 32), lwm_read_i32(params + 36),
                           lwm_read_i32(params + 40), lwm_read_i32(params + 44)};
        uint32_t groups = lwm_read_u32(params + 4);
        lw_status parallel_status;
        if (input_count != 2u && input_count != 3u) {
            return LW_STATUS_INVALID_SHAPE;
        }
        {
            if (prepared_constant != NULL &&
                prepared_constant->kind == LW_PREPARED_CONSTANT_CONV1X1_PACKED4) {
                const float* packed_weights =
                    (const float*)(const void*)(session->packed_weights +
                                                (size_t)prepared_constant->packed_weight_offset);
                if (profile != NULL && profile->packed_conv1x1_invocations != UINT64_MAX) {
                    ++profile->packed_conv1x1_invocations;
                }
                parallel_status = dispatch_parallel_conv(
                    session, inputs[0], packed_weights,
                    input_count == 3u ? inputs[2] : NULL, output,
                    input_tensors[0]->dimensions, input_tensors[1]->dimensions,
                    output_tensor->dimensions, kernel, strides, dilations, pads, groups, 1u,
                    bound_kernel, NULL, profile);
                if (parallel_status == LW_STATUS_OK) {
                    return LW_STATUS_OK;
                }
                if (bound_kernel != NULL) {
                    bound_kernel(inputs[0], packed_weights,
                                 input_count == 3u ? inputs[2] : NULL, output,
                                 input_tensors[0]->dimensions, output_tensor->dimensions);
                } else {
                    lw_packed_conv1x1_f32(
                        inputs[0], packed_weights, input_count == 3u ? inputs[2] : NULL,
                        output, input_tensors[0]->dimensions, output_tensor->dimensions);
                }
                return LW_STATUS_OK;
            }
        }
        {
            if (prepared_constant != NULL &&
                prepared_constant->kind == LW_PREPARED_CONSTANT_CONV3X3_STRIDE2_PACKED8) {
                const float* packed_weights =
                    (const float*)(const void*)(session->packed_weights +
                                                (size_t)prepared_constant->packed_weight_offset);
                if (profile != NULL &&
                    profile->packed_conv3x3_stride2_invocations != UINT64_MAX) {
                    ++profile->packed_conv3x3_stride2_invocations;
                }
                parallel_status = dispatch_parallel_conv(
                    session, inputs[0], packed_weights,
                    input_count == 3u ? inputs[2] : NULL, output,
                    input_tensors[0]->dimensions, input_tensors[1]->dimensions,
                    output_tensor->dimensions, kernel, strides, dilations, pads, groups,
                    LW_PREPARED_CONSTANT_CONV3X3_STRIDE2_PACKED8, NULL, bound_conv3x3_kernel, profile);
                if (parallel_status == LW_STATUS_OK) {
                    return LW_STATUS_OK;
                }
                if (bound_conv3x3_kernel != NULL) {
                    bound_conv3x3_kernel(inputs[0], packed_weights,
                                         input_count == 3u ? inputs[2] : NULL, output,
                                         input_tensors[0]->dimensions, output_tensor->dimensions);
                } else {
                    lw_packed_conv3x3_stride2_pad1_f32(
                        inputs[0], packed_weights, input_count == 3u ? inputs[2] : NULL,
                        output, input_tensors[0]->dimensions, output_tensor->dimensions);
                }
                return LW_STATUS_OK;
            }
        }
        if (profile != NULL && profile->unpacked_conv_invocations != UINT64_MAX) {
            ++profile->unpacked_conv_invocations;
        }
        parallel_status = dispatch_parallel_conv(
            session, inputs[0], inputs[1], input_count == 3u ? inputs[2] : NULL, output,
            input_tensors[0]->dimensions, input_tensors[1]->dimensions, output_tensor->dimensions,
            kernel, strides, dilations, pads, groups, 0u, NULL, NULL, profile);
        if (parallel_status == LW_STATUS_OK) {
            return LW_STATUS_OK;
        }
        return lw_scalar_conv2d_f32(
            inputs[0], inputs[1], input_count == 3u ? inputs[2] : NULL,
            input_count == 3u ? (uint32_t)tensor_element_count(input_tensors[2]) : 0u, output,
            input_tensors[0]->dimensions, input_tensors[1]->dimensions, output_tensor->dimensions,
            kernel, strides, dilations, pads, groups);
    }
    case LW_OP_ADD:
    case LW_OP_MUL:
    case LW_OP_DIV:
    case LW_OP_SUB:
    case LW_OP_POW: {
        lw_scalar_binary_op operation =
            op == LW_OP_ADD ? LW_SCALAR_BINARY_ADD
                            : (op == LW_OP_MUL ? LW_SCALAR_BINARY_MUL
                                                : (op == LW_OP_DIV ? LW_SCALAR_BINARY_DIV
                                                                    : (op == LW_OP_SUB
                                                                           ? LW_SCALAR_BINARY_SUB
                                                                           : LW_SCALAR_BINARY_POW)));
        if (input_count != 2u) {
            return LW_STATUS_INVALID_SHAPE;
        }
        return lw_scalar_binary_f32(operation, inputs[0], input_tensors[0]->rank,
                                    input_tensors[0]->dimensions, inputs[1], input_tensors[1]->rank,
                                    input_tensors[1]->dimensions, output, output_tensor->rank,
                                    output_tensor->dimensions);
    }
    case LW_OP_ERF: {
        uint64_t element_count;
        if (input_count != 1u) {
            return LW_STATUS_INVALID_SHAPE;
        }
        element_count = tensor_element_count(input_tensors[0]);
        if (lw_simd_level_is_avx2(simd_level)) {
            lw_avx2_erf_f32(inputs[0], output, element_count);
            return LW_STATUS_OK;
        }
#if defined(__EMSCRIPTEN__)
        if (lw_simd_level_is_sse2(simd_level)) {
            lw_wasm128_erf_f32(inputs[0], output, element_count);
            return LW_STATUS_OK;
        }
#endif
        return lw_scalar_erf_f32(inputs[0], output, element_count);
    }
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
        if (input_count != 2u || rank < 2u || input_tensors[1]->rank < 2u) {
            return LW_STATUS_UNSUPPORTED;
        }
        if (input_tensors[1]->rank != 2u) {
            if (profile != NULL && profile->unpacked_matmul_invocations != UINT64_MAX) {
                ++profile->unpacked_matmul_invocations;
            }
            return lw_scalar_matmul_f32(inputs[0], inputs[1], output, rank,
                                        input_tensors[0]->dimensions, input_tensors[1]->rank,
                                        input_tensors[1]->dimensions, output_tensor->rank,
                                        output_tensor->dimensions);
        }
        for (index = 0u; index + 2u < rank; ++index) {
            batch_count *= (uint32_t)input_tensors[0]->dimensions[index];
        }
        if (batch_count > UINT32_MAX) {
            return LW_STATUS_OUT_OF_BOUNDS;
        }
        {
            if (lw_simd_level_is_avx2(simd_level) && prepared_constant != NULL &&
                prepared_constant->kind == LW_PREPARED_CONSTANT_MATMUL_PACKED16) {
                if (profile != NULL && profile->packed_matmul_invocations != UINT64_MAX) {
                    ++profile->packed_matmul_invocations;
                }
            const float* packed_weights =
                (const float*)(const void*)(session->packed_weights +
                                            (size_t)prepared_constant->packed_weight_offset);
                lw_avx2_packed_matmul_shared_f32(inputs[0], packed_weights, output,
                                                 (uint32_t)batch_count,
                                                 (uint32_t)input_tensors[0]->dimensions[rank - 2u],
                                                 (uint32_t)input_tensors[0]->dimensions[rank - 1u],
                                                 (uint32_t)input_tensors[1]->dimensions[1]);
                return LW_STATUS_OK;
            }
        }
        if (profile != NULL && profile->unpacked_matmul_invocations != UINT64_MAX) {
            ++profile->unpacked_matmul_invocations;
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
        if (dispatch_parallel_conv_transpose(
                session, inputs[0], inputs[1], input_count == 3u ? inputs[2] : NULL,
                input_count == 3u ? (uint32_t)tensor_element_count(input_tensors[2]) : 0u,
                output, input_tensors[0]->dimensions, input_tensors[1]->dimensions,
                output_tensor->dimensions, kernel, strides, dilations, pads,
                lwm_read_u32(params + 4), simd_level, profile) == LW_STATUS_OK) {
            return LW_STATUS_OK;
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
    case LW_OP_SQRT:
        return input_count == 1u ? lw_scalar_sqrt_f32(inputs[0], output,
                                                      tensor_element_count(input_tensors[0]))
                                 : LW_STATUS_INVALID_SHAPE;
    case LW_OP_SLICE: {
        int32_t starts[LW_MAX_DIMS];
        int32_t ends[LW_MAX_DIMS];
        int32_t axes[LW_MAX_DIMS];
        int32_t steps[LW_MAX_DIMS];
        uint32_t slice_count = lwm_read_u16(params + 2u);
        uint32_t slice_index;
        if (input_count != 1u || slice_count == 0u || slice_count > LW_MAX_DIMS) {
            return LW_STATUS_INVALID_SHAPE;
        }
        for (slice_index = 0u; slice_index < slice_count; ++slice_index) {
            starts[slice_index] = lwm_read_i32(params + 4u + slice_index * 4u);
            ends[slice_index] = lwm_read_i32(params + 36u + slice_index * 4u);
            axes[slice_index] = lwm_read_i32(params + 68u + slice_index * 4u);
            steps[slice_index] = lwm_read_i32(params + 100u + slice_index * 4u);
        }
        return lw_scalar_slice_f32(inputs[0], output, input_tensors[0]->rank,
                                   input_tensors[0]->dimensions, output_tensor->dimensions,
                                   slice_count, starts, ends, axes, steps);
    }
    default:
        return LW_STATUS_UNSUPPORTED;
    }
}

static int constant_scalar_f32(const lw_session* session, uint32_t tensor_index,
                               uint32_t graph_input_index, const float* graph_input,
                               float expected) {
    const lw_runtime_tensor* tensor = &session->tensors[tensor_index];
    const float* value;
    if (tensor->dtype != LW_DTYPE_F32 || tensor->byte_size != sizeof(float) ||
        (tensor->flags & LWM_V0_TENSOR_FLAG_CONSTANT) == 0u) {
        return 0;
    }
    value = tensor_input_data(session, tensor_index, graph_input_index, graph_input);
    return value != NULL && value[0] == expected;
}

static int match_avx2_gelu(lw_session* session, uint32_t node_index, uint32_t graph_input_index,
                           const float* graph_input, const float** gelu_input, float** gelu_output,
                           uint64_t* element_count) {
    const lw_model* model = session->model;
    const uint8_t* nodes[5];
    uint32_t inputs[5][2];
    uint32_t outputs[5];
    uint32_t index;
    const lw_runtime_tensor* source;
    const lw_runtime_tensor* output;
    if (node_index > model->info.node_count || model->info.node_count - node_index < 5u) {
        return 0;
    }
    for (index = 0u; index < 5u; ++index) {
        nodes[index] = model->bytes + (size_t)model->node_offset +
                       (size_t)(node_index + index) * LWM_V0_NODE_SIZE;
        if (lwm_read_u16(nodes[index] + 2u) != (index == 1u ? 1u : 2u) ||
            lwm_read_u16(nodes[index] + 4u) != 1u) {
            return 0;
        }
        inputs[index][0] = lwm_read_u32(nodes[index] + 8u);
        inputs[index][1] = index == 1u ? UINT32_MAX : lwm_read_u32(nodes[index] + 12u);
        outputs[index] = lwm_read_u32(nodes[index] + 40u);
    }
    if (lwm_read_u16(nodes[0]) != LW_OP_DIV || lwm_read_u16(nodes[1]) != LW_OP_ERF ||
        lwm_read_u16(nodes[2]) != LW_OP_ADD || lwm_read_u16(nodes[3]) != LW_OP_MUL ||
        lwm_read_u16(nodes[4]) != LW_OP_MUL || inputs[1][0] != outputs[0] ||
        (inputs[2][0] != outputs[1] && inputs[2][1] != outputs[1]) ||
        (inputs[3][0] != inputs[0][0] && inputs[3][1] != inputs[0][0]) ||
        (inputs[3][0] != outputs[2] && inputs[3][1] != outputs[2]) ||
        (inputs[4][0] != outputs[3] && inputs[4][1] != outputs[3])) {
        return 0;
    }
    if (!constant_scalar_f32(session, inputs[0][1], graph_input_index, graph_input,
                             1.4142135381698608f) ||
        !constant_scalar_f32(session, inputs[2][0] == outputs[1] ? inputs[2][1] : inputs[2][0],
                             graph_input_index, graph_input, 1.0f) ||
        !constant_scalar_f32(session, inputs[4][0] == outputs[3] ? inputs[4][1] : inputs[4][0],
                             graph_input_index, graph_input, 0.5f)) {
        return 0;
    }
    /* Every skipped temporary must be private to this chain.  Otherwise a
     * later node could observe a value that the fused execution never wrote. */
    for (index = 0u; index < 4u; ++index) {
        if (session->tensors[outputs[index]].last_use_node != (int32_t)(node_index + index + 1u)) {
            return 0;
        }
    }
    source = &session->tensors[inputs[0][0]];
    output = &session->tensors[outputs[4]];
    if (source->dtype != LW_DTYPE_F32 || output->dtype != LW_DTYPE_F32 || source->byte_size == 0u ||
        source->byte_size != output->byte_size) {
        return 0;
    }
    for (index = 0u; index < 4u; ++index) {
        const lw_runtime_tensor* temporary = &session->tensors[outputs[index]];
        if (temporary->dtype != LW_DTYPE_F32 || temporary->byte_size != source->byte_size) {
            return 0;
        }
    }
    *gelu_input = tensor_input_data(session, inputs[0][0], graph_input_index, graph_input);
    *gelu_output = tensor_output_data(session, outputs[4]);
    *element_count = source->byte_size / sizeof(float);
    return *gelu_input != NULL && *gelu_output != NULL;
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

static void profile_fused_gelu(lw_execution_profile* profile, const lw_session* session,
                               uint32_t first_node_index, uint64_t elapsed) {
    uint32_t offset;
    if (profile == NULL) {
        return;
    }
    if (profile->fused_gelu_invocations != UINT64_MAX) {
        ++profile->fused_gelu_invocations;
    }
    /* The public profile schema has no fused-GELU operator. Attribute the
     * combined work to Erf and retain one invocation for every semantic node;
     * one-nanosecond placeholders keep existing per-node coverage checks useful. */
    for (offset = 0u; offset < 5u; ++offset) {
        uint32_t node_index = first_node_index + offset;
        const uint8_t* node = session->model->bytes + (size_t)session->model->node_offset +
                              (size_t)node_index * LWM_V0_NODE_SIZE;
        uint32_t operation = (uint32_t)lwm_read_u16(node);
        uint64_t node_elapsed = offset == 1u ? (elapsed == 0u ? 1u : elapsed) : 1u;
        if (operation < LW_EXECUTION_PROFILE_OPERATOR_CAPACITY &&
            profile->operator_nanoseconds[operation] <= UINT64_MAX - node_elapsed &&
            profile->operator_invocations[operation] != UINT64_MAX) {
            profile->operator_nanoseconds[operation] += node_elapsed;
            profile->operator_invocations[operation] += 1u;
        }
        if (node_index < LW_EXECUTION_PROFILE_NODE_CAPACITY &&
            profile->node_nanoseconds[node_index] <= UINT64_MAX - node_elapsed &&
            profile->node_invocations[node_index] != UINT64_MAX) {
            profile->node_nanoseconds[node_index] += node_elapsed;
            profile->node_invocations[node_index] += 1u;
        }
    }
}

static lw_status execute_bound_conv1x1(lw_session* session, const lw_bound_node* bound,
                                       uint32_t graph_input_index, const float* graph_input,
                                       lw_execution_profile* profile) {
    const lw_bound_conv1x1* conv = &bound->data.conv1x1;
    const lw_runtime_tensor* input_tensor = &session->tensors[conv->input_index];
    const lw_runtime_tensor* output_tensor = &session->tensors[conv->output_index];
    const float* input = tensor_input_data(session, conv->input_index, graph_input_index, graph_input);
    const float* bias = conv->bias_index == UINT32_MAX
                           ? NULL
                           : tensor_input_data(session, conv->bias_index, graph_input_index, graph_input);
    float* output = tensor_output_data(session, conv->output_index);
    int32_t weight_dimensions[4] = {output_tensor->dimensions[1], input_tensor->dimensions[1], 1, 1};
    int32_t kernel[2] = {1, 1};
    int32_t strides[2] = {1, 1};
    int32_t dilations[2] = {1, 1};
    int32_t pads[4] = {0, 0, 0, 0};
    lw_status status;

    if (input == NULL || output == NULL || conv->kernel == NULL ||
        input_tensor->dtype != LW_DTYPE_F32 || output_tensor->dtype != LW_DTYPE_F32 ||
        (conv->bias_index != UINT32_MAX && bias == NULL)) {
        return LW_STATUS_UNSUPPORTED;
    }
    status = dispatch_parallel_conv(
        session, input, conv->packed_weights, bias, output, input_tensor->dimensions,
        weight_dimensions, output_tensor->dimensions, kernel, strides, dilations, pads, 1u,
        LW_PREPARED_CONSTANT_CONV1X1_PACKED4, conv->kernel, NULL, profile);
    if (status == LW_STATUS_OK) {
        return LW_STATUS_OK;
    }
    conv->kernel(input, conv->packed_weights, bias, output,
                 input_tensor->dimensions, output_tensor->dimensions);
    return LW_STATUS_OK;
}
static lw_status execute_bound_conv3x3(lw_session* session, const lw_bound_node* bound,
                                       uint32_t graph_input_index, const float* graph_input,
                                       lw_execution_profile* profile) {
    const lw_bound_conv3x3* conv = &bound->data.conv3x3;
    const lw_runtime_tensor* input_tensor = &session->tensors[conv->input_index];
    const lw_runtime_tensor* output_tensor = &session->tensors[conv->output_index];
    const float* input = tensor_input_data(session, conv->input_index, graph_input_index, graph_input);
    const float* bias = conv->bias_index == UINT32_MAX
                           ? NULL
                           : tensor_input_data(session, conv->bias_index, graph_input_index, graph_input);
    float* output = tensor_output_data(session, conv->output_index);
    int32_t weight_dimensions[4] = {output_tensor->dimensions[1], input_tensor->dimensions[1], 3, 3};
    int32_t kernel[2] = {3, 3};
    int32_t strides[2] = {2, 2};
    int32_t dilations[2] = {1, 1};
    int32_t pads[4] = {1, 1, 1, 1};
    lw_status status;

    if (input == NULL || output == NULL || conv->kernel == NULL ||
        input_tensor->dtype != LW_DTYPE_F32 || output_tensor->dtype != LW_DTYPE_F32 ||
        (conv->bias_index != UINT32_MAX && bias == NULL)) {
        return LW_STATUS_UNSUPPORTED;
    }
    status = dispatch_parallel_conv(
        session, input, conv->packed_weights, bias, output, input_tensor->dimensions,
        weight_dimensions, output_tensor->dimensions, kernel, strides, dilations, pads, 1u,
        LW_PREPARED_CONSTANT_CONV3X3_STRIDE2_PACKED8, NULL, conv->kernel, profile);
    if (status == LW_STATUS_OK) {
        return LW_STATUS_OK;
    }
    conv->kernel(input, conv->packed_weights, bias, output,
                 input_tensor->dimensions, output_tensor->dimensions);
    return LW_STATUS_OK;
}

static lw_status execute_session_nodes_f32(lw_session* session, const float* input,
                                           uint64_t input_element_count, uint32_t node_limit,
                                           lw_execution_profile* profile, lw_error* error) {
    const lw_model* model;
    uint32_t graph_input_index;
    uint32_t node_index;
    lw_simd_level simd_level;
    lw_status status;
    char message[LW_ERROR_MESSAGE_CAPACITY];

    if (session == NULL || input == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "session and input are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    model = session->model;
    /* The session stores one CPU capability snapshot for the complete run. */
    simd_level = session->cpu.simd;
    if (model->info.input_count != 1u || model->info.output_count != 1u) {
        lw_set_error(error, LW_STATUS_UNSUPPORTED,
                     "private executor currently requires one input and one output");
        return LW_STATUS_UNSUPPORTED;
    }
    graph_input_index = lwm_read_u32(model->bytes + (size_t)model->input_offset);
    if (session->tensors[graph_input_index].dtype != LW_DTYPE_F32 ||
        input_element_count != tensor_element_count(&session->tensors[graph_input_index]) ||
        node_limit > model->info.node_count) {
        lw_set_error(error, LW_STATUS_INVALID_SHAPE, "input shape or node limit is invalid");
        return LW_STATUS_INVALID_SHAPE;
    }
    for (node_index = 0u; node_index < node_limit; ++node_index) {
        const uint8_t* node =
            model->bytes + (size_t)model->node_offset + (size_t)node_index * LWM_V0_NODE_SIZE;
        uint32_t operation = (uint32_t)lwm_read_u16(node);
        uint64_t started = 0u;
        if (lw_simd_level_is_avx2(simd_level) && operation == LW_OP_DIV) {
            const float* gelu_input;
            float* gelu_output;
            uint64_t gelu_element_count;
            if (match_avx2_gelu(session, node_index, graph_input_index, input, &gelu_input,
                                &gelu_output, &gelu_element_count)) {
                uint64_t elapsed = 0u;
                if (profile != NULL) {
                    started = profile->clock(profile->clock_context);
                }
                lw_avx2_gelu_f32(gelu_input, gelu_output, gelu_element_count);
                if (profile != NULL) {
                    uint64_t finished = profile->clock(profile->clock_context);
                    if (finished >= started) {
                        elapsed = finished - started;
                    }
                }
                profile_fused_gelu(profile, session, node_index, elapsed);
                node_index += 4u;
                continue;
            }
        }
        if (profile != NULL) {
            started = profile->clock(profile->clock_context);
        }
#if defined(LW_EXPERIMENTAL_PREPARED_EXECUTION)
        if (session->execution_nodes != NULL && node_index < session->execution_node_count &&
            session->execution_nodes[node_index].execution_kind == LW_BOUND_EXEC_CONV1X1_PACKED) {
            status = execute_bound_conv1x1(session, &session->execution_nodes[node_index],
                                          graph_input_index, input, profile);
        } else if (session->execution_nodes != NULL &&
                   node_index < session->execution_node_count &&
                   session->execution_nodes[node_index].execution_kind == LW_BOUND_EXEC_CONV3X3_PACKED) {
            status = execute_bound_conv3x3(session, &session->execution_nodes[node_index],
                                          graph_input_index, input, profile);
        } else {
            status = dispatch_node(session, node, node_index, graph_input_index, input,
                                   simd_level, profile);
            if (profile != NULL && profile->generic_node_invocations != UINT64_MAX) {
                ++profile->generic_node_invocations;
            }
        }
        if (profile != NULL && session->execution_nodes != NULL &&
            node_index < session->execution_node_count &&
            session->execution_nodes[node_index].execution_kind != LW_BOUND_EXEC_GENERIC) {
            if (profile->prepared_binding_lookups != UINT64_MAX) {
                ++profile->prepared_binding_lookups;
            }
            if (profile->prepared_binding_hits != UINT64_MAX) {
                ++profile->prepared_binding_hits;
            }
            if (profile->prepared_node_invocations != UINT64_MAX) {
                ++profile->prepared_node_invocations;
            }
            if (session->execution_nodes[node_index].execution_kind == LW_BOUND_EXEC_CONV1X1_PACKED) {
                if (profile->prepared_conv1x1_invocations != UINT64_MAX) {
                    ++profile->prepared_conv1x1_invocations;
                }
                if (profile->packed_conv1x1_invocations != UINT64_MAX) {
                    ++profile->packed_conv1x1_invocations;
                }
            } else {
                if (profile->prepared_conv3x3_invocations != UINT64_MAX) {
                    ++profile->prepared_conv3x3_invocations;
                }
                if (profile->packed_conv3x3_stride2_invocations != UINT64_MAX) {
                    ++profile->packed_conv3x3_stride2_invocations;
                }
            }
        }
#else
        status =
            dispatch_node(session, node, node_index, graph_input_index, input, simd_level, profile);
#endif
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
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}

static lw_status execute_session_f32(lw_session* session, const float* input,
                                     uint64_t input_element_count, float* output,
                                     uint64_t output_element_count, lw_execution_profile* profile,
                                     lw_error* error) {
    uint32_t graph_output_index;
    lw_status status;
    if (session == NULL || input == NULL || output == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "session, input, and output are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    graph_output_index = lwm_read_u32(session->model->bytes +
                                      (size_t)session->model->output_offset);
    if (session->tensors[graph_output_index].dtype != LW_DTYPE_F32 ||
        output_element_count != tensor_element_count(&session->tensors[graph_output_index])) {
        lw_set_error(error, LW_STATUS_INVALID_SHAPE,
                     "output element count does not match the session");
        return LW_STATUS_INVALID_SHAPE;
    }
    status = execute_session_nodes_f32(session, input, input_element_count,
                                       session->model->info.node_count, profile, error);
    if (status != LW_STATUS_OK) {
        return status;
    }
    memcpy(output, tensor_output_data(session, graph_output_index),
           (size_t)session->tensors[graph_output_index].byte_size);
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}

static int ctc_greedy_tail(const lw_session* session, uint32_t time_steps,
                           uint32_t class_count, uint32_t* logits_index) {
    const lw_model* model;
    const uint8_t* node;
    const uint8_t* params;
    const lw_runtime_tensor* logits;
    const lw_runtime_tensor* output;
    uint32_t graph_output_index;
    uint32_t input_index;
    uint32_t output_index;
    uint32_t index;
    int32_t axis;
    if (session == NULL || time_steps == 0u || class_count == 0u) {
        return 0;
    }
    model = session->model;
    if (model->info.input_count != 1u || model->info.output_count != 1u ||
        model->info.node_count == 0u) {
        return 0;
    }
    /* This is deliberately a structural check rather than a model-name
     * special case. Unsupported or multi-output graphs retain the generic
     * executor and complete probability tensor. */
    node = model->bytes + (size_t)model->node_offset +
           (size_t)(model->info.node_count - 1u) * LWM_V0_NODE_SIZE;
    if (lwm_read_u16(node) != LW_OP_SOFTMAX || lwm_read_u16(node + 2u) != 1u ||
        lwm_read_u16(node + 4u) != 1u) {
        return 0;
    }
    input_index = lwm_read_u32(node + 8u);
    output_index = lwm_read_u32(node + 40u);
    graph_output_index = lwm_read_u32(model->bytes + (size_t)model->output_offset);
    if (output_index != graph_output_index) {
        return 0;
    }
    logits = &session->tensors[input_index];
    output = &session->tensors[output_index];
    if (logits->dtype != LW_DTYPE_F32 || output->dtype != LW_DTYPE_F32 || logits->rank < 2u ||
        logits->rank != output->rank || logits->last_use_node != (int32_t)(model->info.node_count - 1u) ||
        tensor_element_count(output) != (uint64_t)time_steps * class_count ||
        logits->dimensions[logits->rank - 1u] != (int32_t)class_count) {
        return 0;
    }
    for (index = 0u; index < logits->rank; ++index) {
        if (logits->dimensions[index] != output->dimensions[index]) {
            return 0;
        }
    }
    params = model->bytes + (size_t)lwm_read_u64(node + 56u);
    axis = lwm_read_i32(params + 4u);
    if (axis < 0) {
        axis += (int32_t)logits->rank;
    }
    if (axis != (int32_t)logits->rank - 1) {
        return 0;
    }
    if (logits_index != NULL) {
        *logits_index = input_index;
    }
    return 1;
}

int lw_session_supports_ctc_greedy_f32(const lw_session* session, uint32_t time_steps,
                                       uint32_t class_count) {
    return ctc_greedy_tail(session, time_steps, class_count, NULL);
}

typedef struct lw_packed_ctc_projection {
    uint32_t matmul_node_index;
    uint32_t add_node_index;
    const float* activation;
    const float* packed_weights;
    const float* bias;
    float* logits;
    uint32_t rows;
    uint32_t inner_dimension;
    uint32_t columns;
} lw_packed_ctc_projection;

static int match_packed_ctc_projection(lw_session* session, const float* graph_input,
                                       uint32_t logits_index, uint32_t time_steps,
                                       uint32_t class_count, lw_execution_profile* profile,
                                       lw_packed_ctc_projection* projection) {
    const lw_model* model = session->model;
    uint32_t matmul_node_index;
    uint32_t add_node_index;
    const uint8_t* matmul_node;
    const uint8_t* add_node;
    uint32_t matmul_input_index;
    uint32_t weights_index;
    uint32_t matmul_output_index;
    uint32_t bias_index;
    const lw_runtime_tensor* activation;
    const lw_runtime_tensor* weights;
    const lw_runtime_tensor* matmul_output;
    const lw_runtime_tensor* bias;
    const lw_prepared_constant* prepared;
    uint32_t graph_input_index;
    if (projection == NULL || model->info.node_count < 3u ||
        !lw_simd_level_is_avx2(session->cpu.simd) || session->prepared_constants == NULL ||
        session->packed_weights == NULL) {
        return 0;
    }
    matmul_node_index = model->info.node_count - 3u;
    add_node_index = model->info.node_count - 2u;
    matmul_node = model->bytes + (size_t)model->node_offset +
                  (size_t)matmul_node_index * LWM_V0_NODE_SIZE;
    add_node = model->bytes + (size_t)model->node_offset +
               (size_t)add_node_index * LWM_V0_NODE_SIZE;
    if (lwm_read_u16(matmul_node) != LW_OP_MATMUL ||
        lwm_read_u16(matmul_node + 2u) != 2u || lwm_read_u16(matmul_node + 4u) != 1u ||
        lwm_read_u16(add_node) != LW_OP_ADD || lwm_read_u16(add_node + 2u) != 2u ||
        lwm_read_u16(add_node + 4u) != 1u || lwm_read_u32(add_node + 40u) != logits_index) {
        return 0;
    }
    matmul_input_index = lwm_read_u32(matmul_node + 8u);
    weights_index = lwm_read_u32(matmul_node + 12u);
    matmul_output_index = lwm_read_u32(matmul_node + 40u);
    if (lwm_read_u32(add_node + 8u) == matmul_output_index) {
        bias_index = lwm_read_u32(add_node + 12u);
    } else if (lwm_read_u32(add_node + 12u) == matmul_output_index) {
        bias_index = lwm_read_u32(add_node + 8u);
    } else {
        return 0;
    }
    activation = &session->tensors[matmul_input_index];
    weights = &session->tensors[weights_index];
    matmul_output = &session->tensors[matmul_output_index];
    bias = &session->tensors[bias_index];
    prepared = prepared_constant_for_node(session, matmul_node_index, profile);
    if (prepared == NULL || prepared->kind != LW_PREPARED_CONSTANT_MATMUL_PACKED16 ||
        activation->dtype != LW_DTYPE_F32 || activation->rank != 3u ||
        activation->dimensions[0] != 1 || activation->dimensions[1] != (int32_t)time_steps ||
        activation->dimensions[2] <= 0 || weights->dtype != LW_DTYPE_F32 ||
        weights->rank != 2u || weights->dimensions[0] != activation->dimensions[2] ||
        weights->dimensions[1] != (int32_t)class_count ||
        (weights->flags & LWM_V0_TENSOR_FLAG_CONSTANT) == 0u ||
        matmul_output->dtype != LW_DTYPE_F32 || matmul_output->rank != 3u ||
        matmul_output->dimensions[0] != 1 ||
        matmul_output->dimensions[1] != (int32_t)time_steps ||
        matmul_output->dimensions[2] != (int32_t)class_count || bias->dtype != LW_DTYPE_F32 ||
        bias->rank != 1u || bias->dimensions[0] != (int32_t)class_count ||
        (bias->flags & LWM_V0_TENSOR_FLAG_CONSTANT) == 0u) {
        return 0;
    }
    graph_input_index = lwm_read_u32(model->bytes + (size_t)model->input_offset);
    projection->matmul_node_index = matmul_node_index;
    projection->add_node_index = add_node_index;
    projection->activation = tensor_input_data(session, matmul_input_index, graph_input_index,
                                               graph_input);
    projection->packed_weights =
        (const float*)(const void*)(session->packed_weights +
                                    (size_t)prepared->packed_weight_offset);
    projection->bias = tensor_input_data(session, bias_index, graph_input_index, graph_input);
    projection->logits = tensor_output_data(session, matmul_output_index);
    projection->rows = time_steps;
    projection->inner_dimension = (uint32_t)activation->dimensions[2];
    projection->columns = class_count;
    return projection->activation != NULL && projection->bias != NULL &&
           projection->logits != NULL;
}

static void profile_simple_node(lw_execution_profile* profile, const lw_session* session,
                                uint32_t node_index, uint64_t elapsed) {
    const uint8_t* node;
    uint32_t operation;
    if (profile == NULL || node_index >= session->model->info.node_count) {
        return;
    }
    if (elapsed == 0u) {
        elapsed = 1u;
    }
    node = session->model->bytes + (size_t)session->model->node_offset +
           (size_t)node_index * LWM_V0_NODE_SIZE;
    operation = (uint32_t)lwm_read_u16(node);
    if (operation < LW_EXECUTION_PROFILE_OPERATOR_CAPACITY &&
        profile->operator_nanoseconds[operation] <= UINT64_MAX - elapsed &&
        profile->operator_invocations[operation] != UINT64_MAX) {
        profile->operator_nanoseconds[operation] += elapsed;
        ++profile->operator_invocations[operation];
    }
    if (node_index < LW_EXECUTION_PROFILE_NODE_CAPACITY &&
        profile->node_nanoseconds[node_index] <= UINT64_MAX - elapsed &&
        profile->node_invocations[node_index] != UINT64_MAX) {
        profile->node_nanoseconds[node_index] += elapsed;
        ++profile->node_invocations[node_index];
    }
}

lw_status lw_execute_session_f32_ctc_greedy(
    lw_session* session, const float* input, uint64_t input_element_count,
    uint32_t* best_indices, float* best_probabilities, uint32_t time_steps,
    uint32_t class_count, lw_execution_profile* profile, lw_error* error) {
    uint32_t logits_index;
    uint32_t softmax_node_index;
    lw_packed_ctc_projection projection;
    const float* logits;
    int fused_projection;
    uint64_t started = 0u;
    uint64_t elapsed = 0u;
    lw_status status;
    if (session == NULL || input == NULL || best_indices == NULL || best_probabilities == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "session, input, and CTC greedy outputs are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    if (profile != NULL &&
        (profile->struct_size != sizeof(*profile) || profile->reserved != 0u ||
         profile->clock == NULL)) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "an initialized execution profile and clock are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    if (!ctc_greedy_tail(session, time_steps, class_count, &logits_index)) {
        lw_set_error(error, LW_STATUS_UNSUPPORTED,
                     "session output is not a supported terminal CTC Softmax");
        return LW_STATUS_UNSUPPORTED;
    }
    softmax_node_index = session->model->info.node_count - 1u;
    fused_projection = match_packed_ctc_projection(session, input, logits_index, time_steps,
                                                   class_count, profile, &projection);
    if (profile != NULL) {
        if (profile->ctc_greedy_invocations != UINT64_MAX) {
            ++profile->ctc_greedy_invocations;
        }
        if (fused_projection) {
            if (profile->ctc_packed_projection_invocations != UINT64_MAX) {
                ++profile->ctc_packed_projection_invocations;
            }
            if (profile->packed_matmul_invocations != UINT64_MAX) {
                ++profile->packed_matmul_invocations;
            }
        } else if (profile->ctc_generic_projection_invocations != UINT64_MAX) {
            ++profile->ctc_generic_projection_invocations;
        }
    }
    status = execute_session_nodes_f32(
        session, input, input_element_count,
        fused_projection ? projection.matmul_node_index : softmax_node_index, profile, error);
    if (status != LW_STATUS_OK) {
        return status;
    }
    if (fused_projection) {
        uint64_t projection_started = profile == NULL ? 0u : profile->clock(profile->clock_context);
        uint64_t projection_elapsed = 0u;
        lw_avx2_packed_matmul_bias_argmax_f32(
            projection.activation, projection.packed_weights, projection.bias,
            projection.logits, best_indices, 1u, projection.rows,
            projection.inner_dimension, projection.columns);
        if (profile != NULL) {
            uint64_t projection_finished = profile->clock(profile->clock_context);
            if (projection_finished >= projection_started) {
                projection_elapsed = projection_finished - projection_started;
            }
            profile_simple_node(profile, session, projection.matmul_node_index,
                                projection_elapsed);
            profile_simple_node(profile, session, projection.add_node_index, 1u);
        }
        logits = projection.logits;
    } else {
        logits = tensor_input_data(
            session, logits_index,
            lwm_read_u32(session->model->bytes + (size_t)session->model->input_offset), input);
    }
    if (profile != NULL) {
        started = profile->clock(profile->clock_context);
    }
    status = fused_projection
                 ? lw_ctc_emitted_softmax_contiguous_f32(
                       logits, best_indices, best_probabilities, time_steps, class_count)
                 : lw_ctc_greedy_softmax_contiguous_f32(
                       logits, best_indices, best_probabilities, time_steps, class_count);
    if (profile != NULL) {
        uint64_t finished = profile->clock(profile->clock_context);
        if (finished >= started) {
            elapsed = finished - started;
        }
        if (elapsed == 0u) {
            elapsed = 1u;
        }
        if (profile->operator_nanoseconds[LW_OP_SOFTMAX] <= UINT64_MAX - elapsed &&
            profile->operator_invocations[LW_OP_SOFTMAX] != UINT64_MAX) {
            profile->operator_nanoseconds[LW_OP_SOFTMAX] += elapsed;
            ++profile->operator_invocations[LW_OP_SOFTMAX];
        }
        if (softmax_node_index < LW_EXECUTION_PROFILE_NODE_CAPACITY &&
            profile->node_nanoseconds[softmax_node_index] <= UINT64_MAX - elapsed &&
            profile->node_invocations[softmax_node_index] != UINT64_MAX) {
            profile->node_nanoseconds[softmax_node_index] += elapsed;
            ++profile->node_invocations[softmax_node_index];
        }
    }
    if (status != LW_STATUS_OK) {
        lw_set_error(error, status, "terminal CTC logits contain invalid values");
        return status;
    }
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
