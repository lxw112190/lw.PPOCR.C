#include "session_internal.h"
#include "lwm_read.h"

#include <limits.h>
#include <string.h>

static lw_status shape_fail(lw_error* error, const char* message) {
    lw_set_error(error, LW_STATUS_INVALID_SHAPE, message);
    return LW_STATUS_INVALID_SHAPE;
}

static uint32_t dtype_size(uint32_t dtype) {
    switch (dtype) {
        case LW_DTYPE_F32: return 4u;
        case LW_DTYPE_I32: return 4u;
        case LW_DTYPE_I64: return 8u;
        case LW_DTYPE_U8: return 1u;
        default: return 0u;
    }
}

static lw_status set_tensor_shape(
    lw_runtime_tensor* tensor,
    uint32_t dtype,
    uint32_t rank,
    const int32_t* dimensions,
    uint64_t max_tensor_size,
    lw_error* error) {
    uint64_t elements = 1u;
    uint32_t item_size = dtype_size(dtype);
    uint32_t i;
    if (item_size == 0u || rank > LW_MAX_DIMS) {
        return shape_fail(error, "unsupported resolved tensor dtype or rank");
    }
    if (tensor->dtype != dtype || tensor->rank != rank) {
        return shape_fail(error, "resolved tensor type or rank disagrees with the model");
    }
    for (i = 0u; i < rank; ++i) {
        int32_t expected = tensor->dimensions[i];
        int32_t resolved = dimensions[i];
        if (resolved <= 0 || (expected != -1 && expected != resolved)) {
            return shape_fail(error, "resolved tensor dimension disagrees with the model");
        }
        if (elements > UINT64_MAX / (uint32_t)resolved) {
            return shape_fail(error, "resolved tensor element count overflows");
        }
        elements *= (uint32_t)resolved;
    }
    if (elements > UINT64_MAX / item_size) {
        return shape_fail(error, "resolved tensor byte size overflows");
    }
    tensor->byte_size = elements * item_size;
    if (tensor->byte_size > max_tensor_size) {
        lw_set_error(error, LW_STATUS_MEMORY_LIMIT, "resolved tensor exceeds max_tensor_size");
        return LW_STATUS_MEMORY_LIMIT;
    }
    memset(tensor->dimensions, 0, sizeof(tensor->dimensions));
    if (rank != 0u) {
        memcpy(tensor->dimensions, dimensions, (size_t)rank * sizeof(dimensions[0]));
    }
    return LW_STATUS_OK;
}

static int normalize_axis(int32_t axis, uint32_t rank, uint32_t* normalized) {
    int64_t value = axis;
    if (value < 0) {
        value += rank;
    }
    if (value < 0 || value >= rank) {
        return 0;
    }
    *normalized = (uint32_t)value;
    return 1;
}

static int broadcast_dimension(int32_t left, int32_t right, int32_t* output) {
    if (left == right) {
        *output = left;
        return 1;
    }
    if (left == 1) {
        *output = right;
        return 1;
    }
    if (right == 1) {
        *output = left;
        return 1;
    }
    return 0;
}

static lw_status broadcast_shape(
    const lw_runtime_tensor* left,
    const lw_runtime_tensor* right,
    uint32_t* output_rank,
    int32_t* output_dimensions,
    lw_error* error) {
    uint32_t rank = left->rank > right->rank ? left->rank : right->rank;
    uint32_t i;
    if (left->dtype != right->dtype) {
        return shape_fail(error, "elementwise input dtypes do not match");
    }
    for (i = 0u; i < rank; ++i) {
        int32_t left_dim = i < rank - left->rank ? 1 : left->dimensions[i - (rank - left->rank)];
        int32_t right_dim = i < rank - right->rank ? 1 : right->dimensions[i - (rank - right->rank)];
        if (!broadcast_dimension(left_dim, right_dim, &output_dimensions[i])) {
            return shape_fail(error, "elementwise input shapes cannot be broadcast");
        }
    }
    *output_rank = rank;
    return LW_STATUS_OK;
}

static int spatial_output(
    int32_t input,
    int32_t kernel,
    int32_t stride,
    int32_t dilation,
    int32_t pad_before,
    int32_t pad_after,
    uint32_t ceil_mode,
    int32_t* output) {
    int64_t effective;
    int64_t numerator;
    int64_t value;
    if (input <= 0 || kernel <= 0 || stride <= 0 || dilation <= 0 || pad_before < 0 || pad_after < 0) {
        return 0;
    }
    effective = (int64_t)(kernel - 1) * dilation + 1;
    numerator = (int64_t)input + pad_before + pad_after - effective;
    if (numerator < 0) {
        return 0;
    }
    value = (numerator + (ceil_mode != 0u ? stride - 1 : 0)) / stride + 1;
    if (ceil_mode != 0u && (value - 1) * stride >= (int64_t)input + pad_before) {
        --value;
    }
    if (value <= 0 || value > INT32_MAX) {
        return 0;
    }
    *output = (int32_t)value;
    return 1;
}

static lw_status resolve_node(
    lw_session* session,
    const uint8_t* node,
    uint64_t max_tensor_size,
    lw_error* error) {
    const lw_model* model = session->model;
    uint16_t op = lwm_read_u16(node);
    uint16_t input_count = lwm_read_u16(node + 2);
    uint16_t output_count = lwm_read_u16(node + 4);
    uint64_t param_offset = lwm_read_u64(node + 56);
    const uint8_t* params = param_offset == 0u ? NULL : model->bytes + (size_t)param_offset;
    const lw_runtime_tensor* inputs[LWM_V0_MAX_NODE_INPUTS];
    lw_runtime_tensor* output;
    int32_t dimensions[LW_MAX_DIMS] = {0};
    uint32_t rank = 0u;
    uint32_t i;
    lw_status status;

    if (output_count != 1u) {
        return shape_fail(error, "current REC shape resolver requires one output per node");
    }
    for (i = 0u; i < input_count; ++i) {
        inputs[i] = &session->tensors[lwm_read_u32(node + 8u + i * 4u)];
    }
    output = &session->tensors[lwm_read_u32(node + 40)];

    if (op == 2u || op == 3u || op == 4u) {
        if (input_count != 2u) {
            return shape_fail(error, "elementwise node has invalid arity");
        }
        status = broadcast_shape(inputs[0], inputs[1], &rank, dimensions, error);
        if (status != LW_STATUS_OK) {
            return status;
        }
    } else if (op == 5u || op == 6u || op == 9u || op == 15u) {
        if (input_count != 1u) {
            return shape_fail(error, "unary node has invalid arity");
        }
        rank = inputs[0]->rank;
        memcpy(dimensions, inputs[0]->dimensions, sizeof(dimensions));
        if (op == 15u) {
            uint32_t axis;
            if (!normalize_axis(lwm_read_i32(params + 4), rank, &axis)) {
                return shape_fail(error, "Softmax axis is outside the input rank");
            }
        }
    } else if (op == 1u) {
        uint32_t group;
        if ((input_count != 2u && input_count != 3u) || inputs[0]->rank != 4u || inputs[1]->rank != 4u) {
            return shape_fail(error, "Conv requires NCHW input and OIHW weight");
        }
        if (inputs[0]->dtype != inputs[1]->dtype ||
            (input_count == 3u && (inputs[2]->dtype != inputs[0]->dtype || inputs[2]->rank != 1u ||
                                  inputs[2]->dimensions[0] != inputs[1]->dimensions[0]))) {
            return shape_fail(error, "Conv weight or bias type/shape is inconsistent");
        }
        group = lwm_read_u32(params + 4);
        if ((uint64_t)inputs[1]->dimensions[1] * group != (uint32_t)inputs[0]->dimensions[1] ||
            inputs[1]->dimensions[2] != lwm_read_i32(params + 8) ||
            inputs[1]->dimensions[3] != lwm_read_i32(params + 12)) {
            return shape_fail(error, "Conv channel or kernel dimensions are inconsistent");
        }
        rank = 4u;
        dimensions[0] = inputs[0]->dimensions[0];
        dimensions[1] = inputs[1]->dimensions[0];
        if (!spatial_output(inputs[0]->dimensions[2], lwm_read_i32(params + 8), lwm_read_i32(params + 16),
                            lwm_read_i32(params + 24), lwm_read_i32(params + 32), lwm_read_i32(params + 40), 0u,
                            &dimensions[2]) ||
            !spatial_output(inputs[0]->dimensions[3], lwm_read_i32(params + 12), lwm_read_i32(params + 20),
                            lwm_read_i32(params + 28), lwm_read_i32(params + 36), lwm_read_i32(params + 44), 0u,
                            &dimensions[3])) {
            return shape_fail(error, "Conv output shape is invalid");
        }
    } else if (op == 7u) {
        if (input_count != 5u || inputs[0]->rank < 2u) {
            return shape_fail(error, "BatchNormalization has invalid input arity or rank");
        }
        for (i = 1u; i < 5u; ++i) {
            if (inputs[i]->dtype != inputs[0]->dtype || inputs[i]->rank != 1u ||
                inputs[i]->dimensions[0] != inputs[0]->dimensions[1]) {
                return shape_fail(error, "BatchNormalization parameter shape is invalid");
            }
        }
        rank = inputs[0]->rank;
        memcpy(dimensions, inputs[0]->dimensions, sizeof(dimensions));
    } else if (op == 8u) {
        uint32_t axes_count;
        uint32_t keepdims;
        int reduced[LW_MAX_DIMS] = {0};
        uint32_t output_index = 0u;
        if (input_count != 1u) {
            return shape_fail(error, "ReduceMean has invalid arity");
        }
        axes_count = lwm_read_u16(params + 2);
        keepdims = lwm_read_u32(params + 4);
        if (axes_count == 0u && lwm_read_u32(params + 8) != 0u) {
            rank = inputs[0]->rank;
            memcpy(dimensions, inputs[0]->dimensions, sizeof(dimensions));
        } else if (axes_count == 0u) {
            for (i = 0u; i < inputs[0]->rank; ++i) {
                reduced[i] = 1;
            }
        } else {
            for (i = 0u; i < axes_count; ++i) {
                uint32_t axis;
                if (!normalize_axis(lwm_read_i32(params + 12u + i * 4u), inputs[0]->rank, &axis) || reduced[axis]) {
                    return shape_fail(error, "ReduceMean axes are invalid or duplicated");
                }
                reduced[axis] = 1;
            }
        }
        for (i = 0u; i < inputs[0]->rank; ++i) {
            if (reduced[i]) {
                if (keepdims != 0u) {
                    dimensions[output_index++] = 1;
                }
            } else {
                dimensions[output_index++] = inputs[0]->dimensions[i];
            }
        }
        rank = output_index;
    } else if (op == 10u) {
        if (input_count != 1u || inputs[0]->rank != 4u) {
            return shape_fail(error, "AveragePool requires one NCHW input");
        }
        rank = 4u;
        dimensions[0] = inputs[0]->dimensions[0];
        dimensions[1] = inputs[0]->dimensions[1];
        if (!spatial_output(inputs[0]->dimensions[2], lwm_read_i32(params + 8), lwm_read_i32(params + 16), 1,
                            lwm_read_i32(params + 24), lwm_read_i32(params + 32), lwm_read_u32(params + 40),
                            &dimensions[2]) ||
            !spatial_output(inputs[0]->dimensions[3], lwm_read_i32(params + 12), lwm_read_i32(params + 20), 1,
                            lwm_read_i32(params + 28), lwm_read_i32(params + 36), lwm_read_u32(params + 40),
                            &dimensions[3])) {
            return shape_fail(error, "AveragePool output shape is invalid");
        }
    } else if (op == 11u) {
        uint32_t axes_count;
        int squeezed[LW_MAX_DIMS] = {0};
        uint32_t output_index = 0u;
        if (input_count != 1u) {
            return shape_fail(error, "Squeeze has invalid arity");
        }
        axes_count = lwm_read_u16(params + 2);
        if (axes_count == 0u) {
            for (i = 0u; i < inputs[0]->rank; ++i) {
                squeezed[i] = inputs[0]->dimensions[i] == 1;
            }
        } else {
            for (i = 0u; i < axes_count; ++i) {
                uint32_t axis;
                if (!normalize_axis(lwm_read_i32(params + 4u + i * 4u), inputs[0]->rank, &axis) ||
                    squeezed[axis] || inputs[0]->dimensions[axis] != 1) {
                    return shape_fail(error, "Squeeze axes are invalid, duplicated, or not unit dimensions");
                }
                squeezed[axis] = 1;
            }
        }
        for (i = 0u; i < inputs[0]->rank; ++i) {
            if (!squeezed[i]) {
                dimensions[output_index++] = inputs[0]->dimensions[i];
            }
        }
        rank = output_index;
    } else if (op == 13u) {
        uint32_t axes_count;
        uint32_t output_rank;
        int inserted[LW_MAX_DIMS] = {0};
        uint32_t input_index = 0u;
        if (input_count != 1u) {
            return shape_fail(error, "Unsqueeze has invalid arity");
        }
        axes_count = lwm_read_u16(params + 2);
        output_rank = inputs[0]->rank + axes_count;
        if (output_rank > LW_MAX_DIMS) {
            return shape_fail(error, "Unsqueeze output rank exceeds the runtime limit");
        }
        for (i = 0u; i < axes_count; ++i) {
            uint32_t axis;
            if (!normalize_axis(lwm_read_i32(params + 4u + i * 4u), output_rank, &axis) || inserted[axis]) {
                return shape_fail(error, "Unsqueeze axes are invalid or duplicated");
            }
            inserted[axis] = 1;
        }
        for (i = 0u; i < output_rank; ++i) {
            dimensions[i] = inserted[i] ? 1 : inputs[0]->dimensions[input_index++];
        }
        rank = output_rank;
    } else if (op == 12u) {
        uint32_t perm_count;
        int used[LW_MAX_DIMS] = {0};
        if (input_count != 1u) {
            return shape_fail(error, "Transpose has invalid arity");
        }
        perm_count = lwm_read_u16(params + 2);
        if (perm_count != inputs[0]->rank) {
            return shape_fail(error, "Transpose perm length disagrees with input rank");
        }
        rank = inputs[0]->rank;
        for (i = 0u; i < rank; ++i) {
            int32_t axis_value = lwm_read_i32(params + 4u + i * 4u);
            if (axis_value < 0 || (uint32_t)axis_value >= rank || used[axis_value]) {
                return shape_fail(error, "Transpose perm is invalid or duplicated");
            }
            used[axis_value] = 1;
            dimensions[i] = inputs[0]->dimensions[axis_value];
        }
    } else if (op == 14u) {
        uint32_t batch_rank;
        uint32_t left_offset;
        uint32_t right_offset;
        if (input_count != 2u || inputs[0]->dtype != inputs[1]->dtype ||
            inputs[0]->rank < 2u || inputs[1]->rank < 2u) {
            return shape_fail(error, "MatMul currently requires inputs with rank at least two");
        }
        if (inputs[0]->dimensions[inputs[0]->rank - 1u] != inputs[1]->dimensions[inputs[1]->rank - 2u]) {
            return shape_fail(error, "MatMul inner dimensions do not match");
        }
        batch_rank = (inputs[0]->rank - 2u) > (inputs[1]->rank - 2u) ? inputs[0]->rank - 2u : inputs[1]->rank - 2u;
        rank = batch_rank + 2u;
        left_offset = batch_rank - (inputs[0]->rank - 2u);
        right_offset = batch_rank - (inputs[1]->rank - 2u);
        for (i = 0u; i < batch_rank; ++i) {
            int32_t left_dim = i < left_offset ? 1 : inputs[0]->dimensions[i - left_offset];
            int32_t right_dim = i < right_offset ? 1 : inputs[1]->dimensions[i - right_offset];
            if (!broadcast_dimension(left_dim, right_dim, &dimensions[i])) {
                return shape_fail(error, "MatMul batch dimensions cannot be broadcast");
            }
        }
        dimensions[batch_rank] = inputs[0]->dimensions[inputs[0]->rank - 2u];
        dimensions[batch_rank + 1u] = inputs[1]->dimensions[inputs[1]->rank - 1u];
    } else {
        lw_set_error(error, LW_STATUS_UNSUPPORTED, "shape resolver encountered an unsupported operator");
        return LW_STATUS_UNSUPPORTED;
    }
    return set_tensor_shape(output, inputs[0]->dtype, rank, dimensions, max_tensor_size, error);
}

lw_status lw_resolve_shapes(lw_session* session, uint64_t max_tensor_size, lw_error* error) {
    const lw_model* model = session->model;
    uint32_t i;
    for (i = 0u; i < model->info.node_count; ++i) {
        const uint8_t* node = model->bytes + (size_t)model->node_offset + (size_t)i * LWM_V0_NODE_SIZE;
        lw_status status = resolve_node(session, node, max_tensor_size, error);
        if (status != LW_STATUS_OK) {
            return status;
        }
    }
    return LW_STATUS_OK;
}
