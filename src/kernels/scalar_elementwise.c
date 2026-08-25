#include "scalar_kernels.h"

#include <stddef.h>
#include <stdint.h>

static lw_status make_contiguous_strides(
    uint32_t rank,
    const int32_t* dimensions,
    uint64_t* strides,
    uint64_t* element_count) {
    uint64_t count = 1u;
    uint32_t axis;
    if (rank > LW_MAX_DIMS || (rank != 0u && dimensions == NULL)) {
        return LW_STATUS_INVALID_SHAPE;
    }
    for (axis = rank; axis > 0u; --axis) {
        int32_t dimension = dimensions[axis - 1u];
        if (dimension <= 0 || count > UINT64_MAX / (uint32_t)dimension) {
            return LW_STATUS_INVALID_SHAPE;
        }
        strides[axis - 1u] = count;
        count *= (uint32_t)dimension;
    }
    if (count > (uint64_t)(SIZE_MAX / sizeof(float))) {
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    *element_count = count;
    return LW_STATUS_OK;
}

lw_status lw_scalar_binary_f32(
    lw_scalar_binary_op operation,
    const float* left,
    uint32_t left_rank,
    const int32_t* left_dimensions,
    const float* right,
    uint32_t right_rank,
    const int32_t* right_dimensions,
    float* output,
    uint32_t output_rank,
    const int32_t* output_dimensions) {
    uint64_t left_contiguous[LW_MAX_DIMS] = {0u};
    uint64_t right_contiguous[LW_MAX_DIMS] = {0u};
    uint64_t left_strides[LW_MAX_DIMS] = {0u};
    uint64_t right_strides[LW_MAX_DIMS] = {0u};
    uint32_t coordinates[LW_MAX_DIMS] = {0u};
    uint64_t left_count;
    uint64_t right_count;
    uint64_t output_count = 1u;
    uint64_t left_offset = 0u;
    uint64_t right_offset = 0u;
    uint64_t output_index;
    uint32_t axis;
    lw_status status;

    if (operation != LW_SCALAR_BINARY_ADD && operation != LW_SCALAR_BINARY_MUL &&
        operation != LW_SCALAR_BINARY_DIV) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    if (left == NULL || right == NULL || output == NULL || output == left || output == right) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    if (left_rank > LW_MAX_DIMS || right_rank > LW_MAX_DIMS ||
        output_rank > LW_MAX_DIMS || output_rank != (left_rank > right_rank ? left_rank : right_rank) ||
        (output_rank != 0u && output_dimensions == NULL)) {
        return LW_STATUS_INVALID_SHAPE;
    }
    status = make_contiguous_strides(left_rank, left_dimensions, left_contiguous, &left_count);
    if (status != LW_STATUS_OK) {
        return status;
    }
    status = make_contiguous_strides(right_rank, right_dimensions, right_contiguous, &right_count);
    if (status != LW_STATUS_OK) {
        return status;
    }
    (void)left_count;
    (void)right_count;

    for (axis = 0u; axis < output_rank; ++axis) {
        uint32_t left_padding = output_rank - left_rank;
        uint32_t right_padding = output_rank - right_rank;
        int32_t left_dimension = axis < left_padding ? 1 : left_dimensions[axis - left_padding];
        int32_t right_dimension = axis < right_padding ? 1 : right_dimensions[axis - right_padding];
        int32_t expected_dimension;
        if (left_dimension <= 0 || right_dimension <= 0 ||
            (left_dimension != right_dimension && left_dimension != 1 && right_dimension != 1)) {
            return LW_STATUS_INVALID_SHAPE;
        }
        expected_dimension = left_dimension > right_dimension ? left_dimension : right_dimension;
        if (output_dimensions[axis] != expected_dimension) {
            return LW_STATUS_INVALID_SHAPE;
        }
        if (output_count > UINT64_MAX / (uint32_t)expected_dimension) {
            return LW_STATUS_OUT_OF_BOUNDS;
        }
        output_count *= (uint32_t)expected_dimension;
        if (axis >= left_padding && left_dimension != 1) {
            left_strides[axis] = left_contiguous[axis - left_padding];
        }
        if (axis >= right_padding && right_dimension != 1) {
            right_strides[axis] = right_contiguous[axis - right_padding];
        }
    }
    if (output_count > (uint64_t)(SIZE_MAX / sizeof(float))) {
        return LW_STATUS_OUT_OF_BOUNDS;
    }

    for (output_index = 0u; output_index < output_count; ++output_index) {
        float left_value = left[(size_t)left_offset];
        float right_value = right[(size_t)right_offset];
        if (operation == LW_SCALAR_BINARY_ADD) {
            output[(size_t)output_index] = left_value + right_value;
        } else if (operation == LW_SCALAR_BINARY_MUL) {
            output[(size_t)output_index] = left_value * right_value;
        } else {
            output[(size_t)output_index] = left_value / right_value;
        }

        for (axis = output_rank; axis > 0u; --axis) {
            uint32_t current_axis = axis - 1u;
            ++coordinates[current_axis];
            if (coordinates[current_axis] < (uint32_t)output_dimensions[current_axis]) {
                left_offset += left_strides[current_axis];
                right_offset += right_strides[current_axis];
                break;
            }
            coordinates[current_axis] = 0u;
            left_offset -= left_strides[current_axis] *
                           ((uint32_t)output_dimensions[current_axis] - 1u);
            right_offset -= right_strides[current_axis] *
                            ((uint32_t)output_dimensions[current_axis] - 1u);
        }
    }
    return LW_STATUS_OK;
}
