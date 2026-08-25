#include "scalar_kernels.h"
#include "cpu_features.h"
#include "simd_kernels.h"

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

void lw_scalar_binary_contiguous_f32(
    lw_scalar_binary_op operation,
    const float* left,
    const float* right,
    float* output,
    uint64_t element_count) {
    uint64_t index;
    if (operation == LW_SCALAR_BINARY_ADD) {
        for (index = 0u; index < element_count; ++index) {
            output[(size_t)index] =
                left[(size_t)index] + right[(size_t)index];
        }
    } else if (operation == LW_SCALAR_BINARY_MUL) {
        for (index = 0u; index < element_count; ++index) {
            output[(size_t)index] =
                left[(size_t)index] * right[(size_t)index];
        }
    } else {
        for (index = 0u; index < element_count; ++index) {
            output[(size_t)index] =
                left[(size_t)index] / right[(size_t)index];
        }
    }
}

void lw_scalar_binary_right_scalar_f32(
    lw_scalar_binary_op operation,
    const float* left,
    float right,
    float* output,
    uint64_t element_count) {
    uint64_t index;
    if (operation == LW_SCALAR_BINARY_ADD) {
        for (index = 0u; index < element_count; ++index) {
            output[(size_t)index] = left[(size_t)index] + right;
        }
    } else if (operation == LW_SCALAR_BINARY_MUL) {
        for (index = 0u; index < element_count; ++index) {
            output[(size_t)index] = left[(size_t)index] * right;
        }
    } else {
        for (index = 0u; index < element_count; ++index) {
            output[(size_t)index] = left[(size_t)index] / right;
        }
    }
}

static void dispatch_binary_contiguous_for_level_f32(
    lw_scalar_binary_op operation,
    const float* left,
    const float* right,
    float* output,
    uint64_t element_count,
    lw_simd_level simd_level) {
    if (simd_level >= LW_SIMD_LEVEL_AVX2) {
        lw_avx2_binary_contiguous_f32(
            operation, left, right, output, element_count);
    } else if (simd_level >= LW_SIMD_LEVEL_SSE2) {
        lw_sse2_binary_contiguous_f32(
            operation, left, right, output, element_count);
    } else {
        lw_scalar_binary_contiguous_f32(
            operation, left, right, output, element_count);
    }
}

static void dispatch_binary_contiguous_f32(
    lw_scalar_binary_op operation,
    const float* left,
    const float* right,
    float* output,
    uint64_t element_count) {
    dispatch_binary_contiguous_for_level_f32(
        operation, left, right, output, element_count, lw_detect_simd_level());
}

static void dispatch_binary_right_scalar_for_level_f32(
    lw_scalar_binary_op operation,
    const float* left,
    float right,
    float* output,
    uint64_t element_count,
    lw_simd_level simd_level) {
    if (simd_level >= LW_SIMD_LEVEL_AVX2) {
        lw_avx2_binary_right_scalar_f32(
            operation, left, right, output, element_count);
    } else if (simd_level >= LW_SIMD_LEVEL_SSE2) {
        lw_sse2_binary_right_scalar_f32(
            operation, left, right, output, element_count);
    } else {
        lw_scalar_binary_right_scalar_f32(
            operation, left, right, output, element_count);
    }
}

static void dispatch_binary_right_scalar_f32(
    lw_scalar_binary_op operation,
    const float* left,
    float right,
    float* output,
    uint64_t element_count) {
    dispatch_binary_right_scalar_for_level_f32(
        operation, left, right, output, element_count, lw_detect_simd_level());
}

static int try_dispatch_binary_single_axis_broadcast_f32(
    lw_scalar_binary_op operation,
    const float* left,
    const float* right,
    float* output,
    uint32_t right_rank,
    const int32_t* right_dimensions,
    uint32_t output_rank,
    const int32_t* output_dimensions,
    uint64_t output_count) {
    uint32_t right_padding = output_rank - right_rank;
    uint32_t non_unit_dimensions = 0u;
    uint32_t broadcast_axis = 0u;
    uint32_t axis;
    uint64_t inner_count = 1u;
    uint64_t right_dimension;
    uint64_t outer_count;
    uint64_t outer_index;
    lw_simd_level simd_level;

    for (axis = 0u; axis < output_rank; ++axis) {
        int32_t dimension = axis < right_padding ?
            1 : right_dimensions[axis - right_padding];
        if (dimension != 1) {
            ++non_unit_dimensions;
            broadcast_axis = axis;
        }
    }
    if (non_unit_dimensions != 1u) {
        return 0;
    }
    for (axis = broadcast_axis + 1u; axis < output_rank; ++axis) {
        inner_count *= (uint32_t)output_dimensions[axis];
    }
    right_dimension = (uint32_t)output_dimensions[broadcast_axis];
    outer_count = output_count / (right_dimension * inner_count);
    simd_level = lw_detect_simd_level();
    for (outer_index = 0u; outer_index < outer_count; ++outer_index) {
        uint64_t outer_offset = outer_index * right_dimension * inner_count;
        if (inner_count == 1u) {
            dispatch_binary_contiguous_for_level_f32(
                operation, left + (size_t)outer_offset, right,
                output + (size_t)outer_offset, right_dimension, simd_level);
        } else {
            uint64_t right_index;
            for (right_index = 0u; right_index < right_dimension; ++right_index) {
                uint64_t block_offset = outer_offset + right_index * inner_count;
                dispatch_binary_right_scalar_for_level_f32(
                    operation, left + (size_t)block_offset,
                    right[(size_t)right_index], output + (size_t)block_offset,
                    inner_count, simd_level);
            }
        }
    }
    return 1;
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

    if (left_count == output_count && right_count == output_count) {
        dispatch_binary_contiguous_f32(
            operation, left, right, output, output_count);
        return LW_STATUS_OK;
    }
    if (left_count == output_count && right_count == 1u) {
        dispatch_binary_right_scalar_f32(
            operation, left, right[0], output, output_count);
        return LW_STATUS_OK;
    }
    if (left_count == output_count &&
        try_dispatch_binary_single_axis_broadcast_f32(
            operation, left, right, output, right_rank, right_dimensions,
            output_rank, output_dimensions, output_count)) {
        return LW_STATUS_OK;
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
