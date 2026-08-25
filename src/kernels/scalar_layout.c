#include "scalar_kernels.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

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

static lw_status tensor_element_count(
    uint32_t rank,
    const int32_t* dimensions,
    uint64_t* count) {
    uint64_t value = 1u;
    uint32_t index;
    if (rank > LW_MAX_DIMS || (rank != 0u && dimensions == NULL)) {
        return LW_STATUS_INVALID_SHAPE;
    }
    for (index = 0u; index < rank; ++index) {
        if (dimensions[index] <= 0 ||
            value > UINT64_MAX / (uint32_t)dimensions[index]) {
            return LW_STATUS_INVALID_SHAPE;
        }
        value *= (uint32_t)dimensions[index];
    }
    if (value > (uint64_t)(SIZE_MAX / sizeof(float))) {
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    *count = value;
    return LW_STATUS_OK;
}

static lw_status copy_reshape(
    const float* input,
    float* output,
    uint64_t element_count) {
    if (input == NULL || output == NULL) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    if (input != output) {
        memcpy(output, input, (size_t)element_count * sizeof(float));
    }
    return LW_STATUS_OK;
}

lw_status lw_scalar_transpose_f32(
    const float* input,
    float* output,
    uint32_t rank,
    const int32_t* input_dimensions,
    uint32_t permutation_count,
    const int32_t* permutation,
    const int32_t* output_dimensions) {
    uint64_t input_strides[LW_MAX_DIMS] = {0u};
    int used[LW_MAX_DIMS] = {0};
    uint64_t element_count;
    uint64_t stride = 1u;
    uint64_t output_index;
    uint32_t axis;
    lw_status status;
    if (input == NULL || output == NULL || input == output) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    if (rank > LW_MAX_DIMS || permutation_count != rank ||
        (rank != 0u && (permutation == NULL || output_dimensions == NULL))) {
        return LW_STATUS_INVALID_SHAPE;
    }
    status = tensor_element_count(rank, input_dimensions, &element_count);
    if (status != LW_STATUS_OK) {
        return status;
    }
    for (axis = rank; axis > 0u; --axis) {
        input_strides[axis - 1u] = stride;
        stride *= (uint32_t)input_dimensions[axis - 1u];
    }
    for (axis = 0u; axis < rank; ++axis) {
        int32_t source_axis = permutation[axis];
        if (source_axis < 0 || (uint32_t)source_axis >= rank || used[source_axis] ||
            output_dimensions[axis] != input_dimensions[source_axis]) {
            return LW_STATUS_INVALID_SHAPE;
        }
        used[source_axis] = 1;
    }
    for (output_index = 0u; output_index < element_count; ++output_index) {
        uint64_t remaining = output_index;
        uint64_t input_offset = 0u;
        for (axis = rank; axis > 0u; --axis) {
            uint32_t output_axis = axis - 1u;
            uint32_t coordinate = (uint32_t)(remaining %
                (uint32_t)output_dimensions[output_axis]);
            remaining /= (uint32_t)output_dimensions[output_axis];
            input_offset += (uint64_t)coordinate *
                input_strides[(uint32_t)permutation[output_axis]];
        }
        output[(size_t)output_index] = input[(size_t)input_offset];
    }
    return LW_STATUS_OK;
}

lw_status lw_scalar_squeeze_f32(
    const float* input,
    float* output,
    uint32_t input_rank,
    const int32_t* input_dimensions,
    uint32_t axes_count,
    const int32_t* axes,
    uint32_t output_rank,
    const int32_t* output_dimensions) {
    int squeezed[LW_MAX_DIMS] = {0};
    uint64_t input_count;
    uint64_t output_count;
    uint32_t axis;
    uint32_t output_axis = 0u;
    lw_status status;
    if (axes_count > LW_MAX_DIMS || (axes_count != 0u && axes == NULL)) {
        return LW_STATUS_INVALID_SHAPE;
    }
    status = tensor_element_count(input_rank, input_dimensions, &input_count);
    if (status != LW_STATUS_OK) {
        return status;
    }
    if (axes_count == 0u) {
        for (axis = 0u; axis < input_rank; ++axis) {
            squeezed[axis] = input_dimensions[axis] == 1;
        }
    } else {
        for (axis = 0u; axis < axes_count; ++axis) {
            uint32_t normalized;
            if (!normalize_axis(axes[axis], input_rank, &normalized) ||
                squeezed[normalized] || input_dimensions[normalized] != 1) {
                return LW_STATUS_INVALID_SHAPE;
            }
            squeezed[normalized] = 1;
        }
    }
    for (axis = 0u; axis < input_rank; ++axis) {
        if (!squeezed[axis]) {
            if (output_axis >= output_rank || output_dimensions == NULL ||
                output_dimensions[output_axis] != input_dimensions[axis]) {
                return LW_STATUS_INVALID_SHAPE;
            }
            ++output_axis;
        }
    }
    if (output_axis != output_rank) {
        return LW_STATUS_INVALID_SHAPE;
    }
    status = tensor_element_count(output_rank, output_dimensions, &output_count);
    if (status != LW_STATUS_OK || output_count != input_count) {
        return status == LW_STATUS_OK ? LW_STATUS_INVALID_SHAPE : status;
    }
    return copy_reshape(input, output, input_count);
}

lw_status lw_scalar_unsqueeze_f32(
    const float* input,
    float* output,
    uint32_t input_rank,
    const int32_t* input_dimensions,
    uint32_t axes_count,
    const int32_t* axes,
    uint32_t output_rank,
    const int32_t* output_dimensions) {
    int inserted[LW_MAX_DIMS] = {0};
    uint64_t input_count;
    uint64_t output_count;
    uint32_t axis;
    uint32_t input_axis = 0u;
    lw_status status;
    if (input_rank > LW_MAX_DIMS || axes_count > LW_MAX_DIMS ||
        input_rank + axes_count != output_rank || output_rank > LW_MAX_DIMS ||
        (axes_count != 0u && axes == NULL)) {
        return LW_STATUS_INVALID_SHAPE;
    }
    status = tensor_element_count(input_rank, input_dimensions, &input_count);
    if (status != LW_STATUS_OK) {
        return status;
    }
    for (axis = 0u; axis < axes_count; ++axis) {
        uint32_t normalized;
        if (!normalize_axis(axes[axis], output_rank, &normalized) || inserted[normalized]) {
            return LW_STATUS_INVALID_SHAPE;
        }
        inserted[normalized] = 1;
    }
    for (axis = 0u; axis < output_rank; ++axis) {
        int32_t expected = inserted[axis] ? 1 : input_dimensions[input_axis++];
        if (output_dimensions == NULL || output_dimensions[axis] != expected) {
            return LW_STATUS_INVALID_SHAPE;
        }
    }
    status = tensor_element_count(output_rank, output_dimensions, &output_count);
    if (status != LW_STATUS_OK || output_count != input_count) {
        return status == LW_STATUS_OK ? LW_STATUS_INVALID_SHAPE : status;
    }
    return copy_reshape(input, output, input_count);
}

lw_status lw_scalar_reshape_f32(
    const float* input,
    float* output,
    uint32_t input_rank,
    const int32_t* input_dimensions,
    uint32_t output_rank,
    const int32_t* output_dimensions) {
    uint64_t input_count;
    uint64_t output_count;
    lw_status status = tensor_element_count(
        input_rank, input_dimensions, &input_count);
    if (status != LW_STATUS_OK) {
        return status;
    }
    status = tensor_element_count(output_rank, output_dimensions, &output_count);
    if (status != LW_STATUS_OK) {
        return status;
    }
    if (input_count != output_count) {
        return LW_STATUS_INVALID_SHAPE;
    }
    return copy_reshape(input, output, input_count);
}

lw_status lw_scalar_concat_f32(
    const float* const* inputs,
    uint32_t input_count,
    const uint32_t* input_ranks,
    const int32_t* const* input_dimensions,
    float* output,
    uint32_t output_rank,
    const int32_t* output_dimensions,
    int32_t axis) {
    uint32_t normalized_axis;
    uint64_t outer = 1u;
    uint64_t inner = 1u;
    uint64_t output_count;
    uint64_t concatenated_axis = 0u;
    uint32_t input_index;
    uint32_t dimension_index;
    uint64_t outer_index;
    lw_status status;
    if (inputs == NULL || input_ranks == NULL || input_dimensions == NULL ||
        output == NULL || output_dimensions == NULL || input_count < 1u ||
        !normalize_axis(axis, output_rank, &normalized_axis)) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    status = tensor_element_count(output_rank, output_dimensions, &output_count);
    if (status != LW_STATUS_OK) {
        return status;
    }
    (void)output_count;
    for (dimension_index = 0u; dimension_index < normalized_axis; ++dimension_index) {
        outer *= (uint32_t)output_dimensions[dimension_index];
    }
    for (dimension_index = normalized_axis + 1u;
         dimension_index < output_rank; ++dimension_index) {
        inner *= (uint32_t)output_dimensions[dimension_index];
    }
    for (input_index = 0u; input_index < input_count; ++input_index) {
        uint64_t ignored_count;
        if (inputs[input_index] == NULL || inputs[input_index] == output ||
            input_ranks[input_index] != output_rank ||
            input_dimensions[input_index] == NULL) {
            return LW_STATUS_INVALID_ARGUMENT;
        }
        status = tensor_element_count(
            input_ranks[input_index], input_dimensions[input_index], &ignored_count);
        if (status != LW_STATUS_OK) {
            return status;
        }
        for (dimension_index = 0u; dimension_index < output_rank; ++dimension_index) {
            if (dimension_index != normalized_axis &&
                input_dimensions[input_index][dimension_index] !=
                    output_dimensions[dimension_index]) {
                return LW_STATUS_INVALID_SHAPE;
            }
        }
        concatenated_axis +=
            (uint32_t)input_dimensions[input_index][normalized_axis];
    }
    if (concatenated_axis !=
        (uint32_t)output_dimensions[normalized_axis]) {
        return LW_STATUS_INVALID_SHAPE;
    }
    for (outer_index = 0u; outer_index < outer; ++outer_index) {
        uint64_t output_axis_offset = 0u;
        for (input_index = 0u; input_index < input_count; ++input_index) {
            uint64_t axis_elements =
                (uint64_t)(uint32_t)input_dimensions[input_index][normalized_axis] *
                inner;
            uint64_t input_offset = outer_index * axis_elements;
            uint64_t output_offset =
                (outer_index * (uint32_t)output_dimensions[normalized_axis] * inner) +
                output_axis_offset;
            memcpy(output + (size_t)output_offset,
                   inputs[input_index] + (size_t)input_offset,
                   (size_t)axis_elements * sizeof(float));
            output_axis_offset += axis_elements;
        }
    }
    return LW_STATUS_OK;
}

lw_status lw_scalar_resize_nearest_f32(
    const float* input,
    float* output,
    uint32_t rank,
    const int32_t* input_dimensions,
    const int32_t* output_dimensions,
    const float* scales) {
    uint64_t input_count;
    uint64_t output_count;
    uint64_t output_index;
    uint64_t input_strides[LW_MAX_DIMS] = {0u};
    uint64_t stride = 1u;
    uint32_t axis;
    lw_status status;
    if (input == NULL || output == NULL || input == output || scales == NULL ||
        rank == 0u || rank > LW_MAX_DIMS || input_dimensions == NULL ||
        output_dimensions == NULL) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    status = tensor_element_count(rank, input_dimensions, &input_count);
    if (status != LW_STATUS_OK) {
        return status;
    }
    status = tensor_element_count(rank, output_dimensions, &output_count);
    if (status != LW_STATUS_OK) {
        return status;
    }
    (void)input_count;
    for (axis = rank; axis > 0u; --axis) {
        input_strides[axis - 1u] = stride;
        stride *= (uint32_t)input_dimensions[axis - 1u];
    }
    for (axis = 0u; axis < rank; ++axis) {
        double expected;
        if (!isfinite(scales[axis]) || scales[axis] <= 0.0f) {
            return LW_STATUS_INVALID_SHAPE;
        }
        expected = floor((double)input_dimensions[axis] * scales[axis]);
        if (expected != output_dimensions[axis]) {
            return LW_STATUS_INVALID_SHAPE;
        }
    }
    for (output_index = 0u; output_index < output_count; ++output_index) {
        uint64_t remaining = output_index;
        uint64_t input_offset = 0u;
        for (axis = rank; axis > 0u; --axis) {
            uint32_t current_axis = axis - 1u;
            uint32_t coordinate = (uint32_t)(remaining %
                (uint32_t)output_dimensions[current_axis]);
            uint32_t input_coordinate = (uint32_t)floor(
                (double)coordinate / scales[current_axis]);
            if (input_coordinate >= (uint32_t)input_dimensions[current_axis]) {
                input_coordinate = (uint32_t)input_dimensions[current_axis] - 1u;
            }
            remaining /= (uint32_t)output_dimensions[current_axis];
            input_offset += (uint64_t)input_coordinate * input_strides[current_axis];
        }
        output[(size_t)output_index] = input[(size_t)input_offset];
    }
    return LW_STATUS_OK;
}
