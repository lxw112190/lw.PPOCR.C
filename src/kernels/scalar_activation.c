#include "scalar_kernels.h"

/* Portable activation functions used as the reference implementation. */

#include <math.h>
#include <stddef.h>
#include <stdint.h>

static lw_status validate_elementwise(const float* input, float* output, uint64_t element_count) {
    if (element_count > (uint64_t)(SIZE_MAX / sizeof(float))) {
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    if (element_count != 0u && (input == NULL || output == NULL)) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    return LW_STATUS_OK;
}

lw_status lw_scalar_relu_f32(const float* input, float* output, uint64_t element_count) {
    uint64_t index;
    lw_status status = validate_elementwise(input, output, element_count);
    if (status != LW_STATUS_OK) {
        return status;
    }
    for (index = 0u; index < element_count; ++index) {
        float value = input[(size_t)index];
        output[(size_t)index] = value > 0.0f ? value : 0.0f;
    }
    return LW_STATUS_OK;
}

lw_status lw_scalar_erf_f32(const float* input, float* output, uint64_t element_count) {
    uint64_t index;
    lw_status status = validate_elementwise(input, output, element_count);
    if (status != LW_STATUS_OK) {
        return status;
    }
    for (index = 0u; index < element_count; ++index) {
        output[(size_t)index] = erff(input[(size_t)index]);
    }
    return LW_STATUS_OK;
}

static float gelu_exact(float value) {
    const float activated = erff(value / 1.4142135381698608f) + 1.0f;
    return (value * activated) * 0.5f;
}

void lw_scalar_gelu_exact_f32(const float* input, float* output, uint64_t element_count) {
    const uintptr_t source = (uintptr_t)input;
    const uintptr_t destination = (uintptr_t)output;
    /* The final tensor can reuse a region of the expired input's arena slot. */
    if (destination > source && (destination - source) / sizeof(float) < element_count) {
        for (uint64_t index = element_count; index-- > 0u;) {
            output[(size_t)index] = gelu_exact(input[(size_t)index]);
        }
    } else {
        for (uint64_t index = 0u; index < element_count; ++index) {
            output[(size_t)index] = gelu_exact(input[(size_t)index]);
        }
    }
}

lw_status lw_scalar_hard_sigmoid_f32(const float* input, float* output, uint64_t element_count,
                                     float alpha, float beta) {
    uint64_t index;
    lw_status status = validate_elementwise(input, output, element_count);
    if (status != LW_STATUS_OK) {
        return status;
    }
    if (!isfinite(alpha) || !isfinite(beta)) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    for (index = 0u; index < element_count; ++index) {
        float value = alpha * input[(size_t)index] + beta;
        if (value < 0.0f) {
            value = 0.0f;
        } else if (value > 1.0f) {
            value = 1.0f;
        }
        output[(size_t)index] = value;
    }
    return LW_STATUS_OK;
}

lw_status lw_scalar_sigmoid_f32(const float* input, float* output, uint64_t element_count) {
    uint64_t index;
    lw_status status = validate_elementwise(input, output, element_count);
    if (status != LW_STATUS_OK) {
        return status;
    }
    for (index = 0u; index < element_count; ++index) {
        float value = input[(size_t)index];
        if (value >= 0.0f) {
            output[(size_t)index] = 1.0f / (1.0f + expf(-value));
        } else {
            float exponential = expf(value);
            output[(size_t)index] = exponential / (1.0f + exponential);
        }
    }
    return LW_STATUS_OK;
}

lw_status lw_scalar_sqrt_f32(const float* input, float* output, uint64_t element_count) {
    uint64_t index;
    lw_status status = validate_elementwise(input, output, element_count);
    if (status != LW_STATUS_OK) {
        return status;
    }
    for (index = 0u; index < element_count; ++index) {
        output[(size_t)index] = sqrtf(input[(size_t)index]);
    }
    return LW_STATUS_OK;
}
