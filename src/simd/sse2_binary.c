#include "simd_kernels.h"

#include <stddef.h>

#if defined(_M_IX86) || defined(_M_X64) || \
    defined(__i386__) || defined(__x86_64__)
#  include <emmintrin.h>
#  define LW_COMPILES_SSE2_BINARY 1
#else
#  define LW_COMPILES_SSE2_BINARY 0
#endif

#if LW_COMPILES_SSE2_BINARY && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("sse2")))
#endif
void lw_sse2_binary_contiguous_f32(
    lw_scalar_binary_op operation,
    const float* left,
    const float* right,
    float* output,
    uint64_t element_count) {
    uint64_t index = 0u;
#if LW_COMPILES_SSE2_BINARY
    if (operation == LW_SCALAR_BINARY_ADD) {
        for (; index + 4u <= element_count; index += 4u) {
            __m128 result = _mm_add_ps(
                _mm_loadu_ps(left + (size_t)index),
                _mm_loadu_ps(right + (size_t)index));
            _mm_storeu_ps(output + (size_t)index, result);
        }
    } else if (operation == LW_SCALAR_BINARY_MUL) {
        for (; index + 4u <= element_count; index += 4u) {
            __m128 result = _mm_mul_ps(
                _mm_loadu_ps(left + (size_t)index),
                _mm_loadu_ps(right + (size_t)index));
            _mm_storeu_ps(output + (size_t)index, result);
        }
    } else {
        for (; index + 4u <= element_count; index += 4u) {
            __m128 result = _mm_div_ps(
                _mm_loadu_ps(left + (size_t)index),
                _mm_loadu_ps(right + (size_t)index));
            _mm_storeu_ps(output + (size_t)index, result);
        }
    }
#endif
    lw_scalar_binary_contiguous_f32(
        operation, left + (size_t)index, right + (size_t)index,
        output + (size_t)index, element_count - index);
}

#if LW_COMPILES_SSE2_BINARY && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("sse2")))
#endif
void lw_sse2_binary_right_scalar_f32(
    lw_scalar_binary_op operation,
    const float* left,
    float right,
    float* output,
    uint64_t element_count) {
    uint64_t index = 0u;
#if LW_COMPILES_SSE2_BINARY
    __m128 right_values = _mm_set1_ps(right);
    if (operation == LW_SCALAR_BINARY_ADD) {
        for (; index + 4u <= element_count; index += 4u) {
            __m128 result = _mm_add_ps(
                _mm_loadu_ps(left + (size_t)index), right_values);
            _mm_storeu_ps(output + (size_t)index, result);
        }
    } else if (operation == LW_SCALAR_BINARY_MUL) {
        for (; index + 4u <= element_count; index += 4u) {
            __m128 result = _mm_mul_ps(
                _mm_loadu_ps(left + (size_t)index), right_values);
            _mm_storeu_ps(output + (size_t)index, result);
        }
    } else {
        for (; index + 4u <= element_count; index += 4u) {
            __m128 result = _mm_div_ps(
                _mm_loadu_ps(left + (size_t)index), right_values);
            _mm_storeu_ps(output + (size_t)index, result);
        }
    }
#endif
    lw_scalar_binary_right_scalar_f32(
        operation, left + (size_t)index, right, output + (size_t)index,
        element_count - index);
}
