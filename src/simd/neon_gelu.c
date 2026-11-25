#include "simd_kernels.h"

#include <stddef.h>

#if defined(_M_ARM64) || defined(__aarch64__)
#include <arm_neon.h>

/* The same three Erf polynomials as the AVX2 and WASM GELU kernels.
 * NEON evaluates each Horner step with an explicit fused multiply-add. */
static inline float32x4_t erf_small_magnitude_f32(float32x4_t absolute, float32x4_t squared) {
    float32x4_t polynomial = vdupq_n_f32(1.0590875083315439e-6f);
    polynomial = vfmaq_f32(vdupq_n_f32(-1.3906452410711274e-5f), squared, polynomial);
    polynomial = vfmaq_f32(vdupq_n_f32(1.1955437252428243e-4f), squared, polynomial);
    polynomial = vfmaq_f32(vdupq_n_f32(-8.5424759600797662e-4f), squared, polynomial);
    polynomial = vfmaq_f32(vdupq_n_f32(5.2237718994271529e-3f), squared, polynomial);
    polynomial = vfmaq_f32(vdupq_n_f32(-2.6866128888288668e-2f), squared, polynomial);
    polynomial = vfmaq_f32(vdupq_n_f32(1.1283791226625459e-1f), squared, polynomial);
    polynomial = vfmaq_f32(vdupq_n_f32(-3.7612638883043881e-1f), squared, polynomial);
    polynomial = vfmaq_f32(vdupq_n_f32(1.1283791670929921f), squared, polynomial);
    return vmulq_f32(absolute, polynomial);
}

static inline float32x4_t erf_middle_magnitude_f32(float32x4_t absolute) {
    float32x4_t shifted = vsubq_f32(absolute, vdupq_n_f32(1.5f));
    float32x4_t polynomial = vdupq_n_f32(-2.4006675278365739e-3f);
    polynomial = vfmaq_f32(vdupq_n_f32(-3.8855162788028288e-3f), shifted, polynomial);
    polynomial = vfmaq_f32(vdupq_n_f32(1.9332860298601401e-2f), shifted, polynomial);
    polynomial = vfmaq_f32(vdupq_n_f32(-1.5013607234871428e-2f), shifted, polynomial);
    polynomial = vfmaq_f32(vdupq_n_f32(-4.4599369472787913e-2f), shifted, polynomial);
    polynomial = vfmaq_f32(vdupq_n_f32(1.3876136033191724e-1f), shifted, polynomial);
    polynomial = vfmaq_f32(vdupq_n_f32(-1.7839541988688759e-1f), shifted, polynomial);
    polynomial = vfmaq_f32(vdupq_n_f32(1.1893013063163335e-1f), shifted, polynomial);
    return vfmaq_f32(vdupq_n_f32(9.6610514641406819e-1f), shifted, polynomial);
}

static inline float32x4_t erf_large_magnitude_f32(float32x4_t absolute) {
    float32x4_t shifted = vsubq_f32(absolute, vdupq_n_f32(3.0f));
    float32x4_t polynomial = vdupq_n_f32(-8.8750765320565021e-5f);
    polynomial = vfmaq_f32(vdupq_n_f32(3.8805285630072217e-4f), shifted, polynomial);
    polynomial = vfmaq_f32(vdupq_n_f32(-7.7812010711428433e-4f), shifted, polynomial);
    polynomial = vfmaq_f32(vdupq_n_f32(1.0255980996254569e-3f), shifted, polynomial);
    polynomial = vfmaq_f32(vdupq_n_f32(-1.0307062838910627e-3f), shifted, polynomial);
    polynomial = vfmaq_f32(vdupq_n_f32(7.8586080129729565e-4f), shifted, polynomial);
    polynomial = vfmaq_f32(vdupq_n_f32(-4.1936053855221229e-4f), shifted, polynomial);
    polynomial = vfmaq_f32(vdupq_n_f32(1.3951109720745927e-4f), shifted, polynomial);
    return vfmaq_f32(vdupq_n_f32(9.9997793886838715e-1f), shifted, polynomial);
}

static float32x4_t gelu4(float32x4_t value) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t two = vdupq_n_f32(2.0f);
    const float32x4_t four = vdupq_n_f32(4.0f);
    const float32x4_t scaled = vdivq_f32(value, vdupq_n_f32(1.4142135381698608f));
    const float32x4_t absolute = vabsq_f32(scaled);
    const uint32x4_t small = vcltq_f32(absolute, one);
    float32x4_t result;
    if (vminvq_u32(small) != 0u) {
        result = erf_small_magnitude_f32(absolute, vmulq_f32(absolute, absolute));
    } else {
        const uint32x4_t middle = vcltq_f32(absolute, two);
        const float32x4_t capped = vminnmq_f32(absolute, four);
        result = vbslq_f32(middle, erf_middle_magnitude_f32(capped),
                           erf_large_magnitude_f32(capped));
        if (vmaxvq_u32(small) != 0u) {
            result = vbslq_f32(small, erf_small_magnitude_f32(capped, vmulq_f32(capped, capped)),
                               result);
        }
        result = vbslq_f32(vcgeq_f32(absolute, four), one, result);
        result = vbslq_f32(vceqq_f32(scaled, scaled), result, scaled);
    }
    const uint32x4_t sign = vandq_u32(vreinterpretq_u32_f32(scaled), vdupq_n_u32(0x80000000u));
    result = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(result), sign));
    return vmulq_f32(vmulq_f32(value, vaddq_f32(result, one)), vdupq_n_f32(0.5f));
}
#endif

void lw_neon_gelu_f32(const float* input, float* output, uint64_t count) {
#if defined(_M_ARM64) || defined(__aarch64__)
    const uintptr_t source = (uintptr_t)input;
    const uintptr_t destination = (uintptr_t)output;
    const uint64_t vector_count = count - count % 4u;
    if (count == 0u) {
        return;
    }
    if (destination > source && (destination - source) / sizeof(float) < count) {
        lw_scalar_gelu_exact_f32(input + vector_count, output + vector_count, count - vector_count);
        for (uint64_t end = vector_count; end != 0u;) {
            end -= 4u;
            vst1q_f32(output + end, gelu4(vld1q_f32(input + end)));
        }
        return;
    }
    for (uint64_t i = 0u; i < vector_count; i += 4u) {
        vst1q_f32(output + i, gelu4(vld1q_f32(input + i)));
    }
    lw_scalar_gelu_exact_f32(input + vector_count, output + vector_count, count - vector_count);
#else
    lw_scalar_gelu_exact_f32(input, output, count);
#endif
}
