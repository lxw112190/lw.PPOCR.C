#include "cpu_features.h"

#if defined(_M_IX86) || defined(_M_X64)
#  include <intrin.h>
#elif defined(__i386__) || defined(__x86_64__)
#  include <cpuid.h>
#endif

lw_simd_level lw_detect_simd_level(void) {
#if defined(_M_X64) || defined(__x86_64__)
    return LW_SIMD_LEVEL_SSE2;
#elif defined(_M_IX86)
    int registers[4];
    __cpuid(registers, 1);
    return (registers[3] & (1 << 26)) != 0 ?
        LW_SIMD_LEVEL_SSE2 : LW_SIMD_LEVEL_SCALAR;
#elif defined(__i386__)
    unsigned int eax;
    unsigned int ebx;
    unsigned int ecx;
    unsigned int edx;
    if (__get_cpuid(1u, &eax, &ebx, &ecx, &edx) != 0 &&
        (edx & bit_SSE2) != 0u) {
        return LW_SIMD_LEVEL_SSE2;
    }
    return LW_SIMD_LEVEL_SCALAR;
#else
    return LW_SIMD_LEVEL_SCALAR;
#endif
}

const char* lw_simd_level_name(lw_simd_level level) {
    return level >= LW_SIMD_LEVEL_SSE2 ? "sse2" : "scalar";
}
