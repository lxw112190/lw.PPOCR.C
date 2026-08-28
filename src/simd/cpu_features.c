#include "cpu_features.h"

/*
 * One-time runtime CPU detection. Compiling AVX2 objects does not mean the host
 * CPU supports AVX2, so callers must always dispatch through this result.
 */

#include <stddef.h>

#if defined(__EMSCRIPTEN__) && defined(__wasm_simd128__)
#  define LW_EMSCRIPTEN_SIMD128 1
#else
#  define LW_EMSCRIPTEN_SIMD128 0
#endif

#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
#  include <intrin.h>
#elif defined(__i386__) || defined(__x86_64__)
#  include <cpuid.h>
#endif

lw_simd_level lw_detect_simd_level(void) {
#if LW_EMSCRIPTEN_SIMD128
    /* The module itself requires SIMD128, so no runtime CPUID probe exists. */
    return LW_SIMD_LEVEL_SSE2;
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
    int registers[4];
    int maximum_leaf;
    int has_sse2;
    __cpuid(registers, 0);
    maximum_leaf = registers[0];
    if (maximum_leaf < 1) {
        return LW_SIMD_LEVEL_SCALAR;
    }
    __cpuid(registers, 1);
    has_sse2 = (registers[3] & (1 << 26)) != 0;
    /* AVX2 is safe only when both hardware and the operating system preserve
     * XMM/YMM state. XGETBV verifies the OS-enabled extended-state bits. */
    if (maximum_leaf >= 7 && (registers[2] & (1 << 27)) != 0 && (registers[2] & (1 << 28)) != 0 &&
        (_xgetbv(0) & 6u) == 6u) {
        __cpuidex(registers, 7, 0);
        if ((registers[1] & (1 << 5)) != 0) {
            return LW_SIMD_LEVEL_AVX2;
        }
    }
    return has_sse2 ? LW_SIMD_LEVEL_SSE2 : LW_SIMD_LEVEL_SCALAR;
#elif defined(__i386__) || defined(__x86_64__)
    unsigned int eax;
    unsigned int ebx;
    unsigned int ecx;
    unsigned int edx;
    unsigned int maximum_leaf = __get_cpuid_max(0u, NULL);
    int has_sse2;
    if (maximum_leaf < 1u || __get_cpuid(1u, &eax, &ebx, &ecx, &edx) == 0) {
        return LW_SIMD_LEVEL_SCALAR;
    }
    has_sse2 = (edx & bit_SSE2) != 0u;
    /* GCC/Clang branch performs the same hardware + OS state check. */
    if (maximum_leaf >= 7u && (ecx & bit_OSXSAVE) != 0u && (ecx & bit_AVX) != 0u) {
        unsigned int xcr0_eax;
        unsigned int xcr0_edx;
        __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(xcr0_eax), "=d"(xcr0_edx) : "c"(0u));
        (void)xcr0_edx;
        if ((xcr0_eax & 6u) == 6u) {
            __cpuid_count(7u, 0u, eax, ebx, ecx, edx);
            if ((ebx & bit_AVX2) != 0u) {
                return LW_SIMD_LEVEL_AVX2;
            }
        }
    }
    return has_sse2 ? LW_SIMD_LEVEL_SSE2 : LW_SIMD_LEVEL_SCALAR;
#else
    return LW_SIMD_LEVEL_SCALAR;
#endif
}

const char* lw_simd_level_name(lw_simd_level level) {
    if (level >= LW_SIMD_LEVEL_AVX2) {
        return "avx2";
    }
    return level >= LW_SIMD_LEVEL_SSE2 ? "sse2" : "scalar";
}
