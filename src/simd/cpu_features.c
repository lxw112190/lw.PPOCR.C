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

#if defined(__linux__) && defined(__loongarch__)
#  include <sys/auxv.h>
#  include <asm/hwcap.h>
#  ifndef HWCAP_LOONGARCH_LSX
#    define HWCAP_LOONGARCH_LSX (1UL << 4)
#  endif
#  ifndef HWCAP_LOONGARCH_LASX
#    define HWCAP_LOONGARCH_LASX (1UL << 5)
#  endif
#endif

static lw_simd_level lw_detect_simd_level_uncached(void) {
#if LW_EMSCRIPTEN_SIMD128
    /* The module itself requires SIMD128, so no runtime CPUID probe exists. */
    return LW_SIMD_LEVEL_SSE2;
#elif defined(_M_ARM64) || defined(__aarch64__)
    /* Advanced SIMD is part of the AArch64 execution environment. */
    return LW_SIMD_LEVEL_NEON;
#elif defined(__linux__) && defined(__loongarch__)
    {
        const unsigned long capabilities = getauxval(AT_HWCAP);
        if ((capabilities & HWCAP_LOONGARCH_LASX) != 0u) {
            return LW_SIMD_LEVEL_LASX;
        }
        if ((capabilities & HWCAP_LOONGARCH_LSX) != 0u) {
            return LW_SIMD_LEVEL_LSX;
        }
        return LW_SIMD_LEVEL_SCALAR;
    }
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

lw_simd_level lw_detect_simd_level(void) {
#if defined(_MSC_VER)
    static volatile long cached = -1;
    long value = _InterlockedCompareExchange(&cached, -1, -1);
    if (value >= 0) {
        return (lw_simd_level)value;
    }
    value = (long)lw_detect_simd_level_uncached();
    if (_InterlockedCompareExchange(&cached, value, -1) != -1) {
        value = _InterlockedCompareExchange(&cached, -1, -1);
    }
    return (lw_simd_level)value;
#elif defined(__GNUC__) || defined(__clang__)
    static int cached = -1;
    int value = __atomic_load_n(&cached, __ATOMIC_ACQUIRE);
    int expected = -1;
    if (value >= 0) {
        return (lw_simd_level)value;
    }
    value = (int)lw_detect_simd_level_uncached();
    (void)__atomic_compare_exchange_n(
        &cached, &expected, value, 0, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE);
    return (lw_simd_level)__atomic_load_n(&cached, __ATOMIC_ACQUIRE);
#else
    static int cached = -1;
    if (cached < 0) {
        cached = (int)lw_detect_simd_level_uncached();
    }
    return (lw_simd_level)cached;
#endif
}
const char* lw_simd_level_name(lw_simd_level level) {
    if (level == LW_SIMD_LEVEL_AVX2) {
        return "avx2";
    }
    if (level == LW_SIMD_LEVEL_SSE2) {
        return "sse2";
    }
    if (level == LW_SIMD_LEVEL_NEON) {
        return "neon";
    }
    if (level == LW_SIMD_LEVEL_LSX) {
        return "lsx";
    }
    if (level == LW_SIMD_LEVEL_LASX) {
        return "lasx";
    }
    return "scalar";
}

static int lw_detect_fma(lw_simd_level simd) {
    (void)simd;
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
    int registers[4];
    int maximum_leaf;
    __cpuid(registers, 0);
    maximum_leaf = registers[0];
    if (maximum_leaf < 1) {
        return 0;
    }
    __cpuid(registers, 1);
    if ((registers[2] & (1 << 12)) == 0 || (registers[2] & (1 << 27)) == 0 ||
        (registers[2] & (1 << 28)) == 0 || (_xgetbv(0) & 6u) != 6u) {
        return 0;
    }
    return 1;
#elif (defined(__i386__) || defined(__x86_64__)) && !defined(__EMSCRIPTEN__)
    unsigned int eax;
    unsigned int ebx;
    unsigned int ecx;
    unsigned int edx;
    unsigned int maximum_leaf;
    unsigned int xcr0_eax;
    unsigned int xcr0_edx;
    maximum_leaf = __get_cpuid_max(0u, NULL);
    if (maximum_leaf < 1u || __get_cpuid(1u, &eax, &ebx, &ecx, &edx) == 0) {
        return 0;
    }
    if ((ecx & bit_FMA) == 0u || (ecx & bit_OSXSAVE) == 0u || (ecx & bit_AVX) == 0u) {
        return 0;
    }
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0"
                     : "=a"(xcr0_eax), "=d"(xcr0_edx)
                     : "c"(0u));
    (void)xcr0_edx;
    return (xcr0_eax & 6u) == 6u;
#else
    (void)simd;
    return 0;
#endif
}

lw_cpu_capabilities lw_get_cpu_capabilities(void) {
    static lw_cpu_capabilities cached;
#if defined(_MSC_VER)
    static volatile long state;
    long observed = _InterlockedCompareExchange(&state, 1, 0);
    if (observed == 0) {
        cached.simd = lw_detect_simd_level();
        cached.has_fma = lw_detect_fma(cached.simd);
        cached.has_avx2_fma = lw_simd_level_is_avx2(cached.simd) && cached.has_fma;
        _InterlockedExchange(&state, 2);
    } else {
        while (_InterlockedCompareExchange(&state, 2, 2) != 2) {
        }
    }
#elif defined(__GNUC__) || defined(__clang__)
    static int state;
    int expected = 0;
    if (__atomic_compare_exchange_n(
            &state, &expected, 1, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
        cached.simd = lw_detect_simd_level();
        cached.has_fma = lw_detect_fma(cached.simd);
        cached.has_avx2_fma = lw_simd_level_is_avx2(cached.simd) && cached.has_fma;
        __atomic_store_n(&state, 2, __ATOMIC_RELEASE);
    } else {
        while (__atomic_load_n(&state, __ATOMIC_ACQUIRE) != 2) {
        }
    }
#else
    static int initialized;
    if (!initialized) {
        cached.simd = lw_detect_simd_level();
        cached.has_fma = lw_detect_fma(cached.simd);
        cached.has_avx2_fma = lw_simd_level_is_avx2(cached.simd) && cached.has_fma;
        initialized = 1;
    }
#endif
    return cached;
}
