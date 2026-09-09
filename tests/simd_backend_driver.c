#include "cpu_features.h"

#include <stdio.h>

int main(void) {
    const lw_cpu_capabilities capabilities = lw_get_cpu_capabilities();
    const lw_cpu_capabilities second_capabilities = lw_get_cpu_capabilities();
    const lw_simd_level level = capabilities.simd;
    const char* name = lw_simd_level_name(level);
    if (capabilities.simd != second_capabilities.simd ||
        capabilities.has_fma != second_capabilities.has_fma ||
        capabilities.has_avx2_fma != second_capabilities.has_avx2_fma) {
        fprintf(stderr, "CPU capability snapshot is not stable\n");
        return 1;
    }
    printf("SIMD backend: %s\n", name);
    printf("CPU features: fma=%d avx2_fma=%d\n", capabilities.has_fma,
           capabilities.has_avx2_fma);

    if (capabilities.has_avx2_fma &&
        (!lw_simd_level_is_avx2(level) || !capabilities.has_fma)) {
        fprintf(stderr, "AVX2+FMA capability flags are inconsistent\n");
        return 1;
    }

#if defined(_M_ARM64) || defined(__aarch64__)
    if (!lw_simd_level_is_neon(level)) {
        fprintf(stderr, "AArch64 build did not select the NEON backend\n");
        return 1;
    }
#elif defined(__loongarch__)
    if (level != LW_SIMD_LEVEL_SCALAR && !lw_simd_level_is_lsx(level)) {
        fprintf(stderr, "LoongArch build selected an invalid SIMD backend\n");
        return 1;
    }
#endif

    return 0;
}
