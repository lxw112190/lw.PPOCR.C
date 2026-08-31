#include "cpu_features.h"

#include <stdio.h>

int main(void) {
    const lw_simd_level level = lw_detect_simd_level();
    const char* name = lw_simd_level_name(level);
    printf("SIMD backend: %s\n", name);

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
