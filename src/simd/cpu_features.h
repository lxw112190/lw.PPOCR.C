#ifndef LW_CPU_FEATURES_H
#define LW_CPU_FEATURES_H

typedef enum lw_simd_level {
    LW_SIMD_LEVEL_SCALAR = 0,
    LW_SIMD_LEVEL_SSE2 = 1,
    LW_SIMD_LEVEL_AVX2 = 2
} lw_simd_level;

lw_simd_level lw_detect_simd_level(void);
const char* lw_simd_level_name(lw_simd_level level);

#endif
