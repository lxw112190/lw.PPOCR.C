#ifndef LW_CPU_FEATURES_H
#define LW_CPU_FEATURES_H

/*
 * Backends are mutually exclusive architecture implementations, not an
 * ordered capability level. Callers must use the helpers below instead of
 * comparing enum values: NEON and LSX are not supersets of SSE2/AVX2.
 */

typedef enum lw_simd_level {
    LW_SIMD_LEVEL_SCALAR = 0,
    LW_SIMD_LEVEL_SSE2 = 1,
    LW_SIMD_LEVEL_AVX2 = 2,
    LW_SIMD_LEVEL_NEON = 3,
    LW_SIMD_LEVEL_LSX = 4,
    LW_SIMD_LEVEL_LASX = 5
} lw_simd_level;

static inline int lw_simd_level_is_sse2(lw_simd_level level) {
    return level == LW_SIMD_LEVEL_SSE2 || level == LW_SIMD_LEVEL_AVX2;
}

static inline int lw_simd_level_is_avx2(lw_simd_level level) {
    return level == LW_SIMD_LEVEL_AVX2;
}

static inline int lw_simd_level_is_neon(lw_simd_level level) {
    return level == LW_SIMD_LEVEL_NEON;
}

static inline int lw_simd_level_is_lsx(lw_simd_level level) {
    /* Every LASX processor also provides the 128-bit LSX instruction set. */
    return level == LW_SIMD_LEVEL_LSX || level == LW_SIMD_LEVEL_LASX;
}

static inline int lw_simd_level_has_packed_conv1x1(lw_simd_level level) {
    return lw_simd_level_is_sse2(level) || lw_simd_level_is_neon(level) ||
           lw_simd_level_is_lsx(level);
}

lw_simd_level lw_detect_simd_level(void);
const char* lw_simd_level_name(lw_simd_level level);

#endif
