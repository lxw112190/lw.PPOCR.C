#ifndef LW_SIMD_PLATFORM_H
#define LW_SIMD_PLATFORM_H

/*
 * The Emscripten toolchain translates the portable SSE2 intrinsic subset to
 * WebAssembly SIMD128 when the translation unit is compiled with
 * -msimd128.  Keeping this small compatibility layer lets the existing SSE2
 * kernels serve as the first WASM SIMD backend without changing their
 * numerical algorithms or the public runtime ABI.
 */
#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__) || \
    (defined(__EMSCRIPTEN__) && defined(__wasm_simd128__))
#  include <emmintrin.h>
#  define LW_SIMD_HAS_SSE2_INTRINSICS 1
#else
#  define LW_SIMD_HAS_SSE2_INTRINSICS 0
#endif

/* target("sse2") is meaningful only for native x86 Clang/GCC builds. */
#if LW_SIMD_HAS_SSE2_INTRINSICS && !defined(__EMSCRIPTEN__) && \
    (defined(__GNUC__) || defined(__clang__))
#  define LW_SIMD_SSE2_TARGET __attribute__((target("sse2")))
#else
#  define LW_SIMD_SSE2_TARGET
#endif

#endif
