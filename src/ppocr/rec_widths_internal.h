#ifndef LW_REC_WIDTHS_INTERNAL_H
#define LW_REC_WIDTHS_INTERNAL_H

#include <stdint.h>

/* Fine buckets are experimental because recognizer padding can change
 * attention and decoding. Keep the default policy at five widths. */
#if defined(__EMSCRIPTEN__) && defined(LW_WASM_FINE_REC_WIDTHS)
#define LW_REC_RESIDENT_WIDTH_COUNT 13u
static const uint32_t lw_rec_adaptive_widths[LW_REC_RESIDENT_WIDTH_COUNT] = {
    192u, 256u, 320u, 384u, 448u, 512u, 576u,
    640u, 704u, 768u, 832u, 896u, 960u
};
#elif !defined(__EMSCRIPTEN__) && defined(LW_NATIVE_FINE_REC_WIDTHS)
/* Preserve every production bucket. A finer policy must not increase padding
 * for content that already fits 480; the WASM experiment stays independent. */
#define LW_REC_RESIDENT_WIDTH_COUNT 14u
static const uint32_t lw_rec_adaptive_widths[LW_REC_RESIDENT_WIDTH_COUNT] = {
    192u, 256u, 320u, 384u, 448u, 480u, 512u, 576u,
    640u, 704u, 768u, 832u, 896u, 960u
};
#elif !defined(__EMSCRIPTEN__) && defined(LW_ARM64_FINE_REC_WIDTHS)
/* Preserve the production buckets and add two steps in the widest gap. */
#define LW_REC_RESIDENT_WIDTH_COUNT 10u
static const uint32_t lw_rec_adaptive_widths[LW_REC_RESIDENT_WIDTH_COUNT] = {
    192u, 256u, 320u, 384u, 480u, 576u, 640u, 768u, 896u, 960u
};
#else
#define LW_REC_RESIDENT_WIDTH_COUNT 5u
static const uint32_t lw_rec_adaptive_widths[LW_REC_RESIDENT_WIDTH_COUNT] = {
    192u, 320u, 480u, 640u, 960u
};
#endif

#endif
