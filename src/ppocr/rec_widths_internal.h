#ifndef LW_REC_WIDTHS_INTERNAL_H
#define LW_REC_WIDTHS_INTERNAL_H

#include <stdint.h>

/* Private width policy shared with the lazy-fallback contract. Fine buckets
 * are opt-in for compiled WASM/native. Other builds keep five widths.
 * Changing padding can affect attention/decoding, so native is experimental
 * and requires a separate accuracy review; it is not a byte-parity kernel. */
#if (defined(__EMSCRIPTEN__) && defined(LW_WASM_FINE_REC_WIDTHS)) || \
    (!defined(__EMSCRIPTEN__) && defined(LW_NATIVE_FINE_REC_WIDTHS))
#define LW_REC_RESIDENT_WIDTH_COUNT 13u
static const uint32_t lw_rec_adaptive_widths[LW_REC_RESIDENT_WIDTH_COUNT] = {
    192u, 256u, 320u, 384u, 448u, 512u, 576u,
    640u, 704u, 768u, 832u, 896u, 960u
};
#else
#define LW_REC_RESIDENT_WIDTH_COUNT 5u
static const uint32_t lw_rec_adaptive_widths[LW_REC_RESIDENT_WIDTH_COUNT] = {
    192u, 320u, 480u, 640u, 960u
};
#endif

#endif
