#include "rec_widths_internal.h"
#include <stdio.h>

int main(void) {
#ifdef LW_TEST_FINE_WIDTHS
    static const uint32_t expected[] = {
        192u, 256u, 320u, 384u, 448u, 480u, 512u, 576u,
        640u, 704u, 768u, 832u, 896u, 960u
    };
#else
    static const uint32_t expected[] = {192u, 320u, 480u, 640u, 960u};
#endif
    uint32_t index;
    /* Exercise the header in Release without assert/NDEBUG or MSVC C4127. */
    volatile uint32_t actual_count = LW_REC_RESIDENT_WIDTH_COUNT;
    if (actual_count != sizeof(expected) / sizeof(expected[0])) {
        fprintf(stderr, "wrong REC width count\n");
        return 1;
    }
    for (index = 0u; index < LW_REC_RESIDENT_WIDTH_COUNT; ++index) {
        if (lw_rec_adaptive_widths[index] != expected[index]) {
            fprintf(stderr, "wrong REC width at %u\n", (unsigned)index);
            return 1;
        }
    }
#ifdef LW_TEST_FINE_WIDTHS
    /* Finer native buckets must never round a line above its old bucket. */
    {
        static const uint32_t production[] = {192u, 320u, 480u, 640u, 960u};
        uint32_t content;
        for (content = 1u; content <= 960u; ++content) {
            uint32_t old_index = 0u;
            uint32_t fine_index = 0u;
            while (production[old_index] < content) ++old_index;
            while (lw_rec_adaptive_widths[fine_index] < content) ++fine_index;
            if (lw_rec_adaptive_widths[fine_index] > production[old_index]) {
                fprintf(stderr, "fine policy increases padding at %u\n", (unsigned)content);
                return 1;
            }
        }
    }
#endif
    return 0;
}
