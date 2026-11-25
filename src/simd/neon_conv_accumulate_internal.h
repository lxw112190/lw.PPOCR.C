#ifndef LW_NEON_CONV_ACCUMULATE_INTERNAL_H
#define LW_NEON_CONV_ACCUMULATE_INTERNAL_H

/* One input vector updates four output channels. Keep separate FP32 multiply
 * and add instructions, with four independent products in flight. GCC 9
 * otherwise spills accumulators when scheduling the full 4x16 tile. */
#if defined(__GNUC__) && !defined(__clang__)
#define LW_NEON_ACCUMULATE4(input, weights, sum0, sum1, sum2, sum3)                  \
    do {                                                                           \
        float32x4_t product0, product1, product2, product3;                         \
        __asm__(                                                                   \
            "fmul %[p0].4s, %[x].4s, %[w].s[0]\n\t"                                 \
            "fmul %[p1].4s, %[x].4s, %[w].s[1]\n\t"                                 \
            "fmul %[p2].4s, %[x].4s, %[w].s[2]\n\t"                                 \
            "fmul %[p3].4s, %[x].4s, %[w].s[3]\n\t"                                 \
            "fadd %[s0].4s, %[s0].4s, %[p0].4s\n\t"                                 \
            "fadd %[s1].4s, %[s1].4s, %[p1].4s\n\t"                                 \
            "fadd %[s2].4s, %[s2].4s, %[p2].4s\n\t"                                 \
            "fadd %[s3].4s, %[s3].4s, %[p3].4s"                                     \
            : [s0] "+w"(sum0), [s1] "+w"(sum1), [s2] "+w"(sum2), [s3] "+w"(sum3),    \
              [p0] "=&w"(product0), [p1] "=&w"(product1),                            \
              [p2] "=&w"(product2), [p3] "=&w"(product3)                             \
            : [x] "w"(input), [w] "w"(weights));                                    \
    } while (0)
#else
#define LW_NEON_ACCUMULATE4(input, weights, sum0, sum1, sum2, sum3)                  \
    do {                                                                           \
        (sum0) = vaddq_f32((sum0), vmulq_laneq_f32((input), (weights), 0));            \
        (sum1) = vaddq_f32((sum1), vmulq_laneq_f32((input), (weights), 1));            \
        (sum2) = vaddq_f32((sum2), vmulq_laneq_f32((input), (weights), 2));            \
        (sum3) = vaddq_f32((sum3), vmulq_laneq_f32((input), (weights), 3));            \
    } while (0)
#endif

#endif
