#ifndef LW_LWM_READ_H
#define LW_LWM_READ_H

#include <stdint.h>

static inline uint16_t lwm_read_u16(const uint8_t* p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static inline uint32_t lwm_read_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline int32_t lwm_read_i32(const uint8_t* p) {
    return (int32_t)lwm_read_u32(p);
}

static inline uint64_t lwm_read_u64(const uint8_t* p) {
    return (uint64_t)lwm_read_u32(p) | ((uint64_t)lwm_read_u32(p + 4) << 32);
}

#endif
