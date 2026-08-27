#ifndef LW_PARALLEL_INTERNAL_H
#define LW_PARALLEL_INTERNAL_H

/* Small dependency-free parallel-for used by coarse inference kernels. */

#include <stdint.h>

#define LW_PARALLEL_MAX_WORKERS 16u

typedef void (*lw_parallel_callback)(void* context, uint32_t worker_index,
                                     uint32_t worker_count);

typedef struct lw_thread_pool lw_thread_pool;

lw_thread_pool* lw_thread_pool_create(uint32_t worker_count);
void lw_thread_pool_free(lw_thread_pool* pool);
uint32_t lw_thread_pool_worker_count(const lw_thread_pool* pool);
void lw_thread_pool_run(lw_thread_pool* pool, uint32_t worker_count,
                        lw_parallel_callback callback, void* context);

#endif
