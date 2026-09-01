#ifndef LW_CPU_TOPOLOGY_H
#define LW_CPU_TOPOLOGY_H

/* Best-effort CPU availability used by the private parallel policy. The
 * values describe processors available to this process, not necessarily the
 * complete host, so affinity and common Linux CPU quotas can reduce them. */

#include <stdint.h>

typedef struct lw_cpu_topology {
    uint32_t logical_processors;
    uint32_t physical_cores;
    uint32_t available_processors;
    uint32_t smt_width;
} lw_cpu_topology;

void lw_cpu_topology_detect(lw_cpu_topology* topology);
uint32_t lw_parallel_default_line_worker_count(const lw_cpu_topology* topology);
uint32_t lw_parallel_default_det_thread_count(const lw_cpu_topology* topology);

#endif
