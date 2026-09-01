#include "cpu_topology.h"

#include <stdint.h>
#include <stdio.h>

int main(void) {
    lw_cpu_topology topology;
    lw_cpu_topology four_core = {8u, 4u, 8u, 2u};
    lw_cpu_topology quota_limited = {16u, 8u, 4u, 2u};
    lw_cpu_topology large_host = {128u, 64u, 128u, 2u};
    uint32_t line_workers;
    uint32_t det_threads;
    lw_cpu_topology_detect(&topology);
    line_workers = lw_parallel_default_line_worker_count(&topology);
    det_threads = lw_parallel_default_det_thread_count(&topology);
    if (topology.logical_processors == 0u || topology.physical_cores == 0u ||
        topology.available_processors == 0u || topology.smt_width == 0u ||
        topology.available_processors > topology.logical_processors ||
        topology.physical_cores > topology.logical_processors || line_workers == 0u ||
        line_workers > 8u || det_threads == 0u || det_threads > 8u ||
        det_threads > topology.available_processors) {
        fprintf(stderr,
                "invalid topology: logical=%u physical=%u available=%u smt=%u "
                "line_workers=%u det_threads=%u\n",
                topology.logical_processors, topology.physical_cores, topology.available_processors,
                topology.smt_width, line_workers, det_threads);
        return 1;
    }
#if INTPTR_MAX <= INT32_MAX || defined(__EMSCRIPTEN__)
    if (line_workers != 1u || det_threads != 1u ||
        lw_parallel_default_line_worker_count(&four_core) != 1u ||
        lw_parallel_default_det_thread_count(&four_core) != 1u ||
        lw_parallel_default_line_worker_count(&quota_limited) != 1u ||
        lw_parallel_default_det_thread_count(&quota_limited) != 1u ||
        lw_parallel_default_line_worker_count(&large_host) != 1u ||
        lw_parallel_default_det_thread_count(&large_host) != 1u) {
        return 1;
    }
#else
    if (lw_parallel_default_line_worker_count(&four_core) != 8u ||
        lw_parallel_default_det_thread_count(&four_core) != 4u ||
        lw_parallel_default_line_worker_count(&quota_limited) != 4u ||
        lw_parallel_default_det_thread_count(&quota_limited) != 4u ||
        lw_parallel_default_line_worker_count(&large_host) != 8u ||
        lw_parallel_default_det_thread_count(&large_host) != 8u ||
        lw_parallel_default_line_worker_count(NULL) != 4u ||
        lw_parallel_default_det_thread_count(NULL) != 4u) {
        return 1;
    }
#endif
    printf("logical=%u physical=%u available=%u smt=%u line_workers=%u det_threads=%u\n",
           topology.logical_processors, topology.physical_cores, topology.available_processors,
           topology.smt_width, line_workers, det_threads);
    return 0;
}
