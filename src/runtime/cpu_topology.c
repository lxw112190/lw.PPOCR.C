#if defined(__linux__) && !defined(_GNU_SOURCE)
#  define _GNU_SOURCE
#endif

#include "cpu_topology.h"

/* CPU discovery is deliberately dependency-free and best effort. Failure to
 * inspect topology must reduce parallelism conservatively, never prevent OCR. */

#include <errno.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  if !defined(_WIN32_WINNT)
#    define _WIN32_WINNT 0x0601
#  endif
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#elif defined(__APPLE__)
#  include <sys/sysctl.h>
#elif !defined(__EMSCRIPTEN__)
#  include <unistd.h>
#  if defined(__linux__)
#    include <sched.h>
#  endif
#endif

#define LW_DEFAULT_NATIVE_PARALLEL_CAP 8u
#define LW_DEFAULT_NATIVE_PROCESSOR_FALLBACK 4u

#if !defined(_WIN32) && !defined(__APPLE__) && !defined(__EMSCRIPTEN__)
static uint32_t clamp_positive_u64(uint64_t value) {
    if (value == 0u) {
        return 1u;
    }
    return value > UINT32_MAX ? UINT32_MAX : (uint32_t)value;
}
#endif

#if defined(_WIN32)
static uint32_t count_affinity_bits(DWORD_PTR mask) {
    uint32_t count = 0u;
    while (mask != 0u) {
        count += (uint32_t)(mask & 1u);
        mask >>= 1u;
    }
    return count;
}

static uint32_t windows_physical_core_count(void) {
    DWORD bytes = 0u;
    uint8_t* buffer;
    uint8_t* cursor;
    uint8_t* end;
    uint32_t count = 0u;
    if (GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &bytes) != FALSE ||
        GetLastError() != ERROR_INSUFFICIENT_BUFFER || bytes == 0u) {
        return 0u;
    }
    buffer = (uint8_t*)malloc(bytes);
    if (buffer == NULL) {
        return 0u;
    }
    if (!GetLogicalProcessorInformationEx(
            RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buffer, &bytes)) {
        free(buffer);
        return 0u;
    }
    cursor = buffer;
    end = buffer + bytes;
    while ((size_t)(end - cursor) >= sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)) {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX item =
            (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(void*)cursor;
        if (item->Size < sizeof(*item) || (size_t)(end - cursor) < item->Size) {
            break;
        }
        if (item->Relationship == RelationProcessorCore && count != UINT32_MAX) {
            ++count;
        }
        cursor += item->Size;
    }
    free(buffer);
    return count;
}
#endif

#if defined(__linux__) && !defined(__EMSCRIPTEN__)
typedef struct lw_linux_core_key {
    int package_id;
    int core_id;
} lw_linux_core_key;

static int read_linux_topology_value(uint32_t cpu, const char* name, int* value) {
    char path[160];
    FILE* file;
    int written =
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u/topology/%s", cpu, name);
    if (written <= 0 || (size_t)written >= sizeof(path)) {
        return 0;
    }
    file = fopen(path, "r");
    if (file == NULL) {
        return 0;
    }
    if (fscanf(file, "%d", value) != 1) {
        fclose(file);
        return 0;
    }
    return fclose(file) == 0;
}

static uint32_t linux_cpu_quota(void) {
    FILE* file = fopen("/sys/fs/cgroup/cpu.max", "r");
    char quota_text[32];
    unsigned long long quota;
    unsigned long long period;
    if (file != NULL) {
        int fields = fscanf(file, "%31s %llu", quota_text, &period);
        fclose(file);
        if (fields == 2 && strcmp(quota_text, "max") != 0) {
            char* end = NULL;
            errno = 0;
            quota = strtoull(quota_text, &end, 10);
            if (errno == 0 && end != quota_text && *end == '\0' && quota > 0u && period > 0u) {
                return clamp_positive_u64((quota + period - 1u) / period);
            }
        }
    }
    file = fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r");
    if (file != NULL) {
        long long signed_quota = -1;
        int fields = fscanf(file, "%lld", &signed_quota);
        fclose(file);
        if (fields == 1 && signed_quota > 0) {
            file = fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r");
            if (file != NULL) {
                int period_fields = fscanf(file, "%llu", &period);
                fclose(file);
                if (period_fields == 1 && period > 0u) {
                    quota = (unsigned long long)signed_quota;
                    return clamp_positive_u64((quota + period - 1u) / period);
                }
            }
        }
    }
    return 0u;
}

static void detect_linux_topology(lw_cpu_topology* topology) {
    cpu_set_t affinity;
    lw_linux_core_key keys[CPU_SETSIZE];
    uint32_t logical = 0u;
    uint32_t physical = 0u;
    uint32_t cpu;
    uint32_t quota;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0) {
        for (cpu = 0u; cpu < (uint32_t)CPU_SETSIZE; ++cpu) {
            int package_id;
            int core_id;
            uint32_t index;
            if (!CPU_ISSET((int)cpu, &affinity)) {
                continue;
            }
            ++logical;
            if (!read_linux_topology_value(cpu, "physical_package_id", &package_id) ||
                !read_linux_topology_value(cpu, "core_id", &core_id)) {
                continue;
            }
            for (index = 0u; index < physical; ++index) {
                if (keys[index].package_id == package_id && keys[index].core_id == core_id) {
                    break;
                }
            }
            if (index == physical && physical < (uint32_t)CPU_SETSIZE) {
                keys[physical].package_id = package_id;
                keys[physical].core_id = core_id;
                ++physical;
            }
        }
    }
    if (logical == 0u) {
        long online = sysconf(_SC_NPROCESSORS_ONLN);
        logical = online > 0 ? clamp_positive_u64((uint64_t)online) : 1u;
    }
    quota = linux_cpu_quota();
    topology->logical_processors = logical;
    topology->available_processors = quota != 0u && quota < logical ? quota : logical;
    topology->physical_cores = physical == 0u ? topology->available_processors : physical;
}
#endif

void lw_cpu_topology_detect(lw_cpu_topology* topology) {
    uint32_t logical = 1u;
    uint32_t physical = 1u;
    uint32_t available = 1u;
    if (topology == NULL) {
        return;
    }
    memset(topology, 0, sizeof(*topology));
#if defined(__EMSCRIPTEN__)
    /* The offline module is intentionally built without pthreads. */
#elif defined(_WIN32)
    {
        SYSTEM_INFO system_info;
        DWORD_PTR process_mask = 0u;
        DWORD_PTR system_mask = 0u;
        GetSystemInfo(&system_info);
        logical = system_info.dwNumberOfProcessors == 0u
                      ? 1u
                      : (uint32_t)system_info.dwNumberOfProcessors;
        available = logical;
        if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask)) {
            uint32_t affinity_count = count_affinity_bits(process_mask);
            if (affinity_count != 0u && affinity_count < available) {
                available = affinity_count;
            }
        }
        physical = windows_physical_core_count();
        if (physical == 0u) {
            physical = available;
        }
        if (physical > available) {
            physical = available;
        }
    }
#elif defined(__linux__)
    detect_linux_topology(topology);
    logical = topology->logical_processors;
    physical = topology->physical_cores;
    available = topology->available_processors;
#elif defined(__APPLE__)
    {
        size_t size = sizeof(logical);
        if (sysctlbyname("hw.logicalcpu", &logical, &size, NULL, 0u) != 0 || logical == 0u) {
            logical = 1u;
        }
        size = sizeof(physical);
        if (sysctlbyname("hw.physicalcpu", &physical, &size, NULL, 0u) != 0 || physical == 0u) {
            physical = logical;
        }
        available = logical;
    }
#else
    {
        long online = sysconf(_SC_NPROCESSORS_ONLN);
        logical = online > 0 ? clamp_positive_u64((uint64_t)online) : 1u;
        physical = logical;
        available = logical;
    }
#endif
    topology->logical_processors = logical == 0u ? 1u : logical;
    topology->available_processors = available == 0u ? 1u : available;
    topology->physical_cores = physical == 0u ? topology->available_processors : physical;
    if (topology->available_processors > topology->logical_processors) {
        topology->available_processors = topology->logical_processors;
    }
    if (topology->physical_cores > topology->logical_processors) {
        topology->physical_cores = topology->logical_processors;
    }
    topology->smt_width =
        (topology->logical_processors + topology->physical_cores - 1u) / topology->physical_cores;
    if (topology->smt_width == 0u) {
        topology->smt_width = 1u;
    }
}

#if INTPTR_MAX > INT32_MAX && !defined(__EMSCRIPTEN__)
static uint32_t cap_native_count(uint32_t count) {
    if (count == 0u) {
        count = LW_DEFAULT_NATIVE_PROCESSOR_FALLBACK;
    }
    return count > LW_DEFAULT_NATIVE_PARALLEL_CAP ? LW_DEFAULT_NATIVE_PARALLEL_CAP : count;
}
#endif

uint32_t lw_parallel_default_line_worker_count(const lw_cpu_topology* topology) {
#if INTPTR_MAX <= INT32_MAX || defined(__EMSCRIPTEN__)
    (void)topology;
    return 1u;
#else
    return cap_native_count(topology == NULL ? 0u : topology->available_processors);
#endif
}

uint32_t lw_parallel_default_det_thread_count(const lw_cpu_topology* topology) {
#if INTPTR_MAX <= INT32_MAX || defined(__EMSCRIPTEN__)
    (void)topology;
    return 1u;
#else
    uint32_t count = 0u;
    if (topology != NULL) {
        count = topology->physical_cores;
        if (topology->available_processors != 0u && topology->available_processors < count) {
            count = topology->available_processors;
        }
    }
    return cap_native_count(count);
#endif
}
