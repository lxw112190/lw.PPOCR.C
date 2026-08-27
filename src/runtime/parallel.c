#include "parallel_internal.h"

#include <stddef.h>
#include <stdlib.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <process.h>
#  include <windows.h>
#elif !defined(__EMSCRIPTEN__)
#  include <pthread.h>
#endif

typedef struct lw_pool_worker {
    struct lw_thread_pool* pool;
    uint32_t worker_index;
} lw_pool_worker;

struct lw_thread_pool {
    uint32_t worker_count;
    uint32_t active_worker_count;
    uint32_t pending_worker_count;
    uint64_t generation;
    uint32_t stopping;
    lw_parallel_callback callback;
    void* context;
    lw_pool_worker workers[LW_PARALLEL_MAX_WORKERS - 1u];
#if defined(_WIN32)
    CRITICAL_SECTION mutex;
    CONDITION_VARIABLE work_condition;
    CONDITION_VARIABLE done_condition;
    HANDLE threads[LW_PARALLEL_MAX_WORKERS - 1u];
#elif !defined(__EMSCRIPTEN__)
    pthread_mutex_t mutex;
    pthread_cond_t work_condition;
    pthread_cond_t done_condition;
    pthread_t threads[LW_PARALLEL_MAX_WORKERS - 1u];
#endif
};

#if defined(_WIN32)
static void lock_pool(lw_thread_pool* pool) {
    EnterCriticalSection(&pool->mutex);
}
static void unlock_pool(lw_thread_pool* pool) {
    LeaveCriticalSection(&pool->mutex);
}
static void wait_for_work(lw_thread_pool* pool) {
    (void)SleepConditionVariableCS(&pool->work_condition, &pool->mutex, INFINITE);
}
static void wait_for_done(lw_thread_pool* pool) {
    (void)SleepConditionVariableCS(&pool->done_condition, &pool->mutex, INFINITE);
}
static void wake_all_workers(lw_thread_pool* pool) {
    WakeAllConditionVariable(&pool->work_condition);
}
static void wake_caller(lw_thread_pool* pool) {
    WakeConditionVariable(&pool->done_condition);
}
#elif !defined(__EMSCRIPTEN__)
static void lock_pool(lw_thread_pool* pool) {
    (void)pthread_mutex_lock(&pool->mutex);
}
static void unlock_pool(lw_thread_pool* pool) {
    (void)pthread_mutex_unlock(&pool->mutex);
}
static void wait_for_work(lw_thread_pool* pool) {
    (void)pthread_cond_wait(&pool->work_condition, &pool->mutex);
}
static void wait_for_done(lw_thread_pool* pool) {
    (void)pthread_cond_wait(&pool->done_condition, &pool->mutex);
}
static void wake_all_workers(lw_thread_pool* pool) {
    (void)pthread_cond_broadcast(&pool->work_condition);
}
static void wake_caller(lw_thread_pool* pool) {
    (void)pthread_cond_signal(&pool->done_condition);
}
#endif

#if !defined(__EMSCRIPTEN__)
static void worker_loop(lw_pool_worker* worker) {
    lw_thread_pool* pool = worker->pool;
    uint64_t observed_generation = 0u;
    for (;;) {
        lw_parallel_callback callback;
        void* context;
        uint32_t active_worker_count;
        lock_pool(pool);
        while (pool->stopping == 0u && pool->generation == observed_generation) {
            wait_for_work(pool);
        }
        if (pool->stopping != 0u) {
            unlock_pool(pool);
            return;
        }
        observed_generation = pool->generation;
        callback = pool->callback;
        context = pool->context;
        active_worker_count = pool->active_worker_count;
        unlock_pool(pool);
        if (worker->worker_index >= active_worker_count) {
            continue;
        }
        callback(context, worker->worker_index, active_worker_count);
        lock_pool(pool);
        --pool->pending_worker_count;
        if (pool->pending_worker_count == 0u) {
            wake_caller(pool);
        }
        unlock_pool(pool);
    }
}
#endif

#if defined(_WIN32)
static unsigned __stdcall pool_worker_entry(void* context) {
    worker_loop((lw_pool_worker*)context);
    return 0u;
}
#elif !defined(__EMSCRIPTEN__)
static void* pool_worker_entry(void* context) {
    worker_loop((lw_pool_worker*)context);
    return NULL;
}
#endif

lw_thread_pool* lw_thread_pool_create(uint32_t worker_count) {
#if defined(__EMSCRIPTEN__)
    (void)worker_count;
    return NULL;
#else
    lw_thread_pool* pool;
    uint32_t worker_index;
    if (worker_count < 2u) {
        return NULL;
    }
    if (worker_count > LW_PARALLEL_MAX_WORKERS) {
        worker_count = LW_PARALLEL_MAX_WORKERS;
    }
    pool = (lw_thread_pool*)calloc(1u, sizeof(*pool));
    if (pool == NULL) {
        return NULL;
    }
#if defined(_WIN32)
    InitializeCriticalSection(&pool->mutex);
    InitializeConditionVariable(&pool->work_condition);
    InitializeConditionVariable(&pool->done_condition);
#else
    if (pthread_mutex_init(&pool->mutex, NULL) != 0) {
        free(pool);
        return NULL;
    }
    if (pthread_cond_init(&pool->work_condition, NULL) != 0) {
        (void)pthread_mutex_destroy(&pool->mutex);
        free(pool);
        return NULL;
    }
    if (pthread_cond_init(&pool->done_condition, NULL) != 0) {
        (void)pthread_cond_destroy(&pool->work_condition);
        (void)pthread_mutex_destroy(&pool->mutex);
        free(pool);
        return NULL;
    }
#endif
    pool->worker_count = 1u;
    for (worker_index = 1u; worker_index < worker_count; ++worker_index) {
        lw_pool_worker* worker = &pool->workers[worker_index - 1u];
        worker->pool = pool;
        worker->worker_index = worker_index;
#if defined(_WIN32)
        pool->threads[worker_index - 1u] =
            (HANDLE)_beginthreadex(NULL, 0u, pool_worker_entry, worker, 0u, NULL);
        if (pool->threads[worker_index - 1u] == NULL) {
            break;
        }
#else
        if (pthread_create(&pool->threads[worker_index - 1u], NULL, pool_worker_entry, worker) != 0) {
            break;
        }
#endif
        ++pool->worker_count;
    }
    if (pool->worker_count < 2u) {
        lw_thread_pool_free(pool);
        return NULL;
    }
    return pool;
#endif
}

void lw_thread_pool_free(lw_thread_pool* pool) {
#if defined(__EMSCRIPTEN__)
    (void)pool;
#else
    uint32_t worker_index;
    if (pool == NULL) {
        return;
    }
    lock_pool(pool);
    pool->stopping = 1u;
    ++pool->generation;
    wake_all_workers(pool);
    unlock_pool(pool);
    for (worker_index = 1u; worker_index < pool->worker_count; ++worker_index) {
#if defined(_WIN32)
        WaitForSingleObject(pool->threads[worker_index - 1u], INFINITE);
        CloseHandle(pool->threads[worker_index - 1u]);
#else
        (void)pthread_join(pool->threads[worker_index - 1u], NULL);
#endif
    }
#if defined(_WIN32)
    DeleteCriticalSection(&pool->mutex);
#else
    (void)pthread_cond_destroy(&pool->done_condition);
    (void)pthread_cond_destroy(&pool->work_condition);
    (void)pthread_mutex_destroy(&pool->mutex);
#endif
    free(pool);
#endif
}

uint32_t lw_thread_pool_worker_count(const lw_thread_pool* pool) {
    return pool == NULL ? 1u : pool->worker_count;
}

void lw_thread_pool_run(lw_thread_pool* pool, uint32_t worker_count,
                        lw_parallel_callback callback, void* context) {
#if defined(__EMSCRIPTEN__)
    (void)pool;
    (void)worker_count;
    if (callback != NULL) {
        callback(context, 0u, 1u);
    }
#else
    if (callback == NULL) {
        return;
    }
    if (pool == NULL || worker_count < 2u) {
        callback(context, 0u, 1u);
        return;
    }
    if (worker_count > pool->worker_count) {
        worker_count = pool->worker_count;
    }
    lock_pool(pool);
    pool->callback = callback;
    pool->context = context;
    pool->active_worker_count = worker_count;
    pool->pending_worker_count = worker_count - 1u;
    ++pool->generation;
    wake_all_workers(pool);
    unlock_pool(pool);
    callback(context, 0u, worker_count);
    lock_pool(pool);
    while (pool->pending_worker_count != 0u) {
        wait_for_done(pool);
    }
    unlock_pool(pool);
#endif
}
