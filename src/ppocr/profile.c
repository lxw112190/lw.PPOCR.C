#include "profile_internal.h"

#include <string.h>

static void add_saturated(uint64_t* destination, uint64_t value) {
    if (*destination > UINT64_MAX - value) {
        *destination = UINT64_MAX;
    } else {
        *destination += value;
    }
}

void lw_profile_add_value(uint64_t* destination, uint64_t value) {
    if (destination != NULL) {
        add_saturated(destination, value);
    }
}

uint32_t lw_rec_width_histogram_bucket(uint32_t resized_width) {
    static const uint32_t upper_bounds[LW_REC_WIDTH_HISTOGRAM_BUCKET_COUNT - 1u] = {
        192u, 256u, 320u, 480u, 640u, 800u, 960u};
    uint32_t bucket = 0u;
    while (bucket + 1u < LW_REC_WIDTH_HISTOGRAM_BUCKET_COUNT &&
           resized_width > upper_bounds[bucket]) {
        ++bucket;
    }
    return bucket;
}

void lw_pipeline_component_profile_reset(lw_pipeline_component_profile* profile,
                                         lw_execution_profile_clock clock, void* clock_context) {
    if (profile == NULL) {
        return;
    }
    memset(profile, 0, sizeof(*profile));
    profile->execution.struct_size = (uint32_t)sizeof(profile->execution);
    profile->execution.clock = clock;
    profile->execution.clock_context = clock_context;
}

void lw_ocr_execution_profile_init(lw_ocr_execution_profile* profile,
                                   lw_execution_profile_clock clock, void* clock_context) {
    if (profile == NULL) {
        return;
    }
    memset(profile, 0, sizeof(*profile));
    profile->struct_size = (uint32_t)sizeof(*profile);
    profile->clock = clock;
    profile->clock_context = clock_context;
    lw_pipeline_component_profile_reset(&profile->detector, clock, clock_context);
    lw_pipeline_component_profile_reset(&profile->classifier, clock, clock_context);
    lw_pipeline_component_profile_reset(&profile->recognizer, clock, clock_context);
}

void lw_pipeline_component_profile_accumulate(lw_pipeline_component_profile* destination,
                                              const lw_pipeline_component_profile* source) {
    uint32_t index;
    if (destination == NULL || source == NULL) {
        return;
    }
    add_saturated(&destination->preprocess_nanoseconds, source->preprocess_nanoseconds);
    add_saturated(&destination->graph_nanoseconds, source->graph_nanoseconds);
    add_saturated(&destination->postprocess_nanoseconds, source->postprocess_nanoseconds);
    add_saturated(&destination->session_cache_hits, source->session_cache_hits);
    add_saturated(&destination->session_cache_misses, source->session_cache_misses);
    add_saturated(&destination->session_reconfigurations, source->session_reconfigurations);
    for (index = 0u; index < LW_REC_WIDTH_HISTOGRAM_BUCKET_COUNT; ++index) {
        uint32_t node;
        for (node = 0u; node < LW_EXECUTION_PROFILE_NODE_CAPACITY; ++node) {
            add_saturated(&destination->node_nanoseconds_by_width[index][node],
                          source->node_nanoseconds_by_width[index][node]);
            add_saturated(&destination->node_invocations_by_width[index][node],
                          source->node_invocations_by_width[index][node]);
        }
    }
    for (index = 0u; index < LW_EXECUTION_PROFILE_OPERATOR_CAPACITY; ++index) {
        add_saturated(&destination->execution.operator_nanoseconds[index],
                      source->execution.operator_nanoseconds[index]);
        add_saturated(&destination->execution.operator_invocations[index],
                      source->execution.operator_invocations[index]);
    }
    for (index = 0u; index < LW_EXECUTION_PROFILE_NODE_CAPACITY; ++index) {
        add_saturated(&destination->execution.node_nanoseconds[index],
                      source->execution.node_nanoseconds[index]);
        add_saturated(&destination->execution.node_invocations[index],
                      source->execution.node_invocations[index]);
    }
    for (index = 0u; index < LW_EXECUTION_PROFILE_CONV_CLASS_CAPACITY; ++index) {
        add_saturated(&destination->execution.conv_class_nanoseconds[index],
                      source->execution.conv_class_nanoseconds[index]);
        add_saturated(&destination->execution.conv_class_invocations[index],
                      source->execution.conv_class_invocations[index]);
    }
    for (index = 0u; index < LW_EXECUTION_PROFILE_THREAD_HISTOGRAM_CAPACITY; ++index) {
        add_saturated(&destination->execution.conv_thread_histogram[index],
                      source->execution.conv_thread_histogram[index]);
        add_saturated(&destination->execution.conv_transpose_thread_histogram[index],
                      source->execution.conv_transpose_thread_histogram[index]);
    }
    add_saturated(&destination->execution.packed_conv1x1_invocations,
                  source->execution.packed_conv1x1_invocations);
    add_saturated(&destination->execution.packed_conv3x3_stride2_invocations,
                  source->execution.packed_conv3x3_stride2_invocations);
    add_saturated(&destination->execution.unpacked_conv_invocations,
                  source->execution.unpacked_conv_invocations);
    add_saturated(&destination->execution.packed_matmul_invocations,
                  source->execution.packed_matmul_invocations);
    add_saturated(&destination->execution.unpacked_matmul_invocations,
                  source->execution.unpacked_matmul_invocations);
    add_saturated(&destination->execution.fused_gelu_invocations,
                  source->execution.fused_gelu_invocations);
    add_saturated(&destination->execution.ctc_greedy_invocations,
                  source->execution.ctc_greedy_invocations);
    add_saturated(&destination->execution.ctc_packed_projection_invocations,
                  source->execution.ctc_packed_projection_invocations);
    add_saturated(&destination->execution.ctc_generic_projection_invocations,
                  source->execution.ctc_generic_projection_invocations);
}

void lw_pipeline_profile_capture_node_width_delta(
    lw_pipeline_component_profile* profile, uint32_t width_bucket,
    const uint64_t before_nanoseconds[LW_EXECUTION_PROFILE_NODE_CAPACITY],
    const uint64_t before_invocations[LW_EXECUTION_PROFILE_NODE_CAPACITY]) {
    uint32_t node;
    if (profile == NULL || before_nanoseconds == NULL || before_invocations == NULL ||
        width_bucket >= LW_REC_WIDTH_HISTOGRAM_BUCKET_COUNT) {
        return;
    }
    for (node = 0u; node < LW_EXECUTION_PROFILE_NODE_CAPACITY; ++node) {
        uint64_t current_nanoseconds = profile->execution.node_nanoseconds[node];
        uint64_t current_invocations = profile->execution.node_invocations[node];
        uint64_t elapsed = current_nanoseconds >= before_nanoseconds[node]
                               ? current_nanoseconds - before_nanoseconds[node]
                               : UINT64_MAX - before_nanoseconds[node] + current_nanoseconds + 1u;
        uint64_t invocations = current_invocations >= before_invocations[node]
                                   ? current_invocations - before_invocations[node]
                                   : UINT64_MAX - before_invocations[node] +
                                         current_invocations + 1u;
        add_saturated(&profile->node_nanoseconds_by_width[width_bucket][node], elapsed);
        add_saturated(&profile->node_invocations_by_width[width_bucket][node], invocations);
    }
}

uint64_t lw_pipeline_profile_now(const lw_pipeline_component_profile* profile) {
    return profile == NULL || profile->execution.clock == NULL
               ? 0u
               : profile->execution.clock(profile->execution.clock_context);
}

void lw_pipeline_profile_add_elapsed(uint64_t* destination, uint64_t started,
                                     const lw_pipeline_component_profile* profile) {
    uint64_t finished = lw_pipeline_profile_now(profile);
    if (destination != NULL && finished >= started) {
        add_saturated(destination, finished - started);
    }
}

uint64_t lw_ocr_profile_now(const lw_ocr_execution_profile* profile) {
    return profile == NULL || profile->clock == NULL ? 0u : profile->clock(profile->clock_context);
}

void lw_ocr_profile_add_elapsed(uint64_t* destination, uint64_t started,
                                const lw_ocr_execution_profile* profile) {
    uint64_t finished = lw_ocr_profile_now(profile);
    if (destination != NULL && finished >= started) {
        add_saturated(destination, finished - started);
    }
}
