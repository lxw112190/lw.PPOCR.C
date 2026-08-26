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
