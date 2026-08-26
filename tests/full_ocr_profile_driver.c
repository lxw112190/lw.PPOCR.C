#if !defined(_WIN32) && !defined(__APPLE__)
#  define _POSIX_C_SOURCE 200809L
#endif

#include "lw_infer.h"
#include "det_internal.h"
#include "lwm_read.h"
#include "model_internal.h"
#include "ppm_image.h"
#include "profile_internal.h"
#include "session_internal.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#elif defined(__APPLE__)
#  include <mach/mach_time.h>
#else
#  include <time.h>
#endif

static uint64_t profile_clock(void* context) {
#if defined(_WIN32)
    LARGE_INTEGER counter;
    const LARGE_INTEGER* frequency = (const LARGE_INTEGER*)context;
    if (frequency == NULL || frequency->QuadPart <= 0 || !QueryPerformanceCounter(&counter)) {
        return 0u;
    }
    return (uint64_t)((double)counter.QuadPart * 1000000000.0 / (double)frequency->QuadPart);
#elif defined(__APPLE__)
    const mach_timebase_info_data_t* timebase = (const mach_timebase_info_data_t*)context;
    uint64_t ticks = mach_absolute_time();
    if (timebase == NULL || timebase->denom == 0u) {
        return 0u;
    }
    return (uint64_t)((double)ticks * (double)timebase->numer / (double)timebase->denom);
#else
    struct timespec value;
    (void)context;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0u;
    }
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
#endif
}

static int parse_positive_u32(const char* text, uint32_t* value) {
    char* end = NULL;
    unsigned long parsed;
    errno = 0;
    parsed = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0u || parsed > UINT32_MAX) {
        return 0;
    }
    *value = (uint32_t)parsed;
    return 1;
}

static int allocation_fits(uint64_t count, size_t element_size) {
    return element_size != 0u && count <= (uint64_t)SIZE_MAX / (uint64_t)element_size;
}

static uint64_t add_saturated(uint64_t left, uint64_t right) {
    return left > UINT64_MAX - right ? UINT64_MAX : left + right;
}

static uint64_t operator_nanoseconds(const lw_ocr_execution_profile* profile, uint32_t operation) {
    uint64_t total = profile->detector.execution.operator_nanoseconds[operation];
    total = add_saturated(total, profile->classifier.execution.operator_nanoseconds[operation]);
    return add_saturated(total, profile->recognizer.execution.operator_nanoseconds[operation]);
}

static uint64_t operator_invocations(const lw_ocr_execution_profile* profile, uint32_t operation) {
    uint64_t total = profile->detector.execution.operator_invocations[operation];
    total = add_saturated(total, profile->classifier.execution.operator_invocations[operation]);
    return add_saturated(total, profile->recognizer.execution.operator_invocations[operation]);
}

static void print_tensor_dimensions(const lw_runtime_tensor* tensor) {
    uint32_t dimension;
    putchar('[');
    for (dimension = 0u; dimension < tensor->rank; ++dimension) {
        if (dimension != 0u) {
            putchar(',');
        }
        printf("%d", tensor->dimensions[dimension]);
    }
    putchar(']');
}

static int run_ocr(lw_ocr* ocr, const lw_example_ppm_image* image, lw_ocr_line* lines,
                   uint32_t line_capacity, char* text, uint64_t text_capacity,
                   lw_ocr_execution_profile* profile, lw_ocr_result* result) {
    lw_error error;
    lw_status status;
    lw_error_init(&error);
    lw_ocr_result_init(result);
    status =
        profile == NULL
            ? lw_ocr_run_bgr_u8(ocr, image->pixels, image->byte_count, image->width, image->height,
                                image->width * 3u, lines, line_capacity, text, text_capacity,
                                result, &error)
            : lw_ocr_run_bgr_u8_profiled(ocr, image->pixels, image->byte_count, image->width,
                                         image->height, image->width * 3u, lines, line_capacity,
                                         text, text_capacity, result, profile, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "OCR failed: %s: %s\n", lw_status_string(status), error.message);
        return 0;
    }
    return 1;
}

int main(int argc, char** argv) {
    static const char* operator_names[LW_EXECUTION_PROFILE_OPERATOR_CAPACITY] = {
        "unknown",    "Conv",      "Add",           "Mul",
        "Div",        "Erf",       "HardSigmoid",   "BatchNormalization",
        "ReduceMean", "Relu",      "AveragePool",   "Squeeze",
        "Transpose",  "Unsqueeze", "MatMul",        "Softmax",
        "Reshape",    "Concat",    "ConvTranspose", "MaxPool",
        "Resize",     "Sigmoid"};
    static const char* conv_class_names[LW_EXECUTION_PROFILE_CONV_CLASS_CAPACITY] = {
        "Conv1x1", "Conv3x3", "Depthwise3x3", "Stride2Conv3x3", "OtherConv"};
    lw_example_ppm_image image;
    lw_ocr_options options;
    lw_ocr_info info;
    lw_ocr* ocr = NULL;
    lw_model* det_model = NULL;
    lw_session* det_session = NULL;
    lw_tensor_desc det_input_desc;
    lw_ocr_line* lines = NULL;
    lw_ocr_result result;
    lw_ocr_execution_profile profile;
    char* text = NULL;
    char* reference_text = NULL;
    uint32_t iterations;
    uint32_t workers;
    uint32_t iteration;
    uint32_t reference_line_count;
    uint32_t det_width;
    uint32_t det_height;
    float det_width_ratio;
    float det_height_ratio;
    uint64_t reference_text_capacity;
    uint64_t graph_work_nanoseconds = 0u;
    uint64_t conv_nanoseconds = 0u;
    uint64_t conv_invocations = 0u;
    uint32_t index;
    lw_error error;
    lw_status status;
    int exit_code = 1;
#if defined(_WIN32)
    LARGE_INTEGER clock_context;
#elif defined(__APPLE__)
    mach_timebase_info_data_t clock_context;
#endif

    memset(&image, 0, sizeof(image));
    if (argc != 8 || !parse_positive_u32(argv[6], &iterations) || iterations > 100u ||
        !parse_positive_u32(argv[7], &workers) || workers > 16u) {
        fprintf(stderr, "usage: full-ocr-profile-driver <det.lwm> <cls.lwm> <rec.lwm> "
                        "<dictionary.txt> <image.ppm> <iterations> <workers>\n");
        return 2;
    }
    if (!lw_example_ppm_image_load_bgr(argv[5], &image) || image.width > UINT32_MAX / 3u) {
        fprintf(stderr, "invalid P6 PPM image: %s\n", argv[5]);
        goto cleanup;
    }
    lw_ocr_options_init(&options);
    options.worker_count = workers;
    status = lw_det_compute_size(image.width, image.height, options.detector.limit_side_length,
                                 &det_width, &det_height, &det_width_ratio, &det_height_ratio);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "unable to resolve DET input size\n");
        goto cleanup;
    }
    (void)det_width_ratio;
    (void)det_height_ratio;
    lw_error_init(&error);
    status = lw_model_load(argv[1], NULL, &det_model, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "DET model load failed: %s: %s\n", lw_status_string(status), error.message);
        goto cleanup;
    }
    lw_tensor_desc_init(&det_input_desc);
    det_input_desc.dtype = LW_DTYPE_F32;
    det_input_desc.rank = 4u;
    det_input_desc.dimensions[0] = 1;
    det_input_desc.dimensions[1] = 3;
    det_input_desc.dimensions[2] = (int32_t)det_height;
    det_input_desc.dimensions[3] = (int32_t)det_width;
    lw_error_init(&error);
    status = lw_session_create(det_model, &det_input_desc, 1u, NULL, &det_session, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "DET analysis session create failed: %s: %s\n", lw_status_string(status),
                error.message);
        goto cleanup;
    }
    lw_error_init(&error);
    status = lw_ocr_create(argv[1], argv[2], argv[3], argv[4], &options, &ocr, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "OCR create failed: %s: %s\n", lw_status_string(status), error.message);
        goto cleanup;
    }
    lw_ocr_info_init(&info);
    if (lw_ocr_get_info(ocr, &info) != LW_STATUS_OK || info.max_line_capacity == 0u ||
        !allocation_fits(info.max_line_capacity, sizeof(*lines)) || info.max_text_capacity == 0u ||
        info.max_text_capacity > SIZE_MAX) {
        fprintf(stderr, "unable to query OCR output capacities\n");
        goto cleanup;
    }
    lines = (lw_ocr_line*)calloc(info.max_line_capacity, sizeof(*lines));
    text = (char*)malloc((size_t)info.max_text_capacity);
    reference_text = (char*)malloc((size_t)info.max_text_capacity);
    if (lines == NULL || text == NULL || reference_text == NULL) {
        fprintf(stderr, "unable to allocate profiler output buffers\n");
        goto cleanup;
    }
    if (!run_ocr(ocr, &image, lines, info.max_line_capacity, text, info.max_text_capacity, NULL,
                 &result)) {
        goto cleanup;
    }
    reference_line_count = result.line_count;
    reference_text_capacity = result.required_text_capacity;
    memcpy(reference_text, text, (size_t)reference_text_capacity);

#if defined(_WIN32)
    if (!QueryPerformanceFrequency(&clock_context) || clock_context.QuadPart <= 0) {
        fprintf(stderr, "unable to initialize the profile clock\n");
        goto cleanup;
    }
    lw_ocr_execution_profile_init(&profile, profile_clock, &clock_context);
#elif defined(__APPLE__)
    if (mach_timebase_info(&clock_context) != KERN_SUCCESS || clock_context.denom == 0u) {
        fprintf(stderr, "unable to initialize the profile clock\n");
        goto cleanup;
    }
    lw_ocr_execution_profile_init(&profile, profile_clock, &clock_context);
#else
    lw_ocr_execution_profile_init(&profile, profile_clock, NULL);
#endif
    for (iteration = 0u; iteration < iterations; ++iteration) {
        if (!run_ocr(ocr, &image, lines, info.max_line_capacity, text, info.max_text_capacity,
                     &profile, &result)) {
            goto cleanup;
        }
        if (result.line_count != reference_line_count ||
            result.required_text_capacity != reference_text_capacity ||
            memcmp(text, reference_text, (size_t)reference_text_capacity) != 0) {
            fprintf(stderr, "profiled OCR output is not deterministic\n");
            goto cleanup;
        }
    }

    for (index = 1u; index < LW_EXECUTION_PROFILE_OPERATOR_CAPACITY; ++index) {
        graph_work_nanoseconds =
            add_saturated(graph_work_nanoseconds, operator_nanoseconds(&profile, index));
    }
    for (index = 0u; index < LW_EXECUTION_PROFILE_CONV_CLASS_CAPACITY; ++index) {
        uint64_t class_ns = profile.detector.execution.conv_class_nanoseconds[index];
        uint64_t class_calls = profile.detector.execution.conv_class_invocations[index];
        class_ns =
            add_saturated(class_ns, profile.classifier.execution.conv_class_nanoseconds[index]);
        class_ns =
            add_saturated(class_ns, profile.recognizer.execution.conv_class_nanoseconds[index]);
        class_calls =
            add_saturated(class_calls, profile.classifier.execution.conv_class_invocations[index]);
        class_calls =
            add_saturated(class_calls, profile.recognizer.execution.conv_class_invocations[index]);
        conv_nanoseconds = add_saturated(conv_nanoseconds, class_ns);
        conv_invocations = add_saturated(conv_invocations, class_calls);
    }

    printf("{\"schema_version\":1,\"image_width\":%u,\"image_height\":%u,", image.width,
           image.height);
    printf("\"iterations\":%u,\"workers\":%u,\"lines\":%u,", iterations, workers,
           reference_line_count);
    printf("\"wall_nanoseconds\":{");
    printf("\"total\":%llu,\"det_preprocess\":%llu,\"det_graph\":%llu,",
           (unsigned long long)profile.total_nanoseconds,
           (unsigned long long)profile.detector.preprocess_nanoseconds,
           (unsigned long long)profile.detector.graph_nanoseconds);
    printf("\"det_postprocess\":%llu,\"crop\":%llu,\"line_workers\":%llu,"
           "\"line_worker_critical\":%llu,\"line_dispatch_overhead\":%llu,"
           "\"output\":%llu},",
           (unsigned long long)profile.detector.postprocess_nanoseconds,
           (unsigned long long)profile.crop_nanoseconds,
           (unsigned long long)profile.line_workers_nanoseconds,
           (unsigned long long)profile.line_worker_critical_nanoseconds,
           (unsigned long long)profile.line_dispatch_overhead_nanoseconds,
           (unsigned long long)profile.output_nanoseconds);
    printf("\"line_work_nanoseconds\":{");
    printf("\"cls_preprocess\":%llu,\"cls_graph\":%llu,\"cls_postprocess\":%llu,",
           (unsigned long long)profile.classifier.preprocess_nanoseconds,
           (unsigned long long)profile.classifier.graph_nanoseconds,
           (unsigned long long)profile.classifier.postprocess_nanoseconds);
    printf("\"rec_preprocess\":%llu,\"rec_graph\":%llu,"
           "\"rec_postprocess\":%llu},",
           (unsigned long long)profile.recognizer.preprocess_nanoseconds,
           (unsigned long long)profile.recognizer.graph_nanoseconds,
           (unsigned long long)profile.recognizer.postprocess_nanoseconds);
    printf("\"graph_work_nanoseconds\":%llu,\"operators\":[",
           (unsigned long long)graph_work_nanoseconds);
    for (index = 1u; index < LW_EXECUTION_PROFILE_OPERATOR_CAPACITY; ++index) {
        uint64_t nanoseconds = operator_nanoseconds(&profile, index);
        uint64_t invocations = operator_invocations(&profile, index);
        double percentage = graph_work_nanoseconds == 0u
                                ? 0.0
                                : (double)nanoseconds * 100.0 / (double)graph_work_nanoseconds;
        if (index != 1u) {
            putchar(',');
        }
        printf("{\"id\":%u,\"name\":\"%s\",\"invocations\":%llu,"
               "\"nanoseconds\":%llu,\"percentage\":%.6f,"
               "\"det_invocations\":%llu,\"cls_invocations\":%llu,"
               "\"rec_invocations\":%llu,\"det_nanoseconds\":%llu,"
               "\"cls_nanoseconds\":%llu,\"rec_nanoseconds\":%llu}",
               index, operator_names[index], (unsigned long long)invocations,
               (unsigned long long)nanoseconds, percentage,
               (unsigned long long)profile.detector.execution.operator_invocations[index],
               (unsigned long long)profile.classifier.execution.operator_invocations[index],
               (unsigned long long)profile.recognizer.execution.operator_invocations[index],
               (unsigned long long)profile.detector.execution.operator_nanoseconds[index],
               (unsigned long long)profile.classifier.execution.operator_nanoseconds[index],
               (unsigned long long)profile.recognizer.execution.operator_nanoseconds[index]);
    }
    printf("],\"conv_nanoseconds\":%llu,\"conv_invocations\":%llu,\"conv_classes\":[",
           (unsigned long long)conv_nanoseconds, (unsigned long long)conv_invocations);
    for (index = 0u; index < LW_EXECUTION_PROFILE_CONV_CLASS_CAPACITY; ++index) {
        uint64_t nanoseconds = profile.detector.execution.conv_class_nanoseconds[index];
        uint64_t invocations = profile.detector.execution.conv_class_invocations[index];
        nanoseconds =
            add_saturated(nanoseconds, profile.classifier.execution.conv_class_nanoseconds[index]);
        nanoseconds =
            add_saturated(nanoseconds, profile.recognizer.execution.conv_class_nanoseconds[index]);
        invocations =
            add_saturated(invocations, profile.classifier.execution.conv_class_invocations[index]);
        invocations =
            add_saturated(invocations, profile.recognizer.execution.conv_class_invocations[index]);
        if (index != 0u) {
            putchar(',');
        }
        printf("{\"id\":%u,\"name\":\"%s\",\"invocations\":%llu,"
               "\"nanoseconds\":%llu,\"det_nanoseconds\":%llu,"
               "\"cls_nanoseconds\":%llu,\"rec_nanoseconds\":%llu}",
               index, conv_class_names[index], (unsigned long long)invocations,
               (unsigned long long)nanoseconds,
               (unsigned long long)profile.detector.execution.conv_class_nanoseconds[index],
               (unsigned long long)profile.classifier.execution.conv_class_nanoseconds[index],
               (unsigned long long)profile.recognizer.execution.conv_class_nanoseconds[index]);
    }
    printf("],\"det_convolution_nodes\":[");
    {
        uint32_t node_index;
        int first = 1;
        for (node_index = 0u; node_index < det_model->info.node_count; ++node_index) {
            const uint8_t* node = det_model->bytes + (size_t)det_model->node_offset +
                                  (size_t)node_index * LWM_V0_NODE_SIZE;
            uint16_t operation = lwm_read_u16(node);
            uint32_t input_index;
            uint32_t weight_index;
            uint32_t output_index;
            uint64_t param_offset;
            const uint8_t* params;
            const lw_runtime_tensor* input_tensor;
            const lw_runtime_tensor* weight_tensor;
            const lw_runtime_tensor* output_tensor;
            uint64_t node_nanoseconds;
            uint64_t node_invocations;
            if (operation != 1u && operation != 18u) {
                continue;
            }
            input_index = lwm_read_u32(node + 8u);
            weight_index = lwm_read_u32(node + 12u);
            output_index = lwm_read_u32(node + 40u);
            param_offset = lwm_read_u64(node + 56u);
            params = det_model->bytes + (size_t)param_offset;
            input_tensor = &det_session->tensors[input_index];
            weight_tensor = &det_session->tensors[weight_index];
            output_tensor = &det_session->tensors[output_index];
            node_nanoseconds = node_index < LW_EXECUTION_PROFILE_NODE_CAPACITY
                                   ? profile.detector.execution.node_nanoseconds[node_index]
                                   : 0u;
            node_invocations = node_index < LW_EXECUTION_PROFILE_NODE_CAPACITY
                                   ? profile.detector.execution.node_invocations[node_index]
                                   : 0u;
            if (!first) {
                putchar(',');
            }
            first = 0;
            printf("{\"node\":%u,\"operation\":\"%s\","
                   "\"nanoseconds\":%llu,\"invocations\":%llu,\"input\":",
                   node_index, operator_names[operation], (unsigned long long)node_nanoseconds,
                   (unsigned long long)node_invocations);
            print_tensor_dimensions(input_tensor);
            printf(",\"weights\":");
            print_tensor_dimensions(weight_tensor);
            printf(",\"output\":");
            print_tensor_dimensions(output_tensor);
            printf(",\"group\":%u,\"kernel\":[%d,%d],\"strides\":[%d,%d],"
                   "\"dilations\":[%d,%d],\"pads\":[%d,%d,%d,%d]}",
                   lwm_read_u32(params + 4u), lwm_read_i32(params + 8u), lwm_read_i32(params + 12u),
                   lwm_read_i32(params + 16u), lwm_read_i32(params + 20u),
                   lwm_read_i32(params + 24u), lwm_read_i32(params + 28u),
                   lwm_read_i32(params + 32u), lwm_read_i32(params + 36u),
                   lwm_read_i32(params + 40u), lwm_read_i32(params + 44u));
        }
    }
    printf("]}\n");
    exit_code = 0;

cleanup:
    free(reference_text);
    free(text);
    free(lines);
    lw_ocr_free(ocr);
    lw_session_free(det_session);
    lw_model_free(det_model);
    lw_example_ppm_image_free(&image);
    return exit_code;
}
