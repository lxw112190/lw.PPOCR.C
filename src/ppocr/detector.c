#include "lw_infer.h"
#include "abi_compat_internal.h"

/* Public DET handle: preprocessing -> graph execution -> DB postprocessing. */

#include "det_internal.h"
#include "error_internal.h"
#include "executor_internal.h"
#include "lwm_read.h"
#include "model_internal.h"
#include "operator_internal.h"
#include "profile_internal.h"
#include "session_internal.h"

#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
#include "x64_det_backend_internal.h"
#endif

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define LW_DET_DEFAULT_LIMIT_SIDE_LENGTH 960u
#define LW_DET_MAX_LIMIT_SIDE_LENGTH 4096u
#define LW_DET_DEFAULT_MAX_CANDIDATES 1000u
#define LW_DET_MAX_CANDIDATES 10000u
#define LW_DET_DEFAULT_MAX_IMAGE_PIXELS UINT64_C(40000000)

struct lw_detector {
    lw_model* model;
    lw_session* session;
    float* input;
    float* probabilities;
    uint64_t input_element_count;
    uint64_t probability_element_count;
    uint32_t resized_width;
    uint32_t resized_height;
    uint32_t intra_op_thread_count;
    uint32_t reading_order;
    lw_session_options session_options;
    lw_detector_info info;
    lw_db_postprocess_workspace postprocess_workspace;
    lw_det_preprocess_workspace preprocess_workspace;
    uint8_t* padded_source;
    uint32_t padded_width;
    uint32_t padded_height;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    lw_x64_det_program* x64_program;
    lw_x64_det_instance* x64_instance;
    uint8_t x64_backend_enabled;
#endif
};

static void clear_result(lw_detection_result* result) {
    memset(result, 0, sizeof(*result));
    result->struct_size = (uint32_t)sizeof(*result);
}

#if defined(__EMSCRIPTEN__) && defined(LW_WASM_COMPILED_DET)
/* CI-only diagnostics for the compiled DET path. Never scan tensors during
 * ordinary benchmark measurements or in the browser. */
static void detector_debug_tensor(const char* stage, const float* values, uint64_t count) {
    float minimum = INFINITY;
    float maximum = -INFINITY;
    uint64_t finite_count = 0u;
    uint64_t above_threshold = 0u;
    uint64_t index;
    for (index = 0u; index < count; ++index) {
        float value = values[index];
        if (!isfinite(value)) continue;
        ++finite_count;
        if (value < minimum) minimum = value;
        if (value > maximum) maximum = value;
        if (value > 0.3f) ++above_threshold;
    }
    if (finite_count == 0u) {
        minimum = 0.0f;
        maximum = 0.0f;
    }
    (void)fprintf(stderr,
                  "LW_WASM_DET_DEBUG stage=%s count=%llu finite=%llu above_0.3=%llu "
                  "min=%.6g max=%.6g first=%.6g\n",
                  stage, (unsigned long long)count, (unsigned long long)finite_count,
                  (unsigned long long)above_threshold, (double)minimum, (double)maximum,
                  count == 0u ? 0.0 : (double)values[0]);
}
#endif

void lw_detector_options_init(lw_detector_options* options) {
    lw_model_options model_options;
    lw_session_options session_options;
    if (options == NULL)
        return;
    lw_model_options_init(&model_options);
    lw_session_options_init(&session_options);
    memset(options, 0, sizeof(*options));
    options->struct_size = (uint32_t)sizeof(*options);
    options->limit_side_length = LW_DET_DEFAULT_LIMIT_SIDE_LENGTH;
    options->max_candidates = LW_DET_DEFAULT_MAX_CANDIDATES;
    /* Reference (SimdPaddleOCR) defaults, paired with the Clipper-style
     * round-join unclip and the scanline PolygonScore in the postprocess. */
    options->bitmap_threshold = 0.2f;
    options->box_threshold = 0.45f;
    options->unclip_ratio = 1.4f;
    options->max_model_file_size = model_options.max_file_size;
    options->max_workspace_size = session_options.max_workspace_size;
    options->max_tensor_size = session_options.max_tensor_size;
    options->max_image_pixels = LW_DET_DEFAULT_MAX_IMAGE_PIXELS;
}

void lw_detector_info_init(lw_detector_info* info) {
    if (info == NULL)
        return;
    memset(info, 0, sizeof(*info));
    info->struct_size = (uint32_t)sizeof(*info);
}

void lw_detection_result_init(lw_detection_result* result) {
    if (result == NULL)
        return;
    clear_result(result);
}

static lw_status validate_options(const lw_detector_options* options, lw_detector_info* info,
                                  lw_model_options* model_options,
                                  lw_session_options* session_options, lw_error* error) {
    lw_detector_options values;
    lw_detector_options_init(&values);
    if (options != NULL) {
        if (!lw_abi_copy_input_prefix(&values, sizeof(values), options)) {
            lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "invalid detector options structure");
            return LW_STATUS_INVALID_ARGUMENT;
        }
        values.struct_size = (uint32_t)sizeof(values);
        if (values.reserved != 0u) {
            lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "invalid detector options structure");
            return LW_STATUS_INVALID_ARGUMENT;
        }
        if (values.limit_side_length == 0u)
            values.limit_side_length = LW_DET_DEFAULT_LIMIT_SIDE_LENGTH;
        if (values.max_candidates == 0u)
            values.max_candidates = LW_DET_DEFAULT_MAX_CANDIDATES;
        if (values.max_image_pixels == 0u)
            values.max_image_pixels = LW_DET_DEFAULT_MAX_IMAGE_PIXELS;
    }
    if (values.limit_side_length < 32u || values.limit_side_length > LW_DET_MAX_LIMIT_SIDE_LENGTH ||
        values.max_candidates == 0u || values.max_candidates > LW_DET_MAX_CANDIDATES ||
        values.use_dilation > 1u || !isfinite(values.bitmap_threshold) ||
        values.bitmap_threshold < 0.0f || values.bitmap_threshold > 1.0f ||
        !isfinite(values.box_threshold) || values.box_threshold < 0.0f ||
        values.box_threshold > 1.0f || !isfinite(values.unclip_ratio) ||
        values.unclip_ratio <= 0.0f || values.unclip_ratio > 10.0f) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "detector thresholds or limits are invalid");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    lw_model_options_init(model_options);
    lw_session_options_init(session_options);
    if (values.max_model_file_size != 0u)
        model_options->max_file_size = values.max_model_file_size;
    if (values.max_workspace_size != 0u)
        session_options->max_workspace_size = values.max_workspace_size;
    if (values.max_tensor_size != 0u)
        session_options->max_tensor_size = values.max_tensor_size;
    lw_detector_info_init(info);
    info->limit_side_length = values.limit_side_length;
    info->max_candidates = values.max_candidates;
    info->use_dilation = values.use_dilation;
    info->bitmap_threshold = values.bitmap_threshold;
    info->box_threshold = values.box_threshold;
    info->unclip_ratio = values.unclip_ratio;
    info->max_image_pixels = values.max_image_pixels;
    return LW_STATUS_OK;
}

/* Tiny-family sniff: the reference treats a detector whose first Conv emits
 * at most 16 channels as tiny and lowers its box threshold to 0.4. */
static int model_is_tiny(const lw_model* model) {
    uint32_t node_count;
    uint32_t i;
    if (model == NULL || model->node_offset == 0u || model->tensor_offset == 0u) return 0;
    node_count = model->info.node_count;
    for (i = 0u; i < node_count; ++i) {
        const uint8_t* node = model->bytes + (size_t)model->node_offset +
                              (size_t)i * LWM_V0_NODE_SIZE;
        uint32_t output_index;
        const uint8_t* tensor;
        if (lwm_read_u16(node) != LW_OP_CONV) continue;
        output_index = lwm_read_u32(node + 40u);
        if (output_index >= model->info.tensor_count) return 0;
        tensor = model->bytes + (size_t)model->tensor_offset +
                 (size_t)output_index * LWM_V0_TENSOR_SIZE;
        return lwm_read_i32(tensor + 8u + 4u) <= 16;
    }
    return 0;
}

#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
static void detector_compile_backend(const lw_detector* detector, uint32_t width,
                                     uint32_t height, lw_x64_det_program** out_program,
                                     lw_x64_det_instance** out_instance) {
#if defined(LW_EXPERIMENTAL_DET_FIXED_POINT_RESIZE)
    const lw_x64_det_compile_strategy strategy = LW_X64_DET_COMPILE_NHWC_DIRECT;
#else
    const lw_x64_det_compile_strategy strategy = LW_X64_DET_COMPILE_NHWC;
#endif
    lw_error backend_error;
    lw_x64_det_program* program = NULL;
    lw_x64_det_instance* instance = NULL;
    lw_error_init(&backend_error);
    if (lw_x64_det_backend_compile_ex(detector->model, height, width,
                                             strategy, &program,
                                             &backend_error) != LW_X64_DET_COMPILE_OK ||
               lw_x64_det_instance_create(program, &instance, &backend_error) != LW_STATUS_OK) {
        lw_x64_det_program_free(program);
        program = NULL;
        instance = NULL;
#if !defined(__EMSCRIPTEN__)
        if (lw_x64_det_backend_compile_ex(detector->model, height, width,
                                          LW_X64_DET_COMPILE_NCHW, &program,
                                          &backend_error) == LW_X64_DET_COMPILE_OK &&
            lw_x64_det_instance_create(program, &instance, &backend_error) != LW_STATUS_OK) {
            lw_x64_det_program_free(program);
            program = NULL;
            instance = NULL;
        }
#endif
    }
    *out_program = program;
    *out_instance = instance;
#if defined(LW_WASM_COMPILED_DET)
    if (program != NULL && instance != NULL) {
        uint64_t packed_bytes = 0u;
        uint32_t constant_index;
        for (constant_index = 0u; constant_index < program->packed_constant_count;
             ++constant_index) {
            packed_bytes += program->constants[constant_index].bytes;
        }
        (void)fprintf(stderr,
            "LW_WASM_COMPILED_DET input=%ux%u layout=nhwc ops=%u unsupported=%u "
            "conversions=%u direct_input=%u arena_bytes=%llu packed_bytes=%llu\n",
            width, height, program->op_count, program->unsupported_nodes,
            program->layout_conversions, program->direct_nhwc,
            (unsigned long long)program->arena_bytes,
            (unsigned long long)packed_bytes);
    }
#endif
}

static void detector_set_backend_enabled(lw_detector* detector, uint8_t enabled) {
    if (detector == NULL || detector->x64_backend_enabled == enabled) return;
    detector->x64_backend_enabled = enabled;
    lw_session_free(detector->session);
    detector->session = NULL;
}

void lw_detector_test_enable_x64_backend(lw_detector* detector) {
    detector_set_backend_enabled(detector, 1u);
}

void lw_detector_test_disable_x64_backend(lw_detector* detector) {
    detector_set_backend_enabled(detector, 0u);
}
#endif

static lw_status ensure_session(lw_detector* detector, uint32_t width, uint32_t height,
                                lw_error* error) {
    lw_tensor_desc input_desc;
    lw_tensor_desc output_desc;
    lw_session* new_session = NULL;
    float* new_input = NULL;
    float* new_probabilities = NULL;
    uint64_t plane;
    uint64_t input_count;
    lw_status status;
    int needs_input = 1;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    lw_x64_det_program* new_program = NULL;
    lw_x64_det_instance* new_instance = NULL;
    int try_compiled = detector->x64_backend_enabled != 0u;
#endif
    if (detector->session != NULL && detector->resized_width == width &&
        detector->resized_height == height) {
        return LW_STATUS_OK;
    }
    plane = (uint64_t)width * height;
    if (plane > UINT64_MAX / 3u || plane > SIZE_MAX / sizeof(float)) {
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "detector tensor size overflows");
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    input_count = plane * 3u;
    if (input_count > SIZE_MAX / sizeof(float)) {
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "detector input tensor size overflows");
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    lw_tensor_desc_init(&input_desc);
    input_desc.dtype = LW_DTYPE_F32;
    input_desc.rank = 4u;
    input_desc.dimensions[0] = 1;
    input_desc.dimensions[1] = 3;
    input_desc.dimensions[2] = (int32_t)height;
    input_desc.dimensions[3] = (int32_t)width;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
#if !defined(__EMSCRIPTEN__)
    if (try_compiled) {
        const lw_cpu_capabilities cpu = lw_get_cpu_capabilities();
        try_compiled = lw_simd_level_is_avx2(cpu.simd) && cpu.has_avx2_fma;
    }
#endif
    /* Validate the shape and workspace budget before committing an arena. */
    status = try_compiled
        ? lw_session_create_metadata_only(detector->model, &input_desc, 1u,
                                          &detector->session_options, &new_session, error)
        : lw_session_create(detector->model, &input_desc, 1u,
                            &detector->session_options, &new_session, error);
#else
    status = lw_session_create(detector->model, &input_desc, 1u, &detector->session_options,
                               &new_session, error);
#endif
    if (status != LW_STATUS_OK)
        goto fail;
    lw_tensor_desc_init(&output_desc);
    status = lw_session_get_output_desc(new_session, 0u, &output_desc);
    if (status != LW_STATUS_OK || output_desc.dtype != LW_DTYPE_F32 || output_desc.rank != 4u ||
        output_desc.dimensions[0] != 1 || output_desc.dimensions[1] != 1 ||
        output_desc.dimensions[2] != (int32_t)height ||
        output_desc.dimensions[3] != (int32_t)width) {
        lw_set_error(error, LW_STATUS_INVALID_SHAPE,
                     "detector model output must have shape [1,1,height,width]");
        status = LW_STATUS_INVALID_SHAPE;
        goto fail;
    }
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    if (try_compiled) {
        detector_compile_backend(detector, width, height, &new_program, &new_instance);
        needs_input = new_instance == NULL;
        if (needs_input) {
            lw_session_free(new_session);
            new_session = NULL;
            status = lw_session_create(detector->model, &input_desc, 1u,
                                       &detector->session_options, &new_session, error);
            if (status != LW_STATUS_OK) goto fail;
        }
    }
#endif
    if (needs_input)
        new_input = (float*)malloc((size_t)input_count * sizeof(*new_input));
    new_probabilities = (float*)malloc((size_t)plane * sizeof(*new_probabilities));
    if ((needs_input && new_input == NULL) || new_probabilities == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate detector tensor buffers");
        status = LW_STATUS_OUT_OF_MEMORY;
        goto fail;
    }
    free(detector->probabilities);
    free(detector->input);
    lw_session_free(detector->session);
    detector->session = new_session;
    lw_session_set_intra_op_thread_count(new_session, detector->intra_op_thread_count);
    detector->input = new_input;
    detector->probabilities = new_probabilities;
    detector->input_element_count = input_count;
    detector->probability_element_count = plane;
    detector->resized_width = width;
    detector->resized_height = height;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    lw_x64_det_instance_free(detector->x64_instance);
    lw_x64_det_program_free(detector->x64_program);
    detector->x64_instance = new_instance;
    detector->x64_program = new_program;
#endif
    return LW_STATUS_OK;

fail:
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    lw_x64_det_instance_free(new_instance);
    lw_x64_det_program_free(new_program);
#endif
    free(new_probabilities);
    free(new_input);
    lw_session_free(new_session);
    return status;
}

lw_status lw_detector_create(const char* model_path_utf8, const lw_detector_options* options,
                             lw_detector** out_detector, lw_error* error) {
    lw_model_options model_options;
    lw_detector* detector = NULL;
    lw_status status;
    if (out_detector != NULL)
        *out_detector = NULL;
    if (model_path_utf8 == NULL || model_path_utf8[0] == '\0' || out_detector == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "model path and output detector are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    detector = (lw_detector*)calloc(1u, sizeof(*detector));
    if (detector == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate detector handle");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    detector->intra_op_thread_count = 1u;
    detector->reading_order = LW_READING_ORDER_HORIZONTAL_LTR;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    /* Promoted default: the sharded NHWC backend is on; failures inside
     * compiled setup fall back to the canonical executor. */
#if defined(__EMSCRIPTEN__) && !defined(LW_WASM_COMPILED_DET)
    detector->x64_backend_enabled = 0u;
#else
    detector->x64_backend_enabled = 1u;
#endif
#endif
    status = validate_options(options, &detector->info, &model_options, &detector->session_options,
                              error);
    if (status != LW_STATUS_OK)
        goto fail;
    status = lw_model_load(model_path_utf8, &model_options, &detector->model, error);
    if (status != LW_STATUS_OK)
        goto fail;
    if (detector->info.box_threshold == 0.0f ||
        (detector->info.box_threshold == 0.45f && model_is_tiny(detector->model))) {
        /* Auto box threshold (reference behavior): tiny detectors use 0.4,
         * everything else 0.45. A caller explicitly setting 0.45 on a tiny
         * model is indistinguishable from the default and also gets 0.4. */
        detector->info.box_threshold = model_is_tiny(detector->model) ? 0.4f : 0.45f;
    }
    status = ensure_session(detector, 32u, 32u, error);
    if (status != LW_STATUS_OK)
        goto fail;
    *out_detector = detector;
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;

fail:
    lw_detector_free(detector);
    return status;
}

void lw_detector_set_intra_op_thread_count(lw_detector* detector, uint32_t thread_count) {
    if (detector == NULL) {
        return;
    }
    detector->intra_op_thread_count = thread_count == 0u ? 1u : thread_count;
    lw_session_set_intra_op_thread_count(detector->session, detector->intra_op_thread_count);
}

uint32_t lw_detector_get_intra_op_thread_count(const lw_detector* detector) {
    return detector == NULL || detector->session == NULL ? 1u
                                                         : detector->session->intra_op_thread_count;
}

void lw_detector_free(lw_detector* detector) {
    if (detector == NULL)
        return;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    lw_x64_det_program_free(detector->x64_program);
    lw_x64_det_instance_free(detector->x64_instance);
#endif
    free(detector->probabilities);
    free(detector->input);
    lw_session_free(detector->session);
    lw_model_free(detector->model);
    lw_db_postprocess_workspace_free(&detector->postprocess_workspace);
    lw_det_preprocess_workspace_free(&detector->preprocess_workspace);
    free(detector->padded_source);
    free(detector);
}

lw_status lw_detector_get_info(const lw_detector* detector, lw_detector_info* info) {
    if (detector == NULL || info == NULL ||
        !lw_abi_copy_output_prefix(info, info->struct_size, &detector->info,
                                   sizeof(detector->info)))
        return LW_STATUS_INVALID_ARGUMENT;
    return LW_STATUS_OK;
}

lw_status lw_detector_set_reading_order(lw_detector* detector, uint32_t reading_order,
                                        lw_error* error) {
    if (detector == NULL || !lw_reading_order_is_valid(reading_order)) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "invalid reading order");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    detector->reading_order = reading_order;
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}

lw_status lw_detector_get_reading_order(const lw_detector* detector, uint32_t* reading_order,
                                        lw_error* error) {
    if (detector == NULL || reading_order == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "detector and output are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    *reading_order = detector->reading_order;
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}

static lw_status detector_detect_bgr_u8_impl(
    lw_detector* detector, const uint8_t* source, uint64_t source_byte_count, uint32_t source_width,
    uint32_t source_height, uint32_t source_stride, lw_detection_box* boxes, uint32_t box_capacity,
    lw_detection_result* result, lw_pipeline_component_profile* profile, lw_error* error) {
    uint64_t source_pixels;
    uint32_t resized_width;
    uint32_t resized_height;
    uint32_t box_count = 0u;
    float width_ratio;
    float height_ratio;
    lw_detection_result output;
    uint32_t result_size;
    uint64_t started;
    lw_status status;
    float* input;
    if (detector == NULL || source == NULL || result == NULL ||
        result->struct_size < sizeof(uint32_t) || (boxes == NULL && box_capacity != 0u)) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "detector, BGR source, initialized result, and valid box buffer are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    result_size = result->struct_size;
    clear_result(&output);
    if (source_width == 0u || source_height == 0u) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "source image dimensions must be positive");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    source_pixels = (uint64_t)source_width * source_height;
    if (source_pixels > detector->info.max_image_pixels) {
        lw_set_error(error, LW_STATUS_MEMORY_LIMIT, "source image exceeds max_image_pixels");
        return LW_STATUS_MEMORY_LIMIT;
    }
    started = lw_pipeline_profile_now(profile);
    status = lw_det_compute_size(source_width, source_height, detector->info.limit_side_length,
                                 &resized_width, &resized_height, &width_ratio, &height_ratio);
    if (status != LW_STATUS_OK) {
        lw_set_error(error, status, "unable to compute detector input size");
        return status;
    }
    output.resized_width = resized_width;
    output.resized_height = resized_height;
    output.width_ratio = width_ratio;
    output.height_ratio = height_ratio;
    status = ensure_session(detector, resized_width, resized_height, error);
    if (status != LW_STATUS_OK) {
        (void)lw_abi_copy_output_prefix(result, result_size, &output, sizeof(output));
        return status;
    }
    input = detector->input;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    if (detector->x64_instance != NULL) {
        uint64_t count = 0u;
        input = lw_x64_det_instance_input(detector->x64_instance, &count);
        if (input == NULL || count != detector->input_element_count) {
            lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "compiled DET input is invalid");
            (void)lw_abi_copy_output_prefix(result, result_size, &output, sizeof(output));
            return LW_STATUS_INVALID_ARGUMENT;
        }
    }
#endif
#if defined(LW_EXPERIMENTAL_DET_FIXED_POINT_RESIZE)
    {
        const uint8_t* preprocess_source = source;
        uint32_t preprocess_width = source_width;
        uint32_t preprocess_height = source_height;
        uint32_t preprocess_stride = source_stride;
        uint64_t preprocess_byte_count = source_byte_count;
        /* Tiny-image pad: sources with w+h < 64 are zero-padded to a 32x32
         * buffer before resize (reference behavior). The ratios stay based on
         * the original dimensions, so the postprocess mapping is unaffected. */
        if (source_width + source_height < 64u) {
            uint32_t padded_width = source_width > 32u ? source_width : 32u;
            uint32_t padded_height = source_height > 32u ? source_height : 32u;
            uint64_t padded_bytes = (uint64_t)padded_width * padded_height * 3u;
            uint32_t y;
            if (detector->padded_width != padded_width ||
                detector->padded_height != padded_height) {
                uint8_t* padded = (uint8_t*)realloc(detector->padded_source, (size_t)padded_bytes);
                if (padded == NULL) {
                    lw_set_error(error, LW_STATUS_OUT_OF_MEMORY,
                                 "tiny-image pad allocation failed");
                    (void)lw_abi_copy_output_prefix(result, result_size, &output, sizeof(output));
                    return LW_STATUS_OUT_OF_MEMORY;
                }
                detector->padded_source = padded;
                detector->padded_width = padded_width;
                detector->padded_height = padded_height;
            }
            memset(detector->padded_source, 0, (size_t)padded_bytes);
            for (y = 0u; y < source_height; ++y) {
                memcpy(detector->padded_source + (size_t)y * padded_width * 3u,
                       source + (size_t)y * source_stride, (size_t)source_width * 3u);
            }
            preprocess_source = detector->padded_source;
            preprocess_width = padded_width;
            preprocess_height = padded_height;
            preprocess_stride = padded_width * 3u;
            preprocess_byte_count = padded_bytes;
        }
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
        if (detector->x64_program != NULL && detector->x64_program->direct_nhwc != 0u) {
            status = lw_det_preprocess_bgr_u8_fixed_nhwc(
                preprocess_source, preprocess_byte_count, preprocess_width,
                preprocess_height, preprocess_stride, resized_width, resized_height,
                input, detector->input_element_count, &detector->preprocess_workspace,
                detector->session->thread_pool, detector->intra_op_thread_count);
        } else
#endif
        status = lw_det_preprocess_bgr_u8_fixed(
            preprocess_source, preprocess_byte_count, preprocess_width, preprocess_height,
            preprocess_stride, resized_width, resized_height, input,
            detector->input_element_count, &detector->preprocess_workspace,
            detector->session->thread_pool, detector->intra_op_thread_count);
    }
#else
    status = lw_det_preprocess_bgr_u8(source, source_byte_count, source_width, source_height,
                                      source_stride, resized_width, resized_height, input,
                                      detector->input_element_count);
#endif
    if (status != LW_STATUS_OK) {
        lw_set_error(error, status, "BGR source layout is invalid");
        (void)lw_abi_copy_output_prefix(result, result_size, &output, sizeof(output));
        return status;
    }
#if defined(__EMSCRIPTEN__) && defined(LW_WASM_COMPILED_DET)
    if (lw_profile_env_is_one("LW_WASM_DET_DEBUG")) {
        detector_debug_tensor(detector->x64_program != NULL &&
                              detector->x64_program->direct_nhwc != 0u
                                  ? "input_nhwc" : "input_nchw",
                              input, detector->input_element_count);
    }
#endif
    lw_pipeline_profile_add_elapsed(profile == NULL ? NULL : &profile->preprocess_nanoseconds,
                                    started, profile);
    started = lw_pipeline_profile_now(profile);
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    if (detector->x64_instance != NULL) {
        status = lw_x64_det_instance_run_profiled_ex(
            detector->x64_instance, detector->probabilities,
            detector->probability_element_count, detector->session->thread_pool,
            detector->intra_op_thread_count,
            profile == NULL ? NULL : &profile->execution, error);
        if (status == LW_STATUS_OK && profile != NULL)
            lw_profile_add_value(&profile->compiled_backend_runs, 1u);
    } else
#endif
    {
        status = profile == NULL
                     ? lw_execute_session_f32(detector->session, detector->input,
                                              detector->input_element_count,
                                              detector->probabilities,
                                              detector->probability_element_count, error)
                     : lw_execute_session_f32_profiled(
                           detector->session, detector->input, detector->input_element_count,
                           detector->probabilities, detector->probability_element_count,
                           &profile->execution, error);
        if (status == LW_STATUS_OK && profile != NULL)
            lw_profile_add_value(&profile->canonical_fallback_runs, 1u);
    }
    lw_pipeline_profile_add_elapsed(profile == NULL ? NULL : &profile->graph_nanoseconds, started,
                                    profile);
    if (status != LW_STATUS_OK) {
        (void)lw_abi_copy_output_prefix(result, result_size, &output, sizeof(output));
        return status;
    }
#if defined(__EMSCRIPTEN__) && defined(LW_WASM_COMPILED_DET)
    if (lw_profile_env_is_one("LW_WASM_DET_DEBUG"))
        detector_debug_tensor("probabilities", detector->probabilities,
                              detector->probability_element_count);
#endif
    started = lw_pipeline_profile_now(profile);
    status = lw_db_postprocess_f32_ws(detector->probabilities, resized_width, resized_height,
                                      detector->info.bitmap_threshold, detector->info.box_threshold,
                                      detector->info.unclip_ratio, detector->info.use_dilation,
                                      detector->info.max_candidates, source_width, source_height,
                                      width_ratio, height_ratio, boxes, box_capacity, &box_count,
                                      &detector->postprocess_workspace, profile);
    lw_pipeline_profile_add_elapsed(profile == NULL ? NULL : &profile->postprocess_nanoseconds,
                                    started, profile);
    output.box_count = box_count;
    output.required_box_capacity = box_count;
    (void)lw_abi_copy_output_prefix(result, result_size, &output, sizeof(output));
    if (status != LW_STATUS_OK) {
        lw_set_error(error, status,
                     status == LW_STATUS_OUT_OF_BOUNDS ? "box buffer capacity is insufficient"
                                                       : "detector postprocessing failed");
        return status;
    }
    if (boxes != NULL) {
        status = lw_sort_detection_boxes(boxes, box_count, detector->reading_order);
        if (status != LW_STATUS_OK) {
            lw_set_error(error, status, "unable to sort detection boxes");
            return status;
        }
    }
    (void)lw_abi_copy_output_prefix(result, result_size, &output, sizeof(output));
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}

lw_status lw_detector_detect_bgr_u8(lw_detector* detector, const uint8_t* source,
                                    uint64_t source_byte_count, uint32_t source_width,
                                    uint32_t source_height, uint32_t source_stride,
                                    lw_detection_box* boxes, uint32_t box_capacity,
                                    lw_detection_result* result, lw_error* error) {
    return detector_detect_bgr_u8_impl(detector, source, source_byte_count, source_width,
                                       source_height, source_stride, boxes, box_capacity, result,
                                       NULL, error);
}

lw_status lw_detector_detect_bgr_u8_profiled(
    lw_detector* detector, const uint8_t* source, uint64_t source_byte_count, uint32_t source_width,
    uint32_t source_height, uint32_t source_stride, lw_detection_box* boxes, uint32_t box_capacity,
    lw_detection_result* result, lw_pipeline_component_profile* profile, lw_error* error) {
    if (profile == NULL || profile->execution.struct_size != sizeof(profile->execution) ||
        profile->execution.clock == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "an initialized detector profile and clock are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    return detector_detect_bgr_u8_impl(detector, source, source_byte_count, source_width,
                                       source_height, source_stride, boxes, box_capacity, result,
                                       profile, error);
}
