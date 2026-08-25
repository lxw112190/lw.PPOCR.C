#include "det_internal.h"
#include "lw_infer.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_u32(const char* text, uint32_t* value) {
    char* end = NULL;
    unsigned long parsed;
    errno = 0;
    parsed = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed > UINT32_MAX)
        return 0;
    *value = (uint32_t)parsed;
    return 1;
}

static int read_file(const char* path, uint8_t** bytes, size_t* byte_count) {
    FILE* file = fopen(path, "rb");
    long length;
    uint8_t* data;
    size_t count;
    *bytes = NULL;
    *byte_count = 0u;
    if (file == NULL || fseek(file, 0, SEEK_END) != 0 ||
        (length = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0) {
        if (file != NULL) fclose(file);
        return 0;
    }
    data = (uint8_t*)malloc(length == 0 ? 1u : (size_t)length);
    if (data == NULL) {
        fclose(file);
        return 0;
    }
    count = fread(data, 1u, (size_t)length, file);
    if (fclose(file) != 0 || count != (size_t)length) {
        free(data);
        return 0;
    }
    *bytes = data;
    *byte_count = count;
    return 1;
}

static int write_file(const char* path, const void* bytes, size_t byte_count) {
    FILE* file = fopen(path, "wb");
    size_t written;
    if (file == NULL) return 0;
    written = fwrite(bytes, 1u, byte_count, file);
    return fclose(file) == 0 && written == byte_count;
}

static int run_preprocess(int argc, char** argv) {
    uint8_t* source = NULL;
    size_t source_bytes = 0u;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t limit;
    uint32_t resized_width;
    uint32_t resized_height;
    float width_ratio;
    float height_ratio;
    uint64_t output_count;
    float* output = NULL;
    lw_status status;
    int return_code = 1;
    if (argc != 8 || !parse_u32(argv[3], &width) ||
        !parse_u32(argv[4], &height) || !parse_u32(argv[5], &stride) ||
        !parse_u32(argv[6], &limit) ||
        !read_file(argv[2], &source, &source_bytes)) return 2;
    status = lw_det_compute_size(width, height, limit, &resized_width,
                                 &resized_height, &width_ratio, &height_ratio);
    if (status != LW_STATUS_OK) goto cleanup;
    output_count = UINT64_C(3) * resized_width * resized_height;
    if (output_count > SIZE_MAX / sizeof(*output)) goto cleanup;
    output = (float*)malloc((size_t)output_count * sizeof(*output));
    if (output == NULL) goto cleanup;
    status = lw_det_preprocess_bgr_u8(
        source, source_bytes, width, height, stride, resized_width,
        resized_height, output, output_count - 1u);
    if (status != LW_STATUS_INVALID_SHAPE) goto cleanup;
    status = lw_det_preprocess_bgr_u8(
        source, source_bytes, width, height, stride, resized_width,
        resized_height, output, output_count);
    if (status != LW_STATUS_OK ||
        !write_file(argv[7], output, (size_t)output_count * sizeof(*output)))
        goto cleanup;
    printf("width=%u height=%u width_ratio=%.9g height_ratio=%.9g\n",
           resized_width, resized_height, (double)width_ratio,
           (double)height_ratio);
    return_code = 0;
cleanup:
    free(output);
    free(source);
    return return_code;
}

static int run_postprocess(void) {
    float prediction[32u * 32u];
    lw_detection_box box;
    lw_detection_box sentinel;
    uint32_t count = 0u;
    uint32_t x;
    uint32_t y;
    lw_status status;
    memset(prediction, 0, sizeof(prediction));
    for (y = 8u; y <= 15u; ++y)
        for (x = 5u; x <= 22u; ++x)
            prediction[y * 32u + x] = 0.9f;
    status = lw_db_postprocess_f32(
        prediction, 32u, 32u, 0.3f, 0.6f, 1.6f, 0u, 1000u,
        64u, 64u, 0.5f, 0.5f, NULL, 0u, &count);
    if (status != LW_STATUS_OK || count != 1u) return 1;
    memset(&sentinel, 0x5a, sizeof(sentinel));
    status = lw_db_postprocess_f32(
        prediction, 32u, 32u, 0.3f, 0.6f, 1.6f, 0u, 1000u,
        64u, 64u, 0.5f, 0.5f, &sentinel, 0u, &count);
    if (status != LW_STATUS_OUT_OF_BOUNDS || count != 1u) return 1;
    memset(&box, 0, sizeof(box));
    status = lw_db_postprocess_f32(
        prediction, 32u, 32u, 0.3f, 0.6f, 1.6f, 0u, 1000u,
        64u, 64u, 0.5f, 0.5f, &box, 1u, &count);
    if (status != LW_STATUS_OK || count != 1u ||
        !isfinite(box.score) || box.score < 0.89f || box.score > 0.91f)
        return 1;
    printf("count=%u score=%.9g box=%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
           count, (double)box.score, (double)box.x1, (double)box.y1,
           (double)box.x2, (double)box.y2, (double)box.x3,
           (double)box.y3, (double)box.x4, (double)box.y4);
    return 0;
}

static int run_pipeline(int argc, char** argv) {
    lw_detector_options options;
    lw_detector_info info;
    lw_detector* detector = NULL;
    lw_detector* rejected = NULL;
    lw_detector* limited = NULL;
    lw_detection_result query;
    lw_detection_result result;
    lw_detection_box* boxes = NULL;
    lw_error error;
    uint8_t* source = NULL;
    size_t source_bytes = 0u;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t index;
    lw_status status;
    int return_code = 1;
    if (argc != 7 || !parse_u32(argv[4], &width) ||
        !parse_u32(argv[5], &height) || !parse_u32(argv[6], &stride) ||
        !read_file(argv[3], &source, &source_bytes)) return 2;
    lw_detector_options_init(&options);
    options.reserved = 1u;
    lw_error_init(&error);
    status = lw_detector_create(argv[2], &options, &rejected, &error);
    if (status != LW_STATUS_INVALID_ARGUMENT || rejected != NULL) goto cleanup;
    lw_detector_options_init(&options);
    options.limit_side_length = 320u;
    lw_error_init(&error);
    status = lw_detector_create(argv[2], &options, &detector, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "detector create failed: %s: %s\n",
                lw_status_string(status), error.message);
        goto cleanup;
    }
    lw_detector_info_init(&info);
    if (lw_detector_get_info(detector, &info) != LW_STATUS_OK ||
        info.limit_side_length != 320u || info.max_candidates != 1000u)
        goto cleanup;
    memset(&query, 0, sizeof(query));
    lw_error_init(&error);
    status = lw_detector_detect_bgr_u8(
        detector, source, source_bytes, width, height, stride,
        NULL, 0u, &query, &error);
    if (status != LW_STATUS_INVALID_ARGUMENT) goto cleanup;
    lw_detection_result_init(&query);
    lw_error_init(&error);
    status = lw_detector_detect_bgr_u8(
        detector, source, source_bytes, width, height, stride,
        NULL, 0u, &query, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "detector query failed: %s: %s\n",
                lw_status_string(status), error.message);
        goto cleanup;
    }
    if (query.required_box_capacity != query.box_count || query.box_count == 0u)
        goto cleanup;
    boxes = (lw_detection_box*)calloc(query.box_count, sizeof(*boxes));
    if (boxes == NULL) goto cleanup;
    lw_detection_result_init(&result);
    lw_error_init(&error);
    status = lw_detector_detect_bgr_u8(
        detector, source, source_bytes, width, height, stride,
        boxes, query.box_count - 1u, &result, &error);
    if (status != LW_STATUS_OUT_OF_BOUNDS ||
        result.required_box_capacity != query.box_count) goto cleanup;
    lw_detection_result_init(&result);
    lw_error_init(&error);
    status = lw_detector_detect_bgr_u8(
        detector, source, source_bytes, width, height, stride,
        boxes, query.box_count, &result, &error);
    if (status != LW_STATUS_OK || result.box_count != query.box_count ||
        result.resized_width != 320u || result.resized_height != 320u)
        goto cleanup;
    printf("count=%u width=%u height=%u\n", result.box_count,
           result.resized_width, result.resized_height);
    for (index = 0u; index < result.box_count; ++index) {
        printf("box=%u score=%.9g points=%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
               index, (double)boxes[index].score, (double)boxes[index].x1,
               (double)boxes[index].y1, (double)boxes[index].x2,
               (double)boxes[index].y2, (double)boxes[index].x3,
               (double)boxes[index].y3, (double)boxes[index].x4,
               (double)boxes[index].y4);
    }
    lw_detector_options_init(&options);
    options.limit_side_length = 320u;
    options.max_image_pixels = (uint64_t)width * height - 1u;
    lw_error_init(&error);
    status = lw_detector_create(argv[2], &options, &limited, &error);
    if (status != LW_STATUS_OK || limited == NULL) goto cleanup;
    lw_detection_result_init(&result);
    lw_error_init(&error);
    status = lw_detector_detect_bgr_u8(
        limited, source, source_bytes, width, height, stride,
        NULL, 0u, &result, &error);
    if (status != LW_STATUS_MEMORY_LIMIT) goto cleanup;
    return_code = 0;
cleanup:
    free(boxes);
    free(source);
    lw_detector_free(limited);
    lw_detector_free(rejected);
    lw_detector_free(detector);
    return return_code;
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    if (strcmp(argv[1], "preprocess") == 0) return run_preprocess(argc, argv);
    if (strcmp(argv[1], "postprocess") == 0) return run_postprocess();
    if (strcmp(argv[1], "pipeline") == 0) return run_pipeline(argc, argv);
    return 2;
}
