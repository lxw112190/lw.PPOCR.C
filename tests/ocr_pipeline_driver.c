#include "crop_internal.h"
#include "lw_infer.h"

#include <errno.h>
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
    if (file == NULL || fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        if (file != NULL)
            fclose(file);
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
    if (file == NULL)
        return 0;
    written = fwrite(bytes, 1u, byte_count, file);
    return fclose(file) == 0 && written == byte_count;
}

static int run_crop(int argc, char** argv) {
    lw_detection_box box;
    uint8_t* source = NULL;
    uint8_t* crop = NULL;
    size_t source_bytes = 0u;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t vertical;
    uint32_t crop_width;
    uint32_t crop_height;
    uint64_t crop_bytes;
    lw_status status;
    int return_code = 1;
    if (argc != 8 || !parse_u32(argv[3], &width) || !parse_u32(argv[4], &height) ||
        !parse_u32(argv[5], &stride) || !parse_u32(argv[7], &vertical) || vertical > 1u ||
        !read_file(argv[2], &source, &source_bytes))
        return 2;
    memset(&box, 0, sizeof(box));
    if (vertical == 0u) {
        box.x1 = 3.2f;
        box.y1 = 2.4f;
        box.x2 = 24.6f;
        box.y2 = 1.2f;
        box.x3 = 26.1f;
        box.y3 = 11.8f;
        box.x4 = 2.1f;
        box.y4 = 13.4f;
    } else {
        box.x1 = 8.0f;
        box.y1 = 1.0f;
        box.x2 = 14.0f;
        box.y2 = 2.0f;
        box.x3 = 12.0f;
        box.y3 = 16.0f;
        box.x4 = 6.0f;
        box.y4 = 15.0f;
    }
    status = lw_crop_quad_size(&box, &crop_width, &crop_height, &crop_bytes);
    if (status != LW_STATUS_OK || crop_bytes > SIZE_MAX)
        goto cleanup;
    crop = (uint8_t*)malloc((size_t)crop_bytes);
    if (crop == NULL)
        goto cleanup;
    status = lw_crop_quad_bgr_u8(source, source_bytes, width, height, stride, &box, crop,
                                 crop_bytes, &crop_width, &crop_height, &crop_bytes);
    if (status != LW_STATUS_OK || !write_file(argv[6], crop, (size_t)crop_bytes))
        goto cleanup;
    printf("width=%u height=%u bytes=%llu\n", crop_width, crop_height,
           (unsigned long long)crop_bytes);
    return_code = 0;
cleanup:
    free(crop);
    free(source);
    return return_code;
}

static int run_pipeline(int argc, char** argv) {
    lw_ocr_options options;
    lw_ocr_info info;
    lw_ocr_result query;
    lw_ocr_result result;
    lw_ocr* ocr = NULL;
    lw_ocr* disabled = NULL;
    lw_ocr* limited = NULL;
    lw_ocr* rejected = NULL;
    lw_ocr_line* lines = NULL;
    char* text = NULL;
    uint8_t* source = NULL;
    size_t source_bytes = 0u;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t use_classifier;
    uint32_t index;
    lw_error error;
    lw_status status;
    int return_code = 1;
    if (argc != 11 || !parse_u32(argv[7], &width) || !parse_u32(argv[8], &height) ||
        !parse_u32(argv[9], &stride) || !parse_u32(argv[10], &use_classifier) ||
        use_classifier > 1u || !read_file(argv[6], &source, &source_bytes))
        return 2;
    lw_ocr_options_init(&options);
#if INTPTR_MAX > INT32_MAX
    if (options.worker_count == 0u || options.worker_count > 8u)
        goto cleanup;
#else
    if (options.worker_count != 1u)
        goto cleanup;
#endif
    options.classifier.reserved = 1u;
    lw_error_init(&error);
    status = lw_ocr_create(argv[2], argv[3], argv[4], argv[5], &options, &rejected, &error);
    if (status != LW_STATUS_INVALID_ARGUMENT || rejected != NULL)
        goto cleanup;
    lw_ocr_options_init(&options);
    options.worker_count = 17u;
    lw_error_init(&error);
    status = lw_ocr_create(argv[2], argv[3], argv[4], argv[5], &options, &rejected, &error);
    if (status != LW_STATUS_INVALID_ARGUMENT || rejected != NULL)
        goto cleanup;
    lw_ocr_options_init(&options);
    options.use_direction_classification = 0u;
    options.detector.limit_side_length = 320u;
    lw_error_init(&error);
    status = lw_ocr_create(argv[2], NULL, argv[4], argv[5], &options, &disabled, &error);
    if (status != LW_STATUS_OK || disabled == NULL)
        goto cleanup;
    lw_ocr_free(disabled);
    disabled = NULL;

    lw_ocr_options_init(&options);
    options.use_direction_classification = use_classifier;
    options.detector.limit_side_length = 320u;
    lw_error_init(&error);
    status = lw_ocr_create(argv[2], use_classifier != 0u ? argv[3] : NULL, argv[4], argv[5],
                           &options, &ocr, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "OCR create failed: %s: %s\n", lw_status_string(status), error.message);
        goto cleanup;
    }
    lw_ocr_info_init(&info);
    if (lw_ocr_get_info(ocr, &info) != LW_STATUS_OK ||
        info.use_direction_classification != use_classifier || info.max_line_capacity == 0u ||
        info.max_text_capacity == 0u || info.worker_count == 0u || info.worker_count > 16u)
        goto cleanup;
    memset(&query, 0, sizeof(query));
    lw_error_init(&error);
    status = lw_ocr_run_bgr_u8(ocr, source, source_bytes, width, height, stride, NULL, 0u, NULL, 0u,
                               &query, &error);
    if (status != LW_STATUS_INVALID_ARGUMENT)
        goto cleanup;
    lw_ocr_result_init(&query);
    lw_error_init(&error);
    status = lw_ocr_run_bgr_u8(ocr, source, source_bytes, width, height, stride, NULL, 0u, NULL, 0u,
                               &query, &error);
    if (status != LW_STATUS_OK || query.line_count == 0u ||
        query.required_line_capacity != query.line_count || query.required_text_capacity == 0u) {
        fprintf(stderr, "OCR query failed: %s: %s\n", lw_status_string(status), error.message);
        goto cleanup;
    }
    lines = (lw_ocr_line*)calloc(query.line_count, sizeof(*lines));
    text = (char*)malloc((size_t)query.required_text_capacity);
    if (lines == NULL || text == NULL)
        goto cleanup;
    memset(lines, 0x5a, (size_t)query.line_count * sizeof(*lines));
    memset(text, 0x5a, (size_t)query.required_text_capacity);
    lw_ocr_result_init(&result);
    lw_error_init(&error);
    status = lw_ocr_run_bgr_u8(ocr, source, source_bytes, width, height, stride, lines,
                               query.line_count - 1u, text, query.required_text_capacity, &result,
                               &error);
    if (status != LW_STATUS_OUT_OF_BOUNDS || result.required_line_capacity != query.line_count ||
        ((const unsigned char*)lines)[0] != 0x5au || (unsigned char)text[0] != 0x5au)
        goto cleanup;
    lw_ocr_result_init(&result);
    lw_error_init(&error);
    status =
        lw_ocr_run_bgr_u8(ocr, source, source_bytes, width, height, stride, lines, query.line_count,
                          text, query.required_text_capacity - 1u, &result, &error);
    if (status != LW_STATUS_OUT_OF_BOUNDS ||
        result.required_text_capacity != query.required_text_capacity ||
        ((const unsigned char*)lines)[0] != 0x5au || (unsigned char)text[0] != 0x5au)
        goto cleanup;
    lw_ocr_result_init(&result);
    lw_error_init(&error);
    status =
        lw_ocr_run_bgr_u8(ocr, source, source_bytes, width, height, stride, lines, query.line_count,
                          text, query.required_text_capacity, &result, &error);
    if (status != LW_STATUS_OK || result.line_count != query.line_count ||
        result.required_text_capacity != query.required_text_capacity)
        goto cleanup;
    printf("lines=%u detected=%u text_bytes=%llu resized=%ux%u cls=%u\n", result.line_count,
           result.detected_count, (unsigned long long)result.required_text_capacity,
           result.detector_resized_width, result.detector_resized_height, use_classifier);
    for (index = 0u; index < result.line_count; ++index) {
        const lw_ocr_line* line = &lines[index];
        if (line->text_offset + line->text_length >= result.required_text_capacity ||
            text[line->text_offset + line->text_length] != '\0')
            goto cleanup;
        printf("line=%u det=%.9g rec=%.9g cls_label=%u cls=%.9g rotation=%u "
               "point=%.9g,%.9g text=%s\n",
               index, (double)line->box.score, (double)line->recognition_score,
               line->classification_label, (double)line->classification_score,
               line->applied_rotation_degrees, (double)line->box.x1, (double)line->box.y1,
               text + line->text_offset);
    }
    lw_ocr_options_init(&options);
    options.use_direction_classification = 0u;
    options.detector.limit_side_length = 320u;
    options.max_crop_pixels = 1u;
    lw_error_init(&error);
    status = lw_ocr_create(argv[2], NULL, argv[4], argv[5], &options, &limited, &error);
    if (status != LW_STATUS_OK || limited == NULL)
        goto cleanup;
    lw_ocr_result_init(&result);
    lw_error_init(&error);
    status = lw_ocr_run_bgr_u8(limited, source, source_bytes, width, height, stride, NULL, 0u, NULL,
                               0u, &result, &error);
    if (status != LW_STATUS_MEMORY_LIMIT)
        goto cleanup;
    return_code = 0;
cleanup:
    free(text);
    free(lines);
    free(source);
    lw_ocr_free(rejected);
    lw_ocr_free(limited);
    lw_ocr_free(disabled);
    lw_ocr_free(ocr);
    return return_code;
}

static int run_golden(int argc, char** argv) {
    lw_ocr_options options;
    lw_ocr_info info;
    lw_ocr_result result;
    lw_ocr* ocr = NULL;
    lw_ocr_line* lines = NULL;
    char* text = NULL;
    uint8_t* source = NULL;
    size_t source_bytes = 0u;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t use_classifier;
    uint32_t index;
    lw_error error;
    lw_status status;
    int return_code = 1;
    if (argc != 11 || !parse_u32(argv[7], &width) || !parse_u32(argv[8], &height) ||
        !parse_u32(argv[9], &stride) || !parse_u32(argv[10], &use_classifier) ||
        use_classifier > 1u || !read_file(argv[6], &source, &source_bytes))
        return 2;

    lw_ocr_options_init(&options);
    options.use_direction_classification = use_classifier;
    options.detector.limit_side_length = 320u;
    lw_error_init(&error);
    status = lw_ocr_create(argv[2], use_classifier != 0u ? argv[3] : NULL, argv[4], argv[5],
                           &options, &ocr, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "OCR create failed: %s: %s\n", lw_status_string(status), error.message);
        goto cleanup;
    }
    lw_ocr_info_init(&info);
    status = lw_ocr_get_info(ocr, &info);
    if (status != LW_STATUS_OK || info.max_line_capacity == 0u ||
        info.max_text_capacity == 0u || info.max_text_capacity > SIZE_MAX)
        goto cleanup;
    lines = (lw_ocr_line*)calloc(info.max_line_capacity, sizeof(*lines));
    text = (char*)malloc((size_t)info.max_text_capacity);
    if (lines == NULL || text == NULL)
        goto cleanup;

    lw_ocr_result_init(&result);
    lw_error_init(&error);
    status = lw_ocr_run_bgr_u8(ocr, source, source_bytes, width, height, stride, lines,
                               info.max_line_capacity, text, info.max_text_capacity, &result,
                               &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "OCR run failed: %s: %s\n", lw_status_string(status), error.message);
        goto cleanup;
    }
    printf("lines=%u detected=%u text_bytes=%llu resized=%ux%u cls=%u\n", result.line_count,
           result.detected_count, (unsigned long long)result.required_text_capacity,
           result.detector_resized_width, result.detector_resized_height, use_classifier);
    for (index = 0u; index < result.line_count; ++index) {
        const lw_ocr_line* line = &lines[index];
        if (line->text_offset + line->text_length >= result.required_text_capacity ||
            text[line->text_offset + line->text_length] != '\0')
            goto cleanup;
        printf("line=%u det=%.9g rec=%.9g cls_label=%u cls=%.9g rotation=%u "
               "box=%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g text=%s\n",
               index, (double)line->box.score, (double)line->recognition_score,
               line->classification_label, (double)line->classification_score,
               line->applied_rotation_degrees, (double)line->box.x1, (double)line->box.y1,
               (double)line->box.x2, (double)line->box.y2, (double)line->box.x3,
               (double)line->box.y3, (double)line->box.x4, (double)line->box.y4,
               text + line->text_offset);
    }
    return_code = 0;
cleanup:
    free(text);
    free(lines);
    free(source);
    lw_ocr_free(ocr);
    return return_code;
}

int main(int argc, char** argv) {
    if (argc < 2)
        return 2;
    if (strcmp(argv[1], "crop") == 0)
        return run_crop(argc, argv);
    if (strcmp(argv[1], "pipeline") == 0)
        return run_pipeline(argc, argv);
    if (strcmp(argv[1], "golden") == 0)
        return run_golden(argc, argv);
    return 2;
}
