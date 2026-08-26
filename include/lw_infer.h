#ifndef LW_INFER_H
#define LW_INFER_H

#include <stdint.h>

#if defined(_WIN32) && defined(LW_PPOCR_C_SHARED)
#  if defined(LW_PPOCR_C_BUILD)
#    define LW_API __declspec(dllexport)
#  else
#    define LW_API __declspec(dllimport)
#  endif
#elif defined(LW_PPOCR_C_SHARED) && defined(__GNUC__) && __GNUC__ >= 4
#  define LW_API __attribute__((visibility("default")))
#else
#  define LW_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define LW_ERROR_MESSAGE_CAPACITY 256u
#define LW_MAX_DIMS 8u

typedef enum lw_status {
    LW_STATUS_OK = 0,
    LW_STATUS_INVALID_ARGUMENT = 1,
    LW_STATUS_IO_ERROR = 2,
    LW_STATUS_OUT_OF_MEMORY = 3,
    LW_STATUS_INVALID_FORMAT = 4,
    LW_STATUS_UNSUPPORTED_VERSION = 5,
    LW_STATUS_OUT_OF_BOUNDS = 6,
    LW_STATUS_CHECKSUM_MISMATCH = 7,
    LW_STATUS_UNSUPPORTED = 8,
    LW_STATUS_INVALID_SHAPE = 9,
    LW_STATUS_MEMORY_LIMIT = 10
} lw_status;

typedef enum lw_dtype {
    LW_DTYPE_F32 = 1,
    LW_DTYPE_I32 = 2,
    LW_DTYPE_I64 = 3,
    LW_DTYPE_U8 = 4
} lw_dtype;

typedef struct lw_error {
    uint32_t struct_size;
    int32_t code;
    char message[LW_ERROR_MESSAGE_CAPACITY];
} lw_error;

typedef struct lw_model_options {
    uint32_t struct_size;
    uint32_t reserved;
    uint64_t max_file_size;
} lw_model_options;

typedef struct lw_model_info {
    uint32_t struct_size;
    uint16_t format_major;
    uint16_t format_minor;
    uint32_t tensor_count;
    uint32_t node_count;
    uint32_t input_count;
    uint32_t output_count;
    uint64_t file_size;
    uint64_t weight_size;
    uint64_t workspace_size;
    uint64_t content_checksum;
} lw_model_info;

typedef struct lw_tensor_desc {
    uint32_t struct_size;
    uint32_t dtype;
    uint32_t rank;
    uint32_t reserved;
    int32_t dimensions[LW_MAX_DIMS];
} lw_tensor_desc;

typedef struct lw_session_options {
    uint32_t struct_size;
    uint32_t reserved;
    uint64_t max_workspace_size;
    uint64_t max_tensor_size;
} lw_session_options;

typedef struct lw_session_info {
    uint32_t struct_size;
    uint32_t tensor_count;
    uint32_t input_count;
    uint32_t output_count;
    uint64_t workspace_size;
} lw_session_info;

typedef struct lw_recognizer_options {
    uint32_t struct_size;
    uint32_t target_width;
    uint32_t reserved0;
    uint32_t reserved1;
    uint64_t max_model_file_size;
    uint64_t max_workspace_size;
    uint64_t max_tensor_size;
    uint64_t max_image_pixels;
} lw_recognizer_options;

typedef struct lw_recognizer_info {
    uint32_t struct_size;
    uint32_t target_width;
    uint32_t input_height;
    uint32_t time_steps;
    uint32_t class_count;
    uint32_t reserved;
    uint64_t max_text_capacity;
    uint64_t workspace_size;
} lw_recognizer_info;

typedef struct lw_recognition_result {
    uint32_t struct_size;
    uint32_t emitted_count;
    float score;
    uint32_t resized_width;
    uint32_t time_steps;
    uint32_t reserved;
    uint64_t required_text_capacity;
} lw_recognition_result;

typedef struct lw_classifier_options {
    uint32_t struct_size;
    uint32_t reserved;
    uint64_t max_model_file_size;
    uint64_t max_workspace_size;
    uint64_t max_tensor_size;
    uint64_t max_image_pixels;
} lw_classifier_options;

typedef struct lw_classifier_info {
    uint32_t struct_size;
    uint32_t input_width;
    uint32_t input_height;
    uint32_t class_count;
    uint64_t workspace_size;
} lw_classifier_info;

typedef struct lw_classification_result {
    uint32_t struct_size;
    uint32_t label;
    float score;
    uint32_t resized_width;
    uint32_t orientation_degrees;
    uint32_t reserved;
} lw_classification_result;

typedef struct lw_detector_options {
    uint32_t struct_size;
    uint32_t limit_side_length;
    uint32_t max_candidates;
    uint32_t use_dilation;
    float bitmap_threshold;
    float box_threshold;
    float unclip_ratio;
    uint32_t reserved;
    uint64_t max_model_file_size;
    uint64_t max_workspace_size;
    uint64_t max_tensor_size;
    uint64_t max_image_pixels;
} lw_detector_options;

typedef struct lw_detector_info {
    uint32_t struct_size;
    uint32_t limit_side_length;
    uint32_t max_candidates;
    uint32_t use_dilation;
    float bitmap_threshold;
    float box_threshold;
    float unclip_ratio;
    uint32_t reserved;
    uint64_t max_image_pixels;
} lw_detector_info;

typedef struct lw_detection_box {
    float x1;
    float y1;
    float x2;
    float y2;
    float x3;
    float y3;
    float x4;
    float y4;
    float score;
    uint32_t reserved;
} lw_detection_box;

typedef struct lw_detection_result {
    uint32_t struct_size;
    uint32_t box_count;
    uint32_t required_box_capacity;
    uint32_t resized_width;
    uint32_t resized_height;
    uint32_t reserved;
    float width_ratio;
    float height_ratio;
} lw_detection_result;

typedef struct lw_ocr_options {
    uint32_t struct_size;
    uint32_t use_direction_classification;
    float classifier_threshold;
    uint32_t reserved;
    uint64_t max_crop_pixels;
    lw_detector_options detector;
    lw_classifier_options classifier;
    lw_recognizer_options recognizer;
} lw_ocr_options;

typedef struct lw_ocr_info {
    uint32_t struct_size;
    uint32_t use_direction_classification;
    uint32_t max_line_capacity;
    uint32_t reserved;
    uint64_t max_text_capacity;
    uint64_t max_text_capacity_per_line;
    uint64_t max_crop_pixels;
} lw_ocr_info;

typedef struct lw_ocr_line {
    lw_detection_box box;
    float recognition_score;
    float classification_score;
    uint32_t classification_label;
    uint32_t applied_rotation_degrees;
    uint32_t emitted_count;
    uint32_t reserved;
    uint64_t text_offset;
    uint64_t text_length;
} lw_ocr_line;

typedef struct lw_ocr_result {
    uint32_t struct_size;
    uint32_t line_count;
    uint32_t required_line_capacity;
    uint32_t detected_count;
    uint32_t detector_resized_width;
    uint32_t detector_resized_height;
    uint32_t reserved0;
    uint32_t reserved1;
    uint64_t required_text_capacity;
} lw_ocr_result;

typedef struct lw_model lw_model;
typedef struct lw_session lw_session;
typedef struct lw_recognizer lw_recognizer;
typedef struct lw_classifier lw_classifier;
typedef struct lw_detector lw_detector;
typedef struct lw_ocr lw_ocr;

LW_API void lw_model_options_init(lw_model_options* options);
LW_API void lw_model_info_init(lw_model_info* info);
LW_API void lw_error_init(lw_error* error);
LW_API lw_status lw_model_load(
    const char* path_utf8,
    const lw_model_options* options,
    lw_model** out_model,
    lw_error* error);
LW_API void lw_model_free(lw_model* model);
LW_API lw_status lw_model_get_info(const lw_model* model, lw_model_info* info);
LW_API void lw_tensor_desc_init(lw_tensor_desc* tensor);
LW_API void lw_session_options_init(lw_session_options* options);
LW_API void lw_session_info_init(lw_session_info* info);
LW_API lw_status lw_session_create(
    const lw_model* model,
    const lw_tensor_desc* inputs,
    uint32_t input_count,
    const lw_session_options* options,
    lw_session** out_session,
    lw_error* error);
LW_API void lw_session_free(lw_session* session);
LW_API lw_status lw_session_get_info(const lw_session* session, lw_session_info* info);
LW_API lw_status lw_session_get_output_desc(
    const lw_session* session,
    uint32_t output_index,
    lw_tensor_desc* output);
LW_API void lw_recognizer_options_init(lw_recognizer_options* options);
LW_API void lw_recognizer_info_init(lw_recognizer_info* info);
LW_API void lw_recognition_result_init(lw_recognition_result* result);
LW_API lw_status lw_recognizer_create(
    const char* model_path_utf8,
    const char* dictionary_path_utf8,
    const lw_recognizer_options* options,
    lw_recognizer** out_recognizer,
    lw_error* error);
LW_API void lw_recognizer_free(lw_recognizer* recognizer);
LW_API lw_status lw_recognizer_get_info(
    const lw_recognizer* recognizer,
    lw_recognizer_info* info);
LW_API lw_status lw_recognizer_recognize_bgr_u8(
    lw_recognizer* recognizer,
    const uint8_t* source,
    uint64_t source_byte_count,
    uint32_t source_width,
    uint32_t source_height,
    uint32_t source_stride,
    char* text_utf8,
    uint64_t text_capacity,
    lw_recognition_result* result,
    lw_error* error);
LW_API void lw_classifier_options_init(lw_classifier_options* options);
LW_API void lw_classifier_info_init(lw_classifier_info* info);
LW_API void lw_classification_result_init(lw_classification_result* result);
LW_API lw_status lw_classifier_create(
    const char* model_path_utf8,
    const lw_classifier_options* options,
    lw_classifier** out_classifier,
    lw_error* error);
LW_API void lw_classifier_free(lw_classifier* classifier);
LW_API lw_status lw_classifier_get_info(
    const lw_classifier* classifier,
    lw_classifier_info* info);
LW_API lw_status lw_classifier_classify_bgr_u8(
    lw_classifier* classifier,
    const uint8_t* source,
    uint64_t source_byte_count,
    uint32_t source_width,
    uint32_t source_height,
    uint32_t source_stride,
    lw_classification_result* result,
    lw_error* error);
LW_API void lw_detector_options_init(lw_detector_options* options);
LW_API void lw_detector_info_init(lw_detector_info* info);
LW_API void lw_detection_result_init(lw_detection_result* result);
LW_API lw_status lw_detector_create(
    const char* model_path_utf8,
    const lw_detector_options* options,
    lw_detector** out_detector,
    lw_error* error);
LW_API void lw_detector_free(lw_detector* detector);
LW_API lw_status lw_detector_get_info(
    const lw_detector* detector,
    lw_detector_info* info);
LW_API lw_status lw_detector_detect_bgr_u8(
    lw_detector* detector,
    const uint8_t* source,
    uint64_t source_byte_count,
    uint32_t source_width,
    uint32_t source_height,
    uint32_t source_stride,
    lw_detection_box* boxes,
    uint32_t box_capacity,
    lw_detection_result* result,
    lw_error* error);
LW_API void lw_ocr_options_init(lw_ocr_options* options);
LW_API void lw_ocr_info_init(lw_ocr_info* info);
LW_API void lw_ocr_result_init(lw_ocr_result* result);
LW_API lw_status lw_ocr_create(
    const char* detector_model_path_utf8,
    const char* classifier_model_path_utf8,
    const char* recognizer_model_path_utf8,
    const char* dictionary_path_utf8,
    const lw_ocr_options* options,
    lw_ocr** out_ocr,
    lw_error* error);
LW_API void lw_ocr_free(lw_ocr* ocr);
LW_API lw_status lw_ocr_get_info(const lw_ocr* ocr, lw_ocr_info* info);
LW_API lw_status lw_ocr_run_bgr_u8(
    lw_ocr* ocr,
    const uint8_t* source,
    uint64_t source_byte_count,
    uint32_t source_width,
    uint32_t source_height,
    uint32_t source_stride,
    lw_ocr_line* lines,
    uint32_t line_capacity,
    char* text_utf8,
    uint64_t text_capacity,
    lw_ocr_result* result,
    lw_error* error);
LW_API const char* lw_status_string(lw_status status);

#ifdef __cplusplus
}
#endif

#endif
