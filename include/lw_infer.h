#ifndef LW_INFER_H
#define LW_INFER_H

#include <stdint.h>

#if defined(_WIN32) && defined(LW_PPOCR_C_SHARED)
#  if defined(LW_PPOCR_C_BUILD)
#    define LW_API __declspec(dllexport)
#  else
#    define LW_API __declspec(dllimport)
#  endif
#else
#  define LW_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define LW_ERROR_MESSAGE_CAPACITY 256u

typedef enum lw_status {
    LW_STATUS_OK = 0,
    LW_STATUS_INVALID_ARGUMENT = 1,
    LW_STATUS_IO_ERROR = 2,
    LW_STATUS_OUT_OF_MEMORY = 3,
    LW_STATUS_INVALID_FORMAT = 4,
    LW_STATUS_UNSUPPORTED_VERSION = 5,
    LW_STATUS_OUT_OF_BOUNDS = 6,
    LW_STATUS_CHECKSUM_MISMATCH = 7,
    LW_STATUS_UNSUPPORTED = 8
} lw_status;

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

typedef struct lw_model lw_model;

LW_API void lw_model_options_init(lw_model_options* options);
LW_API void lw_error_init(lw_error* error);
LW_API lw_status lw_model_load(
    const char* path_utf8,
    const lw_model_options* options,
    lw_model** out_model,
    lw_error* error);
LW_API void lw_model_free(lw_model* model);
LW_API lw_status lw_model_get_info(const lw_model* model, lw_model_info* info);
LW_API const char* lw_status_string(lw_status status);

#ifdef __cplusplus
}
#endif

#endif
