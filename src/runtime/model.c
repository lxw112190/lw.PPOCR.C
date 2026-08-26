#include "model_internal.h"

/*
 * Model handle lifecycle and platform-specific UTF-8 file opening.
 * A model is not published to the caller until the entire untrusted LWM byte
 * stream has passed validate.c, so later runtime code may rely on its bounds.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

#define LW_DEFAULT_MAX_FILE_SIZE (UINT64_C(1024) * UINT64_C(1024) * UINT64_C(1024))

void lw_set_error(lw_error* error, lw_status status, const char* message) {
    if (error == NULL) {
        return;
    }
    error->struct_size = (uint32_t)sizeof(*error);
    error->code = (int32_t)status;
    if (message == NULL) {
        error->message[0] = '\0';
        return;
    }
#if defined(_MSC_VER)
    (void)strncpy_s(error->message, sizeof(error->message), message, _TRUNCATE);
#else
    (void)snprintf(error->message, sizeof(error->message), "%s", message);
#endif
}

void lw_model_options_init(lw_model_options* options) {
    if (options == NULL) {
        return;
    }
    memset(options, 0, sizeof(*options));
    options->struct_size = (uint32_t)sizeof(*options);
    options->max_file_size = LW_DEFAULT_MAX_FILE_SIZE;
}

void lw_error_init(lw_error* error) {
    if (error == NULL) {
        return;
    }
    memset(error, 0, sizeof(*error));
    error->struct_size = (uint32_t)sizeof(*error);
}

void lw_model_info_init(lw_model_info* info) {
    if (info == NULL) {
        return;
    }
    memset(info, 0, sizeof(*info));
    info->struct_size = (uint32_t)sizeof(*info);
}

#if defined(_WIN32)
static FILE* lw_open_read_utf8(const char* path_utf8) {
    int wide_count;
    wchar_t* wide_path;
    FILE* file = NULL;
    wide_count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path_utf8, -1, NULL, 0);
    if (wide_count <= 0) {
        return NULL;
    }
    wide_path = (wchar_t*)malloc((size_t)wide_count * sizeof(wchar_t));
    if (wide_path == NULL) {
        return NULL;
    }
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path_utf8, -1, wide_path, wide_count) >
        0) {
        if (_wfopen_s(&file, wide_path, L"rb") != 0) {
            file = NULL;
        }
    }
    free(wide_path);
    return file;
}
#else
static FILE* lw_open_read_utf8(const char* path_utf8) {
    return fopen(path_utf8, "rb");
}
#endif

lw_status lw_model_load(const char* path_utf8, const lw_model_options* options,
                        lw_model** out_model, lw_error* error) {
    FILE* file;
    long length;
    uint64_t limit = LW_DEFAULT_MAX_FILE_SIZE;
    lw_model* model;
    size_t read_count;
    lw_status status;

    if (out_model != NULL) {
        *out_model = NULL;
    }
    if (path_utf8 == NULL || path_utf8[0] == '\0' || out_model == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "path and out_model are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    if (options != NULL) {
        if (options->struct_size != sizeof(*options) || options->reserved != 0u) {
            lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "invalid model options structure");
            return LW_STATUS_INVALID_ARGUMENT;
        }
        if (options->max_file_size != 0u) {
            limit = options->max_file_size;
        }
    }
    file = lw_open_read_utf8(path_utf8);
    if (file == NULL) {
        lw_set_error(error, LW_STATUS_IO_ERROR, "unable to open model file");
        return LW_STATUS_IO_ERROR;
    }
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        lw_set_error(error, LW_STATUS_IO_ERROR, "unable to determine model file size");
        return LW_STATUS_IO_ERROR;
    }
    if ((uint64_t)length > limit || (uint64_t)length > (uint64_t)SIZE_MAX) {
        fclose(file);
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "model file exceeds configured size limit");
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    if ((uint64_t)length < LWM_V0_HEADER_SIZE) {
        fclose(file);
        lw_set_error(error, LW_STATUS_INVALID_FORMAT, "model file is smaller than the LWM header");
        return LW_STATUS_INVALID_FORMAT;
    }
    model = (lw_model*)calloc(1u, sizeof(*model));
    if (model == NULL) {
        fclose(file);
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate model handle");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    model->byte_count = (size_t)length;
    model->bytes = (uint8_t*)malloc(model->byte_count);
    if (model->bytes == NULL) {
        fclose(file);
        free(model);
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate model buffer");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    read_count = fread(model->bytes, 1u, model->byte_count, file);
    fclose(file);
    if (read_count != model->byte_count) {
        lw_model_free(model);
        lw_set_error(error, LW_STATUS_IO_ERROR, "unable to read complete model file");
        return LW_STATUS_IO_ERROR;
    }
    /* Do not expose even a successfully read file until its complete binary
     * structure, operator subset, checksums and table ranges are trusted. */
    status = lw_validate_lwm_v0(model, error);
    if (status != LW_STATUS_OK) {
        lw_model_free(model);
        return status;
    }
    *out_model = model;
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}

void lw_model_free(lw_model* model) {
    if (model == NULL) {
        return;
    }
    free(model->bytes);
    model->bytes = NULL;
    free(model);
}

lw_status lw_model_get_info(const lw_model* model, lw_model_info* info) {
    if (model == NULL || info == NULL || info->struct_size != sizeof(*info)) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    *info = model->info;
    return LW_STATUS_OK;
}

const char* lw_status_string(lw_status status) {
    switch (status) {
    case LW_STATUS_OK:
        return "ok";
    case LW_STATUS_INVALID_ARGUMENT:
        return "invalid_argument";
    case LW_STATUS_IO_ERROR:
        return "io_error";
    case LW_STATUS_OUT_OF_MEMORY:
        return "out_of_memory";
    case LW_STATUS_INVALID_FORMAT:
        return "invalid_format";
    case LW_STATUS_UNSUPPORTED_VERSION:
        return "unsupported_version";
    case LW_STATUS_OUT_OF_BOUNDS:
        return "out_of_bounds";
    case LW_STATUS_CHECKSUM_MISMATCH:
        return "checksum_mismatch";
    case LW_STATUS_UNSUPPORTED:
        return "unsupported";
    case LW_STATUS_INVALID_SHAPE:
        return "invalid_shape";
    case LW_STATUS_MEMORY_LIMIT:
        return "memory_limit";
    default:
        return "unknown";
    }
}
