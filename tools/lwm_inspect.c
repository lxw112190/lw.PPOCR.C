#include "lw_infer.h"

/* Load an LWM through the public validator and print its deployment metadata. */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <shellapi.h>
#endif

static int inspect_model(const char* path_utf8) {
    lw_model* model = NULL;
    lw_error error;
    lw_model_info info;
    lw_status status;
    lw_error_init(&error);
    status = lw_model_load(path_utf8, NULL, &model, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "%s: %s\n", lw_status_string(status), error.message);
        return 1;
    }
    memset(&info, 0, sizeof(info));
    info.struct_size = (uint32_t)sizeof(info);
    status = lw_model_get_info(model, &info);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "%s\n", lw_status_string(status));
        lw_model_free(model);
        return 1;
    }
    printf("{\"format\":\"LWM\",\"version\":\"%u.%u\",\"tensors\":%" PRIu32 ",\"nodes\":%" PRIu32
           ",\"inputs\":%" PRIu32 ",\"outputs\":%" PRIu32 ",\"file_size\":%" PRIu64
           ",\"weight_size\":%" PRIu64 ",\"workspace_size\":%" PRIu64
           ",\"checksum\":\"0x%016" PRIx64 "\"}\n",
           (unsigned)info.format_major, (unsigned)info.format_minor, info.tensor_count,
           info.node_count, info.input_count, info.output_count, info.file_size, info.weight_size,
           info.workspace_size, info.content_checksum);
    lw_model_free(model);
    return 0;
}

#if defined(_WIN32)
int main(void) {
    int argc = 0;
    wchar_t** argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    int utf8_count;
    char* path_utf8;
    int result;
    if (argv == NULL || argc != 2) {
        fprintf(stderr, "usage: lwm-inspect <model.lwm>\n");
        if (argv != NULL) {
            LocalFree(argv);
        }
        return 2;
    }
    utf8_count =
        WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[1], -1, NULL, 0, NULL, NULL);
    if (utf8_count <= 0) {
        LocalFree(argv);
        fprintf(stderr, "invalid UTF-16 model path\n");
        return 2;
    }
    path_utf8 = (char*)malloc((size_t)utf8_count);
    if (path_utf8 == NULL) {
        LocalFree(argv);
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[1], -1, path_utf8, utf8_count, NULL,
                            NULL) <= 0) {
        free(path_utf8);
        LocalFree(argv);
        fprintf(stderr, "unable to encode model path\n");
        return 2;
    }
    result = inspect_model(path_utf8);
    free(path_utf8);
    LocalFree(argv);
    return result;
}
#else
int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: lwm-inspect <model.lwm>\n");
        return 2;
    }
    return inspect_model(argv[1]);
}
#endif
