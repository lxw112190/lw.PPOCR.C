#include "lw_infer.h"

#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <shellapi.h>
#endif

#define DEMO_MAX_IMAGE_PIXELS UINT64_C(40000000)

#if defined(_WIN32)
static FILE* open_read_utf8(const char* path_utf8) {
    int wide_count;
    wchar_t* wide_path;
    FILE* file = NULL;
    wide_count = MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, path_utf8, -1, NULL, 0);
    if (wide_count <= 0) {
        return NULL;
    }
    wide_path = (wchar_t*)malloc((size_t)wide_count * sizeof(*wide_path));
    if (wide_path == NULL) {
        return NULL;
    }
    if (MultiByteToWideChar(
            CP_UTF8, MB_ERR_INVALID_CHARS, path_utf8, -1,
            wide_path, wide_count) > 0 &&
        _wfopen_s(&file, wide_path, L"rb") != 0) {
        file = NULL;
    }
    free(wide_path);
    return file;
}
#else
static FILE* open_read_utf8(const char* path_utf8) {
    return fopen(path_utf8, "rb");
}
#endif

static int read_ppm_token(FILE* file, char* token, size_t capacity) {
    int character;
    size_t length = 0u;
    do {
        character = fgetc(file);
        if (character == '#') {
            do {
                character = fgetc(file);
            } while (character != '\n' && character != EOF);
        }
    } while (character != EOF && isspace((unsigned char)character));
    if (character == EOF) {
        return 0;
    }
    do {
        if (length + 1u >= capacity) {
            return 0;
        }
        token[length++] = (char)character;
        character = fgetc(file);
    } while (character != EOF && !isspace((unsigned char)character));
    if (character == '\r') {
        int next = fgetc(file);
        if (next != '\n' && next != EOF) {
            ungetc(next, file);
        }
    }
    token[length] = '\0';
    return length != 0u;
}

static int parse_u32(const char* text, uint32_t* value) {
    char* end = NULL;
    unsigned long parsed;
    errno = 0;
    parsed = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed == 0u || parsed > UINT32_MAX) {
        return 0;
    }
    *value = (uint32_t)parsed;
    return 1;
}

static int load_ppm_bgr(
    const char* path_utf8,
    uint8_t** out_pixels,
    uint64_t* out_byte_count,
    uint32_t* out_width,
    uint32_t* out_height) {
    FILE* file = NULL;
    char token[64];
    uint32_t width;
    uint32_t height;
    uint32_t max_value;
    uint64_t pixel_count;
    uint64_t byte_count;
    uint8_t* pixels = NULL;
    uint64_t index;
    int result = 0;
    *out_pixels = NULL;
    *out_byte_count = 0u;
    *out_width = 0u;
    *out_height = 0u;
    file = open_read_utf8(path_utf8);
    if (file == NULL || !read_ppm_token(file, token, sizeof(token)) ||
        strcmp(token, "P6") != 0 ||
        !read_ppm_token(file, token, sizeof(token)) || !parse_u32(token, &width) ||
        !read_ppm_token(file, token, sizeof(token)) || !parse_u32(token, &height) ||
        !read_ppm_token(file, token, sizeof(token)) ||
        !parse_u32(token, &max_value) || max_value != 255u) {
        goto cleanup;
    }
    pixel_count = (uint64_t)width * height;
    if (pixel_count > DEMO_MAX_IMAGE_PIXELS || pixel_count > SIZE_MAX / 3u) {
        goto cleanup;
    }
    byte_count = pixel_count * 3u;
    pixels = (uint8_t*)malloc((size_t)byte_count);
    if (pixels == NULL ||
        fread(pixels, 1u, (size_t)byte_count, file) != (size_t)byte_count) {
        goto cleanup;
    }
    for (index = 0u; index < byte_count; index += 3u) {
        uint8_t red = pixels[(size_t)index];
        pixels[(size_t)index] = pixels[(size_t)index + 2u];
        pixels[(size_t)index + 2u] = red;
    }
    *out_pixels = pixels;
    *out_byte_count = byte_count;
    *out_width = width;
    *out_height = height;
    pixels = NULL;
    result = 1;
cleanup:
    free(pixels);
    if (file != NULL) {
        fclose(file);
    }
    return result;
}

static int demo_main(int argc, char** argv) {
    lw_recognizer* recognizer = NULL;
    lw_recognizer_info info;
    lw_recognition_result recognition;
    lw_error error;
    uint8_t* pixels = NULL;
    uint64_t byte_count = 0u;
    uint32_t width = 0u;
    uint32_t height = 0u;
    char* text = NULL;
    lw_status status;
    int result = 1;
    if (argc != 4) {
        fprintf(stderr, "usage: lw-recognize-ppm <rec.lwm> <dictionary.txt> <image.ppm>\n");
        return 2;
    }
    if (!load_ppm_bgr(argv[3], &pixels, &byte_count, &width, &height) ||
        width > UINT32_MAX / 3u) {
        fprintf(stderr, "invalid P6 PPM image: %s\n", argv[3]);
        goto cleanup;
    }
    lw_error_init(&error);
    status = lw_recognizer_create(argv[1], argv[2], NULL, &recognizer, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "create failed: %s: %s\n",
                lw_status_string(status), error.message);
        goto cleanup;
    }
    lw_recognizer_info_init(&info);
    if (lw_recognizer_get_info(recognizer, &info) != LW_STATUS_OK ||
        info.max_text_capacity == 0u || info.max_text_capacity > SIZE_MAX) {
        fprintf(stderr, "unable to query recognizer information\n");
        goto cleanup;
    }
    text = (char*)malloc((size_t)info.max_text_capacity);
    if (text == NULL) {
        fprintf(stderr, "unable to allocate text buffer\n");
        goto cleanup;
    }
    lw_recognition_result_init(&recognition);
    lw_error_init(&error);
    status = lw_recognizer_recognize_bgr_u8(
        recognizer, pixels, byte_count, width, height, width * 3u,
        text, info.max_text_capacity, &recognition, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "recognition failed: %s: %s\n",
                lw_status_string(status), error.message);
        goto cleanup;
    }
    printf("text=%s\n", text);
    printf("score=%.8f chars=%u image=%ux%u resized_width=%u time_steps=%u\n",
           recognition.score, recognition.emitted_count, width, height,
           recognition.resized_width, recognition.time_steps);
    result = 0;
cleanup:
    free(text);
    free(pixels);
    lw_recognizer_free(recognizer);
    return result;
}

#if defined(_WIN32)
int main(void) {
    wchar_t** wide_argv;
    char** utf8_argv;
    int argc;
    int index;
    int result = 2;
    SetConsoleOutputCP(CP_UTF8);
    wide_argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (wide_argv == NULL || argc <= 0) {
        return 2;
    }
    utf8_argv = (char**)calloc((size_t)argc, sizeof(*utf8_argv));
    if (utf8_argv == NULL) {
        LocalFree(wide_argv);
        return 2;
    }
    for (index = 0; index < argc; ++index) {
        int bytes = WideCharToMultiByte(
            CP_UTF8, WC_ERR_INVALID_CHARS, wide_argv[index], -1,
            NULL, 0, NULL, NULL);
        if (bytes <= 0) {
            goto cleanup;
        }
        utf8_argv[index] = (char*)malloc((size_t)bytes);
        if (utf8_argv[index] == NULL ||
            WideCharToMultiByte(
                CP_UTF8, WC_ERR_INVALID_CHARS, wide_argv[index], -1,
                utf8_argv[index], bytes, NULL, NULL) <= 0) {
            goto cleanup;
        }
    }
    result = demo_main(argc, utf8_argv);
cleanup:
    for (index = 0; index < argc; ++index) {
        free(utf8_argv[index]);
    }
    free(utf8_argv);
    LocalFree(wide_argv);
    return result;
}
#else
int main(int argc, char** argv) {
    return demo_main(argc, argv);
}
#endif
