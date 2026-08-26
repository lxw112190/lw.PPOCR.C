#ifndef LW_EXAMPLE_PPM_IMAGE_H
#define LW_EXAMPLE_PPM_IMAGE_H

/*
 * Minimal P6 PPM loader shared by the C examples. PPM keeps image decoding out
 * of the core library; real applications may use any decoder that can produce
 * packed BGR8 pixels.
 */

#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

#define LW_EXAMPLE_MAX_IMAGE_PIXELS UINT64_C(40000000)

typedef struct lw_example_ppm_image {
    uint8_t* pixels;
    uint64_t byte_count;
    uint32_t width;
    uint32_t height;
} lw_example_ppm_image;

#if defined(_WIN32)
static FILE* lw_example_open_read_utf8(const char* path_utf8) {
    int wide_count;
    wchar_t* wide_path;
    FILE* file = NULL;
    wide_count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path_utf8, -1, NULL, 0);
    if (wide_count <= 0) {
        return NULL;
    }
    wide_path = (wchar_t*)malloc((size_t)wide_count * sizeof(*wide_path));
    if (wide_path == NULL) {
        return NULL;
    }
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path_utf8, -1, wide_path, wide_count) >
            0 &&
        _wfopen_s(&file, wide_path, L"rb") != 0) {
        file = NULL;
    }
    free(wide_path);
    return file;
}
#else
static FILE* lw_example_open_read_utf8(const char* path_utf8) {
    return fopen(path_utf8, "rb");
}
#endif

static int lw_example_read_ppm_token(FILE* file, char* token, size_t capacity) {
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

static int lw_example_parse_positive_u32(const char* text, uint32_t* value) {
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

static void lw_example_ppm_image_free(lw_example_ppm_image* image) {
    if (image == NULL) {
        return;
    }
    free(image->pixels);
    memset(image, 0, sizeof(*image));
}

static int lw_example_ppm_image_load_bgr(const char* path_utf8, lw_example_ppm_image* image) {
    FILE* file = NULL;
    char token[64];
    uint32_t max_value;
    uint64_t pixel_count;
    uint64_t index;
    int result = 0;
    if (path_utf8 == NULL || image == NULL) {
        return 0;
    }
    memset(image, 0, sizeof(*image));
    file = lw_example_open_read_utf8(path_utf8);
    if (file == NULL || !lw_example_read_ppm_token(file, token, sizeof(token)) ||
        strcmp(token, "P6") != 0 || !lw_example_read_ppm_token(file, token, sizeof(token)) ||
        !lw_example_parse_positive_u32(token, &image->width) ||
        !lw_example_read_ppm_token(file, token, sizeof(token)) ||
        !lw_example_parse_positive_u32(token, &image->height) ||
        !lw_example_read_ppm_token(file, token, sizeof(token)) ||
        !lw_example_parse_positive_u32(token, &max_value) || max_value != 255u) {
        goto cleanup;
    }
    pixel_count = (uint64_t)image->width * image->height;
    if (pixel_count > LW_EXAMPLE_MAX_IMAGE_PIXELS || pixel_count > SIZE_MAX / 3u) {
        goto cleanup;
    }
    image->byte_count = pixel_count * 3u;
    image->pixels = (uint8_t*)malloc((size_t)image->byte_count);
    if (image->pixels == NULL ||
        fread(image->pixels, 1u, (size_t)image->byte_count, file) != (size_t)image->byte_count) {
        goto cleanup;
    }
    for (index = 0u; index < image->byte_count; index += 3u) {
        uint8_t red = image->pixels[(size_t)index];
        image->pixels[(size_t)index] = image->pixels[(size_t)index + 2u];
        image->pixels[(size_t)index + 2u] = red;
    }
    result = 1;
cleanup:
    if (file != NULL) {
        fclose(file);
    }
    if (!result) {
        lw_example_ppm_image_free(image);
    }
    return result;
}

#endif
