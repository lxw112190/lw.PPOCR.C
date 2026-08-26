#include "rec_internal.h"

/* UTF-8 dictionary loading and greedy CTC collapse for REC probabilities. */

#include "error_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

#define LW_REC_MAX_DICTIONARY_SIZE (UINT64_C(1024) * UINT64_C(1024))

struct lw_rec_dictionary {
    uint8_t* bytes;
    uint32_t byte_count;
    uint32_t entry_count;
    uint32_t* offsets;
    uint32_t* lengths;
};

#if defined(_WIN32)
static FILE* open_read_utf8(const char* path_utf8) {
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
static FILE* open_read_utf8(const char* path_utf8) {
    return fopen(path_utf8, "rb");
}
#endif

static int utf8_valid(const uint8_t* bytes, uint32_t count) {
    uint32_t index = 0u;
    while (index < count) {
        uint8_t first = bytes[index++];
        uint32_t continuation_count;
        uint32_t codepoint;
        uint32_t minimum;
        uint32_t continuation;
        if (first == 0u) {
            return 0;
        }
        if (first < 0x80u) {
            continue;
        }
        if (first >= 0xc2u && first <= 0xdfu) {
            continuation_count = 1u;
            codepoint = first & 0x1fu;
            minimum = 0x80u;
        } else if (first >= 0xe0u && first <= 0xefu) {
            continuation_count = 2u;
            codepoint = first & 0x0fu;
            minimum = 0x800u;
        } else if (first >= 0xf0u && first <= 0xf4u) {
            continuation_count = 3u;
            codepoint = first & 0x07u;
            minimum = 0x10000u;
        } else {
            return 0;
        }
        if (continuation_count > count - index) {
            return 0;
        }
        for (continuation = 0u; continuation < continuation_count; ++continuation) {
            uint8_t value = bytes[index++];
            if ((value & 0xc0u) != 0x80u) {
                return 0;
            }
            codepoint = (codepoint << 6) | (value & 0x3fu);
        }
        if (codepoint < minimum || codepoint > 0x10ffffu ||
            (codepoint >= 0xd800u && codepoint <= 0xdfffu)) {
            return 0;
        }
    }
    return 1;
}

static int count_entries(const uint8_t* bytes, uint32_t count, uint32_t start, uint32_t* result) {
    uint32_t entries = 0u;
    uint32_t cursor = start;
    int has_non_empty_label = 0;
    while (cursor < count) {
        uint32_t end = cursor;
        uint32_t label_end;
        while (end < count && bytes[end] != (uint8_t)'\n') {
            ++end;
        }
        label_end = end;
        if (label_end > cursor && bytes[label_end - 1u] == (uint8_t)'\r') {
            --label_end;
        }
        if (!utf8_valid(bytes + cursor, label_end - cursor) || entries == UINT32_MAX - 2u) {
            return 0;
        }
        if (label_end != cursor) {
            has_non_empty_label = 1;
        }
        ++entries;
        cursor = end < count ? end + 1u : count;
    }
    if (entries == 0u || !has_non_empty_label) {
        return 0;
    }
    *result = entries;
    return 1;
}

lw_status lw_rec_dictionary_load(const char* path_utf8, lw_rec_dictionary** out_dictionary,
                                 lw_error* error) {
    FILE* file;
    long length;
    lw_rec_dictionary* dictionary;
    size_t read_count;
    uint32_t start;
    uint32_t cursor;
    uint32_t entry_index = 0u;
    if (out_dictionary != NULL) {
        *out_dictionary = NULL;
    }
    if (path_utf8 == NULL || path_utf8[0] == '\0' || out_dictionary == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "dictionary path and output handle are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    file = open_read_utf8(path_utf8);
    if (file == NULL) {
        lw_set_error(error, LW_STATUS_IO_ERROR, "unable to open recognition dictionary");
        return LW_STATUS_IO_ERROR;
    }
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0 || (uint64_t)length > LW_REC_MAX_DICTIONARY_SIZE) {
        fclose(file);
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "recognition dictionary size is invalid");
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    dictionary = (lw_rec_dictionary*)calloc(1u, sizeof(*dictionary));
    if (dictionary == NULL) {
        fclose(file);
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate dictionary handle");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    dictionary->byte_count = (uint32_t)length;
    dictionary->bytes =
        (uint8_t*)malloc(dictionary->byte_count == 0u ? 1u : dictionary->byte_count);
    if (dictionary->bytes == NULL) {
        fclose(file);
        lw_rec_dictionary_free(dictionary);
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate dictionary bytes");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    read_count = fread(dictionary->bytes, 1u, dictionary->byte_count, file);
    fclose(file);
    if (read_count != dictionary->byte_count) {
        lw_rec_dictionary_free(dictionary);
        lw_set_error(error, LW_STATUS_IO_ERROR, "unable to read complete recognition dictionary");
        return LW_STATUS_IO_ERROR;
    }
    start = dictionary->byte_count >= 3u && dictionary->bytes[0] == 0xefu &&
                    dictionary->bytes[1] == 0xbbu && dictionary->bytes[2] == 0xbfu
                ? 3u
                : 0u;
    if (!count_entries(dictionary->bytes, dictionary->byte_count, start,
                       &dictionary->entry_count)) {
        lw_rec_dictionary_free(dictionary);
        lw_set_error(error, LW_STATUS_INVALID_FORMAT,
                     "recognition dictionary is empty or contains invalid UTF-8");
        return LW_STATUS_INVALID_FORMAT;
    }
    dictionary->offsets =
        (uint32_t*)malloc((size_t)dictionary->entry_count * sizeof(*dictionary->offsets));
    dictionary->lengths =
        (uint32_t*)malloc((size_t)dictionary->entry_count * sizeof(*dictionary->lengths));
    if (dictionary->offsets == NULL || dictionary->lengths == NULL) {
        lw_rec_dictionary_free(dictionary);
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate dictionary index");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    cursor = start;
    while (cursor < dictionary->byte_count) {
        uint32_t end = cursor;
        uint32_t label_end;
        while (end < dictionary->byte_count && dictionary->bytes[end] != (uint8_t)'\n') {
            ++end;
        }
        label_end = end;
        if (label_end > cursor && dictionary->bytes[label_end - 1u] == (uint8_t)'\r') {
            --label_end;
        }
        dictionary->offsets[entry_index] = cursor;
        dictionary->lengths[entry_index] = label_end - cursor;
        ++entry_index;
        cursor = end < dictionary->byte_count ? end + 1u : dictionary->byte_count;
    }
    *out_dictionary = dictionary;
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}

void lw_rec_dictionary_free(lw_rec_dictionary* dictionary) {
    if (dictionary == NULL) {
        return;
    }
    free(dictionary->lengths);
    free(dictionary->offsets);
    free(dictionary->bytes);
    free(dictionary);
}

uint32_t lw_rec_dictionary_class_count(const lw_rec_dictionary* dictionary) {
    return dictionary == NULL ? 0u : dictionary->entry_count + 2u;
}

uint32_t lw_rec_dictionary_max_label_byte_count(const lw_rec_dictionary* dictionary) {
    uint32_t maximum = 1u;
    uint32_t index;
    if (dictionary == NULL) {
        return 0u;
    }
    for (index = 0u; index < dictionary->entry_count; ++index) {
        if (dictionary->lengths[index] > maximum) {
            maximum = dictionary->lengths[index];
        }
    }
    return maximum;
}

static lw_status decode_pass(const lw_rec_dictionary* dictionary, const float* probabilities,
                             uint32_t time_steps, uint32_t class_count, char* text_utf8,
                             uint64_t* text_bytes, float* score, uint32_t* emitted_count) {
    uint64_t bytes = 0u;
    double score_sum = 0.0;
    uint32_t emitted = 0u;
    uint32_t previous = 0u;
    uint32_t step;
    /* Greedy CTC decoding chooses the best class at each time step. Class 0 is
     * blank; consecutive repeats collapse unless a blank separates them. */
    for (step = 0u; step < time_steps; ++step) {
        const float* row = probabilities + (size_t)((uint64_t)step * class_count);
        uint32_t best_index = 0u;
        float best_value = row[0];
        uint32_t class_index;
        if (!isfinite(best_value)) {
            return LW_STATUS_INVALID_ARGUMENT;
        }
        for (class_index = 1u; class_index < class_count; ++class_index) {
            float value = row[class_index];
            if (!isfinite(value)) {
                return LW_STATUS_INVALID_ARGUMENT;
            }
            if (value > best_value) {
                best_value = value;
                best_index = class_index;
            }
        }
        if (best_index != 0u && (step == 0u || best_index != previous)) {
            const uint8_t* label;
            uint32_t label_length;
            if (best_index == dictionary->entry_count + 1u) {
                static const uint8_t space = (uint8_t)' ';
                label = &space;
                label_length = 1u;
            } else {
                uint32_t dictionary_index = best_index - 1u;
                label = dictionary->bytes + dictionary->offsets[dictionary_index];
                label_length = dictionary->lengths[dictionary_index];
            }
            if (bytes > UINT64_MAX - label_length) {
                return LW_STATUS_OUT_OF_BOUNDS;
            }
            if (text_utf8 != NULL) {
                memcpy(text_utf8 + (size_t)bytes, label, label_length);
            }
            bytes += label_length;
            score_sum += best_value;
            ++emitted;
        }
        previous = best_index;
    }
    if (text_utf8 != NULL) {
        text_utf8[(size_t)bytes] = '\0';
    }
    *text_bytes = bytes;
    *score = emitted == 0u ? 0.0f : (float)(score_sum / emitted);
    *emitted_count = emitted;
    return LW_STATUS_OK;
}

lw_status lw_rec_ctc_decode_f32(const lw_rec_dictionary* dictionary, const float* probabilities,
                                uint64_t probability_element_count, uint32_t time_steps,
                                uint32_t class_count, char* text_utf8, uint64_t text_capacity,
                                uint64_t* required_capacity, float* score, uint32_t* emitted_count,
                                lw_error* error) {
    uint64_t expected_elements;
    uint64_t text_bytes;
    float decoded_score;
    uint32_t decoded_count;
    lw_status status;
    if (required_capacity != NULL) {
        *required_capacity = 0u;
    }
    if (score != NULL) {
        *score = 0.0f;
    }
    if (emitted_count != NULL) {
        *emitted_count = 0u;
    }
    if (dictionary == NULL || probabilities == NULL || required_capacity == NULL || score == NULL ||
        emitted_count == NULL || time_steps == 0u || class_count == 0u ||
        (text_utf8 == NULL && text_capacity != 0u)) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "dictionary, probabilities, and decode outputs are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    expected_elements = (uint64_t)time_steps * class_count;
    if (probability_element_count != expected_elements ||
        class_count != dictionary->entry_count + 2u ||
        expected_elements > SIZE_MAX / sizeof(float)) {
        lw_set_error(error, LW_STATUS_INVALID_SHAPE,
                     "CTC probability shape or dictionary class count is invalid");
        return LW_STATUS_INVALID_SHAPE;
    }
    /* First pass computes exact UTF-8 capacity without touching caller memory. */
    status = decode_pass(dictionary, probabilities, time_steps, class_count, NULL, &text_bytes,
                         &decoded_score, &decoded_count);
    if (status != LW_STATUS_OK) {
        lw_set_error(error, status, "CTC probabilities contain invalid values");
        return status;
    }
    if (text_bytes >= SIZE_MAX) {
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "CTC text capacity overflows");
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    *required_capacity = text_bytes + 1u;
    *score = decoded_score;
    *emitted_count = decoded_count;
    if (text_utf8 == NULL) {
        lw_set_error(error, LW_STATUS_OK, "");
        return LW_STATUS_OK;
    }
    if (text_capacity < *required_capacity) {
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "CTC text buffer is too small");
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    status = decode_pass(dictionary, probabilities, time_steps, class_count, text_utf8, &text_bytes,
                         &decoded_score, &decoded_count);
    if (status != LW_STATUS_OK) {
        lw_set_error(error, status, "CTC decoding failed");
        return status;
    }
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}
