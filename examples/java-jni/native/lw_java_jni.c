#include <jni.h>

#include "lw_infer.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct lw_java_engine {
    lw_ocr* ocr;
    lw_ocr_info info;
    lw_ocr_line* lines;
    char* text;
} lw_java_engine;

static int allocation_fits(uint64_t count, size_t element_size) {
    return element_size != 0u && count <= (uint64_t)(SIZE_MAX / element_size);
}

static void throw_runtime(JNIEnv* env, const char* message) {
    jclass exception_class;
    if ((*env)->ExceptionCheck(env)) {
        return;
    }
    exception_class = (*env)->FindClass(env, "java/lang/RuntimeException");
    if (exception_class != NULL) {
        (*env)->ThrowNew(env, exception_class, message == NULL ? "JNI OCR failure" : message);
        (*env)->DeleteLocalRef(env, exception_class);
    }
}

static void throw_status(JNIEnv* env, lw_status status, const lw_error* error) {
    const char* message = NULL;
    if (error != NULL && error->message[0] != '\0') {
        message = error->message;
    }
    if (message == NULL) {
        message = lw_status_string(status);
    }
    throw_runtime(env, message);
}

static int utf8_append(char* output, size_t capacity, size_t* offset, uint32_t codepoint) {
    size_t needed;
    if (codepoint <= 0x7fu) {
        needed = 1u;
    } else if (codepoint <= 0x7ffu) {
        needed = 2u;
    } else if (codepoint <= 0xffffu) {
        needed = 3u;
    } else {
        needed = 4u;
    }
    if (needed > capacity || *offset > capacity - needed) {
        return 0;
    }
    if (needed == 1u) {
        output[(*offset)++] = (char)codepoint;
    } else if (needed == 2u) {
        output[(*offset)++] = (char)(0xc0u | (codepoint >> 6));
        output[(*offset)++] = (char)(0x80u | (codepoint & 0x3fu));
    } else if (needed == 3u) {
        output[(*offset)++] = (char)(0xe0u | (codepoint >> 12));
        output[(*offset)++] = (char)(0x80u | ((codepoint >> 6) & 0x3fu));
        output[(*offset)++] = (char)(0x80u | (codepoint & 0x3fu));
    } else {
        output[(*offset)++] = (char)(0xf0u | (codepoint >> 18));
        output[(*offset)++] = (char)(0x80u | ((codepoint >> 12) & 0x3fu));
        output[(*offset)++] = (char)(0x80u | ((codepoint >> 6) & 0x3fu));
        output[(*offset)++] = (char)(0x80u | (codepoint & 0x3fu));
    }
    return 1;
}

/* Convert Java UTF-16 to standard UTF-8. GetStringUTFChars is intentionally
 * not used because it returns Modified UTF-8, while the C ABI requires UTF-8. */
static char* java_string_to_utf8(JNIEnv* env, jstring value) {
    const jchar* chars;
    jsize length;
    size_t bytes = 0u;
    size_t index;
    char* output;
    size_t offset = 0u;

    if (value == NULL) {
        throw_runtime(env, "model path must not be null");
        return NULL;
    }
    length = (*env)->GetStringLength(env, value);
    chars = (*env)->GetStringChars(env, value, NULL);
    if (chars == NULL) {
        return NULL;
    }
    for (index = 0u; index < (size_t)length; ++index) {
        uint32_t codepoint = chars[index];
        size_t needed;
        if (codepoint >= 0xd800u && codepoint <= 0xdbffu) {
            uint32_t low;
            if (index + 1u >= (size_t)length) {
                (*env)->ReleaseStringChars(env, value, chars);
                throw_runtime(env, "unpaired UTF-16 high surrogate in model path");
                return NULL;
            }
            low = chars[++index];
            if (low < 0xdc00u || low > 0xdfffu) {
                (*env)->ReleaseStringChars(env, value, chars);
                throw_runtime(env, "invalid UTF-16 surrogate pair in model path");
                return NULL;
            }
            codepoint = 0x10000u + ((codepoint - 0xd800u) << 10) + (low - 0xdc00u);
        } else if (codepoint >= 0xdc00u && codepoint <= 0xdfffu) {
            (*env)->ReleaseStringChars(env, value, chars);
            throw_runtime(env, "unpaired UTF-16 low surrogate in model path");
            return NULL;
        }
        needed = codepoint <= 0x7fu ? 1u : codepoint <= 0x7ffu ? 2u :
                 codepoint <= 0xffffu ? 3u : 4u;
        if (bytes > SIZE_MAX - needed) {
            (*env)->ReleaseStringChars(env, value, chars);
            throw_runtime(env, "model path is too long");
            return NULL;
        }
        bytes += needed;
    }
    if (bytes == SIZE_MAX) {
        (*env)->ReleaseStringChars(env, value, chars);
        throw_runtime(env, "model path is too long");
        return NULL;
    }
    output = (char*)malloc(bytes + 1u);
    if (output == NULL) {
        (*env)->ReleaseStringChars(env, value, chars);
        throw_runtime(env, "unable to allocate model path");
        return NULL;
    }
    for (index = 0u; index < (size_t)length; ++index) {
        uint32_t codepoint = chars[index];
        if (codepoint >= 0xd800u && codepoint <= 0xdbffu) {
            uint32_t low = chars[++index];
            codepoint = 0x10000u + ((codepoint - 0xd800u) << 10) + (low - 0xdc00u);
        }
        if (!utf8_append(output, bytes, &offset, codepoint)) {
            free(output);
            (*env)->ReleaseStringChars(env, value, chars);
            throw_runtime(env, "unable to encode model path");
            return NULL;
        }
    }
    output[offset] = '\0';
    (*env)->ReleaseStringChars(env, value, chars);
    return output;
}

static int continuation_byte(uint8_t value) {
    return value >= 0x80u && value <= 0xbfu;
}

/* Decode standard UTF-8 and create a Java UTF-16 string, including surrogate
 * pairs for code points above U+FFFF. Invalid model output is rejected rather
 * than silently replaced. */
static jstring utf8_to_jstring(JNIEnv* env, const char* input, size_t length) {
    jchar* units;
    size_t unit_capacity;
    size_t index = 0u;
    size_t unit_count = 0u;
    jstring result;

    if (length > (SIZE_MAX / sizeof(*units)) / 2u) {
        throw_runtime(env, "OCR text is too long");
        return NULL;
    }
    unit_capacity = length == 0u ? 1u : length * 2u;
    units = (jchar*)malloc(unit_capacity * sizeof(*units));
    if (units == NULL) {
        throw_runtime(env, "unable to allocate OCR text");
        return NULL;
    }
    while (index < length) {
        uint8_t first = (uint8_t)input[index++];
        uint32_t codepoint;
        size_t count;
        if (first <= 0x7fu) {
            codepoint = first;
            count = 0u;
        } else if (first >= 0xc2u && first <= 0xdfu) {
            count = 1u;
            if (index + count > length || !continuation_byte((uint8_t)input[index]))
                goto invalid;
            codepoint = first & 0x1fu;
        } else if (first >= 0xe0u && first <= 0xefu) {
            count = 2u;
            if (index + count > length || !continuation_byte((uint8_t)input[index]) ||
                !continuation_byte((uint8_t)input[index + 1u]))
                goto invalid;
            if ((first == 0xe0u && (uint8_t)input[index] < 0xa0u) ||
                (first == 0xedu && (uint8_t)input[index] >= 0xa0u))
                goto invalid;
            codepoint = first & 0x0fu;
        } else if (first >= 0xf0u && first <= 0xf4u) {
            count = 3u;
            if (index + count > length || !continuation_byte((uint8_t)input[index]) ||
                !continuation_byte((uint8_t)input[index + 1u]) ||
                !continuation_byte((uint8_t)input[index + 2u]))
                goto invalid;
            if ((first == 0xf0u && (uint8_t)input[index] < 0x90u) ||
                (first == 0xf4u && (uint8_t)input[index] > 0x8fu))
                goto invalid;
            codepoint = first & 0x07u;
        } else {
            goto invalid;
        }
        while (count != 0u) {
            codepoint = (codepoint << 6) | ((uint8_t)input[index++] & 0x3fu);
            --count;
        }
        if (codepoint <= 0xffffu) {
            if (codepoint >= 0xd800u && codepoint <= 0xdfffu)
                goto invalid;
            units[unit_count++] = (jchar)codepoint;
        } else {
            uint32_t value = codepoint - 0x10000u;
            units[unit_count++] = (jchar)(0xd800u | (value >> 10));
            units[unit_count++] = (jchar)(0xdc00u | (value & 0x3ffu));
        }
    }
    if (unit_count > (size_t)INT32_MAX) {
        free(units);
        throw_runtime(env, "OCR text exceeds Java string length");
        return NULL;
    }
    result = (*env)->NewString(env, units, (jsize)unit_count);
    free(units);
    return result;

invalid:
    free(units);
    throw_runtime(env, "OCR returned invalid UTF-8 text");
    return NULL;
}

static void destroy_engine(lw_java_engine* engine) {
    if (engine == NULL)
        return;
    lw_ocr_free(engine->ocr);
    free(engine->lines);
    free(engine->text);
    free(engine);
}

JNIEXPORT jlong JNICALL Java_NativeOcr_nativeCreate(
        JNIEnv* env, jclass clazz, jstring detector, jstring classifier, jstring recognizer,
        jstring dictionary, jboolean use_cls, jint worker_count) {
    char* detector_path = NULL;
    char* classifier_path = NULL;
    char* recognizer_path = NULL;
    char* dictionary_path = NULL;
    lw_ocr_options options;
    lw_error error;
    lw_ocr* ocr = NULL;
    lw_java_engine* engine = NULL;
    lw_status status;
    (void)clazz;

    if (worker_count < 0) {
        throw_runtime(env, "workerCount must be zero or positive");
        return 0;
    }
    detector_path = java_string_to_utf8(env, detector);
    if (detector_path == NULL)
        goto fail;
    if (classifier != NULL) {
        classifier_path = java_string_to_utf8(env, classifier);
        if (classifier_path == NULL)
            goto fail;
    }
    recognizer_path = java_string_to_utf8(env, recognizer);
    if (recognizer_path == NULL)
        goto fail;
    dictionary_path = java_string_to_utf8(env, dictionary);
    if (dictionary_path == NULL)
        goto fail;

    lw_ocr_options_init(&options);
    options.use_direction_classification = use_cls == JNI_TRUE ? 1u : 0u;
    options.worker_count = (uint32_t)worker_count;
    options.recognizer.target_width = 960u;
    lw_error_init(&error);
    status = lw_ocr_create(detector_path, classifier_path, recognizer_path, dictionary_path,
                           &options, &ocr, &error);
    if (status != LW_STATUS_OK) {
        throw_status(env, status, &error);
        goto fail;
    }
    engine = (lw_java_engine*)calloc(1u, sizeof(*engine));
    if (engine == NULL) {
        throw_runtime(env, "unable to allocate Java OCR engine");
        goto fail;
    }
    lw_ocr_info_init(&engine->info);
    status = lw_ocr_get_info(ocr, &engine->info);
    if (status != LW_STATUS_OK || engine->info.max_line_capacity == 0u ||
        engine->info.max_text_capacity == 0u ||
        !allocation_fits(engine->info.max_line_capacity, sizeof(*engine->lines)) ||
        !allocation_fits(engine->info.max_text_capacity, sizeof(*engine->text))) {
        throw_status(env, status == LW_STATUS_OK ? LW_STATUS_OUT_OF_BOUNDS : status, &error);
        goto fail;
    }
    engine->lines = (lw_ocr_line*)calloc(engine->info.max_line_capacity, sizeof(*engine->lines));
    engine->text = (char*)malloc((size_t)engine->info.max_text_capacity);
    if (engine->lines == NULL || engine->text == NULL) {
        throw_runtime(env, "unable to allocate Java OCR output buffers");
        goto fail;
    }
    engine->ocr = ocr;
    ocr = NULL;
    free(detector_path);
    free(classifier_path);
    free(recognizer_path);
    free(dictionary_path);
    return (jlong)(intptr_t)engine;

fail:
    destroy_engine(engine);
    lw_ocr_free(ocr);
    free(detector_path);
    free(classifier_path);
    free(recognizer_path);
    free(dictionary_path);
    return 0;
}

JNIEXPORT jobjectArray JNICALL Java_NativeOcr_nativeRecognize(
        JNIEnv* env, jclass clazz, jlong handle, jbyteArray bgr, jint width, jint height,
        jint stride) {
    lw_java_engine* engine = (lw_java_engine*)(intptr_t)handle;
    jsize array_length;
    jbyte* pixels;
    uint64_t minimum_stride;
    uint64_t required_bytes;
    lw_ocr_result result;
    lw_error error;
    lw_status status;
    jclass string_class;
    jobjectArray output;
    uint32_t index;
    (void)clazz;

    if (engine == NULL || bgr == NULL || width <= 0 || height <= 0 || stride <= 0) {
        throw_runtime(env, "invalid OCR image or closed engine");
        return NULL;
    }
    minimum_stride = (uint64_t)(uint32_t)width * 3u;
    if ((uint64_t)(uint32_t)stride < minimum_stride) {
        throw_runtime(env, "OCR stride is smaller than width * 3");
        return NULL;
    }
    required_bytes = (uint64_t)(uint32_t)stride * (uint64_t)(uint32_t)height;
    array_length = (*env)->GetArrayLength(env, bgr);
    if (required_bytes > (uint64_t)(uint32_t)array_length ||
        required_bytes > (uint64_t)SIZE_MAX) {
        throw_runtime(env, "OCR byte array is smaller than the requested image");
        return NULL;
    }
    pixels = (*env)->GetByteArrayElements(env, bgr, NULL);
    if (pixels == NULL)
        return NULL;
    lw_ocr_result_init(&result);
    lw_error_init(&error);
    status = lw_ocr_run_bgr_u8(engine->ocr, (const uint8_t*)pixels, required_bytes,
                               (uint32_t)width, (uint32_t)height, (uint32_t)stride, engine->lines,
                               engine->info.max_line_capacity, engine->text,
                               engine->info.max_text_capacity, &result, &error);
    (*env)->ReleaseByteArrayElements(env, bgr, pixels, JNI_ABORT);
    if (status != LW_STATUS_OK) {
        throw_status(env, status, &error);
        return NULL;
    }
    if (result.line_count > engine->info.max_line_capacity ||
        result.required_text_capacity > engine->info.max_text_capacity ||
        result.line_count > (uint32_t)INT32_MAX) {
        throw_runtime(env, "OCR returned invalid output capacities");
        return NULL;
    }
    string_class = (*env)->FindClass(env, "java/lang/String");
    if (string_class == NULL)
        return NULL;
    output = (*env)->NewObjectArray(env, (jsize)result.line_count, string_class, NULL);
    (*env)->DeleteLocalRef(env, string_class);
    if (output == NULL)
        return NULL;
    for (index = 0u; index < result.line_count; ++index) {
        const lw_ocr_line* line = &engine->lines[index];
        jstring text;
        uint64_t remaining;
        if (line->text_offset >= result.required_text_capacity) {
            (*env)->DeleteLocalRef(env, output);
            throw_runtime(env, "OCR text offset is out of bounds");
            return NULL;
        }
        remaining = result.required_text_capacity - line->text_offset;
        if (line->text_length >= remaining ||
            engine->text[line->text_offset + line->text_length] != '\0') {
            (*env)->DeleteLocalRef(env, output);
            throw_runtime(env, "OCR text length is out of bounds");
            return NULL;
        }
        text = utf8_to_jstring(env, engine->text + (size_t)line->text_offset,
                               (size_t)line->text_length);
        if (text == NULL) {
            (*env)->DeleteLocalRef(env, output);
            return NULL;
        }
        (*env)->SetObjectArrayElement(env, output, (jsize)index, text);
        (*env)->DeleteLocalRef(env, text);
        if ((*env)->ExceptionCheck(env)) {
            (*env)->DeleteLocalRef(env, output);
            return NULL;
        }
    }
    return output;
}

JNIEXPORT void JNICALL Java_NativeOcr_nativeSetReadingOrder(
        JNIEnv* env, jclass clazz, jlong handle, jint reading_order) {
    lw_java_engine* engine = (lw_java_engine*)(intptr_t)handle;
    lw_error error;
    lw_status status;
    (void)clazz;
    if (engine == NULL || reading_order < 0 || reading_order > 2) {
        throw_runtime(env, "invalid OCR reading order or closed engine");
        return;
    }
    lw_error_init(&error);
    status = lw_ocr_set_reading_order(engine->ocr, (uint32_t)reading_order, &error);
    if (status != LW_STATUS_OK)
        throw_status(env, status, &error);
}

JNIEXPORT void JNICALL Java_NativeOcr_nativeDestroy(
        JNIEnv* env, jclass clazz, jlong handle) {
    (void)env;
    (void)clazz;
    destroy_engine((lw_java_engine*)(intptr_t)handle);
}
