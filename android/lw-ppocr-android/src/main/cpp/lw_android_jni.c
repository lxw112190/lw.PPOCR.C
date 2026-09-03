#include <jni.h>
#include <android/bitmap.h>

#include "lw_infer.h"

#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct lw_android_engine {
    uint64_t token;
    lw_ocr* ocr;
    uint64_t max_pixels;
    uint32_t active_calls;
    int closing;
    pthread_mutex_t call_mutex;
    pthread_cond_t idle;
    struct lw_android_engine* next;
} lw_android_engine;

static pthread_mutex_t g_registry_mutex = PTHREAD_MUTEX_INITIALIZER;
static lw_android_engine* g_registry;
static uint64_t g_next_token = 1u;
static char g_last_error[LW_ERROR_MESSAGE_CAPACITY];
static jclass g_packet_class;
static jmethodID g_packet_ctor;

static void set_last_error(const char* message) {
    pthread_mutex_lock(&g_registry_mutex);
    if (message == NULL || message[0] == '\0') {
        message = "Android OCR native call failed";
    }
    (void)snprintf(g_last_error, sizeof(g_last_error), "%s", message);
    pthread_mutex_unlock(&g_registry_mutex);
}

static void set_status_error(lw_status status, const lw_error* error) {
    if (error != NULL && error->message[0] != '\0') {
        set_last_error(error->message);
    } else {
        set_last_error(lw_status_string(status));
    }
}

static lw_android_engine* acquire_engine(uint64_t token) {
    lw_android_engine* entry;
    pthread_mutex_lock(&g_registry_mutex);
    for (entry = g_registry; entry != NULL; entry = entry->next) {
        if (entry->token == token) {
            if (entry->closing != 0) {
                entry = NULL;
            } else {
                ++entry->active_calls;
            }
            break;
        }
    }
    pthread_mutex_unlock(&g_registry_mutex);
    if (entry != NULL) {
        pthread_mutex_lock(&entry->call_mutex);
    }
    return entry;
}

static void release_engine(lw_android_engine* entry) {
    if (entry == NULL) {
        return;
    }
    pthread_mutex_unlock(&entry->call_mutex);
    pthread_mutex_lock(&g_registry_mutex);
    if (entry->active_calls != 0u) {
        --entry->active_calls;
    }
    if (entry->closing != 0 && entry->active_calls == 0u) {
        pthread_cond_signal(&entry->idle);
    }
    pthread_mutex_unlock(&g_registry_mutex);
}

static uint64_t monotonic_millis(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0u;
    }
    return (uint64_t)value.tv_sec * 1000u + (uint64_t)value.tv_nsec / 1000000u;
}

static int checked_image_bytes(uint32_t width, uint32_t height, size_t* bytes) {
    uint64_t pixels = (uint64_t)width * (uint64_t)height;
    if (width == 0u || height == 0u || pixels > SIZE_MAX / 3u) {
        return 0;
    }
    *bytes = (size_t)(pixels * 3u);
    return 1;
}

static jlong native_create(JNIEnv* env, jclass unused, jstring det_path, jstring cls_path,
                           jstring rec_path, jstring dictionary_path, jboolean use_cls,
                           jint worker_count, jlong max_pixels) {
    const char* det_utf8 = NULL;
    const char* cls_utf8 = NULL;
    const char* rec_utf8 = NULL;
    const char* dictionary_utf8 = NULL;
    lw_ocr_options options;
    lw_ocr* ocr = NULL;
    lw_error error;
    lw_android_engine* entry = NULL;
    lw_status status;
    int mutex_initialized = 0;
    uint64_t pixels = max_pixels > 0 ? (uint64_t)max_pixels : 20000000u;
    (void)unused;
    if (det_path == NULL || cls_path == NULL || rec_path == NULL || dictionary_path == NULL) {
        set_last_error("OCR model paths must not be null");
        return 0;
    }
    det_utf8 = (*env)->GetStringUTFChars(env, det_path, NULL);
    cls_utf8 = (*env)->GetStringUTFChars(env, cls_path, NULL);
    rec_utf8 = (*env)->GetStringUTFChars(env, rec_path, NULL);
    dictionary_utf8 = (*env)->GetStringUTFChars(env, dictionary_path, NULL);
    if (det_utf8 == NULL || cls_utf8 == NULL || rec_utf8 == NULL || dictionary_utf8 == NULL) {
        set_last_error("Unable to read OCR model paths");
        goto cleanup_strings;
    }
    lw_ocr_options_init(&options);
    options.use_direction_classification = use_cls == JNI_TRUE ? 1u : 0u;
    options.worker_count = worker_count > 0 ? (uint32_t)worker_count : 2u;
    options.detector.max_image_pixels = pixels;
    options.classifier.max_image_pixels = pixels;
    options.recognizer.max_image_pixels = pixels;
    lw_error_init(&error);
    status = lw_ocr_create(det_utf8, cls_utf8, rec_utf8, dictionary_utf8, &options, &ocr, &error);
    if (status != LW_STATUS_OK) {
        set_status_error(status, &error);
        goto cleanup_strings;
    }
    entry = (lw_android_engine*)calloc(1u, sizeof(*entry));
    if (entry == NULL) {
        set_last_error("Unable to allocate Android OCR engine");
        lw_ocr_free(ocr);
        entry = NULL;
        goto cleanup_strings;
    }
    if (pthread_mutex_init(&entry->call_mutex, NULL) != 0) {
        set_last_error("Unable to initialize Android OCR engine lock");
        free(entry);
        lw_ocr_free(ocr);
        entry = NULL;
        goto cleanup_strings;
    }
    mutex_initialized = 1;
    if (pthread_cond_init(&entry->idle, NULL) != 0) {
        set_last_error("Unable to initialize Android OCR engine condition");
        if (mutex_initialized != 0) {
            (void)pthread_mutex_destroy(&entry->call_mutex);
        }
        free(entry);
        lw_ocr_free(ocr);
        entry = NULL;
        goto cleanup_strings;
    }
    entry->ocr = ocr;
    entry->max_pixels = pixels;
    pthread_mutex_lock(&g_registry_mutex);
    entry->token = g_next_token++;
    if (entry->token == 0u) {
        entry->token = g_next_token++;
    }
    entry->next = g_registry;
    g_registry = entry;
    pthread_mutex_unlock(&g_registry_mutex);
cleanup_strings:
    if (det_utf8 != NULL) {
        (*env)->ReleaseStringUTFChars(env, det_path, det_utf8);
    }
    if (cls_utf8 != NULL) {
        (*env)->ReleaseStringUTFChars(env, cls_path, cls_utf8);
    }
    if (rec_utf8 != NULL) {
        (*env)->ReleaseStringUTFChars(env, rec_path, rec_utf8);
    }
    if (dictionary_utf8 != NULL) {
        (*env)->ReleaseStringUTFChars(env, dictionary_path, dictionary_utf8);
    }
    return entry == NULL ? 0 : (jlong)entry->token;
}

static void native_destroy(JNIEnv* env, jclass unused, jlong handle) {
    lw_android_engine* entry;
    lw_android_engine** cursor;
    (void)env;
    (void)unused;
    if (handle == 0) {
        return;
    }
    pthread_mutex_lock(&g_registry_mutex);
    entry = NULL;
    for (cursor = &g_registry; *cursor != NULL; cursor = &(*cursor)->next) {
        if ((*cursor)->token == (uint64_t)handle) {
            entry = *cursor;
            *cursor = entry->next;
            entry->closing = 1;
            while (entry->active_calls != 0u) {
                (void)pthread_cond_wait(&entry->idle, &g_registry_mutex);
            }
            break;
        }
    }
    pthread_mutex_unlock(&g_registry_mutex);
    if (entry == NULL) {
        return;
    }
    lw_ocr_free(entry->ocr);
    (void)pthread_cond_destroy(&entry->idle);
    (void)pthread_mutex_destroy(&entry->call_mutex);
    free(entry);
}

static jboolean native_set_reading_order(JNIEnv* env, jclass unused, jlong handle,
                                         jint reading_order) {
    lw_android_engine* entry = acquire_engine((uint64_t)handle);
    lw_error error;
    lw_status status;
    (void)env;
    (void)unused;
    if (entry == NULL) {
        set_last_error("OCR engine is closed or invalid");
        return JNI_FALSE;
    }
    lw_error_init(&error);
    status = lw_ocr_set_reading_order(entry->ocr, (uint32_t)reading_order, &error);
    if (status != LW_STATUS_OK) {
        set_status_error(status, &error);
    }
    release_engine(entry);
    return status == LW_STATUS_OK ? JNI_TRUE : JNI_FALSE;
}

static jobject native_recognize(JNIEnv* env, jclass unused, jlong handle, jobject bitmap) {
    lw_android_engine* entry = acquire_engine((uint64_t)handle);
    AndroidBitmapInfo bitmap_info;
    uint8_t* pixels = NULL;
    uint8_t* bgr = NULL;
    lw_ocr_info info;
    lw_ocr_line* lines = NULL;
    char* text = NULL;
    lw_ocr_result result;
    lw_error error;
    lw_status status;
    size_t image_bytes;
    uint32_t line_index;
    uint64_t started;
    uint64_t elapsed;
    jobject packet = NULL;
    jfloatArray boxes_array = NULL;
    jfloatArray det_scores_array = NULL;
    jfloatArray rec_scores_array = NULL;
    jobjectArray text_array = NULL;
    jclass string_class = NULL;
    (void)unused;
    if (entry == NULL) {
        set_last_error("OCR engine is closed or invalid");
        return NULL;
    }
    if (bitmap == NULL || AndroidBitmap_getInfo(env, bitmap, &bitmap_info) != ANDROID_BITMAP_RESULT_SUCCESS ||
        bitmap_info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 ||
        !checked_image_bytes(bitmap_info.width, bitmap_info.height, &image_bytes) ||
        (uint64_t)bitmap_info.width * bitmap_info.height > entry->max_pixels) {
        set_last_error("Bitmap must be RGBA_8888 and within the configured pixel limit");
        goto cleanup;
    }
    bgr = (uint8_t*)malloc(image_bytes);
    if (bgr == NULL) {
        set_last_error("Unable to allocate BGR input buffer");
        goto cleanup;
    }
    if (AndroidBitmap_lockPixels(env, bitmap, (void**)&pixels) != ANDROID_BITMAP_RESULT_SUCCESS ||
        pixels == NULL) {
        set_last_error("Unable to lock bitmap pixels");
        goto cleanup;
    }
    {
        uint32_t y;
        for (y = 0u; y < bitmap_info.height; ++y) {
            const uint8_t* source_row = pixels + (size_t)y * bitmap_info.stride;
            uint8_t* destination_row = bgr + (size_t)y * bitmap_info.width * 3u;
            uint32_t x;
            for (x = 0u; x < bitmap_info.width; ++x) {
                destination_row[x * 3u + 0u] = source_row[x * 4u + 2u];
                destination_row[x * 3u + 1u] = source_row[x * 4u + 1u];
                destination_row[x * 3u + 2u] = source_row[x * 4u + 0u];
            }
        }
    }
    (void)AndroidBitmap_unlockPixels(env, bitmap);
    pixels = NULL;
    lw_ocr_info_init(&info);
    lw_error_init(&error);
    status = lw_ocr_get_info(entry->ocr, &info);
    if (status != LW_STATUS_OK) {
        set_status_error(status, &error);
        goto cleanup;
    }
    if (info.max_line_capacity == 0u || info.max_text_capacity == 0u ||
        info.max_text_capacity > 64u * 1024u * 1024u) {
        set_last_error("OCR returned invalid output capacities");
        goto cleanup;
    }
    lines = (lw_ocr_line*)calloc(info.max_line_capacity, sizeof(*lines));
    text = (char*)calloc((size_t)info.max_text_capacity, sizeof(*text));
    if (lines == NULL || text == NULL) {
        set_last_error("Unable to allocate OCR result buffers");
        goto cleanup;
    }
    lw_ocr_result_init(&result);
    started = monotonic_millis();
    status = lw_ocr_run_bgr_u8(entry->ocr, bgr, (uint64_t)image_bytes, bitmap_info.width,
                               bitmap_info.height, bitmap_info.width * 3u, lines,
                               info.max_line_capacity, text, info.max_text_capacity, &result,
                               &error);
    elapsed = monotonic_millis() - started;
    if (status != LW_STATUS_OK || result.line_count > info.max_line_capacity ||
        result.required_text_capacity > info.max_text_capacity) {
        set_status_error(status, &error);
        goto cleanup;
    }
    boxes_array = (*env)->NewFloatArray(env, (jsize)(result.line_count * 8u));
    det_scores_array = (*env)->NewFloatArray(env, (jsize)result.line_count);
    rec_scores_array = (*env)->NewFloatArray(env, (jsize)result.line_count);
    string_class = (*env)->FindClass(env, "java/lang/String");
    if (boxes_array == NULL || det_scores_array == NULL || rec_scores_array == NULL ||
        string_class == NULL) {
        goto cleanup;
    }
    text_array = (*env)->NewObjectArray(env, (jsize)result.line_count, string_class, NULL);
    if (text_array == NULL) {
        goto cleanup;
    }
    for (line_index = 0u; line_index < result.line_count; ++line_index) {
        const lw_ocr_line* line = &lines[line_index];
        jfloat box_values[8];
        jstring line_text;
        uint32_t box_index;
        if (line->text_offset > result.required_text_capacity ||
            line->text_length > result.required_text_capacity - line->text_offset) {
            set_last_error("OCR returned an invalid text range");
            goto cleanup;
        }
        box_values[0] = line->box.x1;
        box_values[1] = line->box.y1;
        box_values[2] = line->box.x2;
        box_values[3] = line->box.y2;
        box_values[4] = line->box.x3;
        box_values[5] = line->box.y3;
        box_values[6] = line->box.x4;
        box_values[7] = line->box.y4;
        box_index = line_index * 8u;
        (*env)->SetFloatArrayRegion(env, boxes_array, (jsize)box_index, 8, box_values);
        (*env)->SetFloatArrayRegion(env, det_scores_array, (jsize)line_index, 1,
                                    &line->box.score);
        (*env)->SetFloatArrayRegion(env, rec_scores_array, (jsize)line_index, 1,
                                    &line->recognition_score);
        line_text = (*env)->NewStringUTF(env, text + line->text_offset);
        if (line_text == NULL) {
            goto cleanup;
        }
        (*env)->SetObjectArrayElement(env, text_array, (jsize)line_index, line_text);
        (*env)->DeleteLocalRef(env, line_text);
    }
    packet = (*env)->NewObject(env, g_packet_class, g_packet_ctor, (jint)bitmap_info.width,
                               (jint)bitmap_info.height, boxes_array, det_scores_array,
                               rec_scores_array, text_array, (jlong)elapsed);
cleanup:
    if (pixels != NULL) {
        (void)AndroidBitmap_unlockPixels(env, bitmap);
    }
    (*env)->DeleteLocalRef(env, boxes_array);
    (*env)->DeleteLocalRef(env, det_scores_array);
    (*env)->DeleteLocalRef(env, rec_scores_array);
    (*env)->DeleteLocalRef(env, text_array);
    (*env)->DeleteLocalRef(env, string_class);
    free(lines);
    free(text);
    free(bgr);
    release_engine(entry);
    return packet;
}

static jstring native_last_error(JNIEnv* env, jclass unused) {
    char message[LW_ERROR_MESSAGE_CAPACITY];
    (void)unused;
    pthread_mutex_lock(&g_registry_mutex);
    (void)snprintf(message, sizeof(message), "%s", g_last_error);
    pthread_mutex_unlock(&g_registry_mutex);
    return (*env)->NewStringUTF(env, message[0] == '\0' ? "Unknown OCR error" : message);
}

static const JNINativeMethod g_methods[] = {
    {"nativeCreate", "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;ZIJ)J",
     (void*)native_create},
    {"nativeDestroy", "(J)V", (void*)native_destroy},
    {"nativeSetReadingOrder", "(JI)Z", (void*)native_set_reading_order},
    {"nativeRecognize", "(JLandroid/graphics/Bitmap;)Lcom/lxw112190/ppocr/NativeOcrPacket;",
     (void*)native_recognize},
    {"nativeLastError", "()Ljava/lang/String;", (void*)native_last_error},
};

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    JNIEnv* env = NULL;
    jclass packet_local;
    jclass bridge_local;
    (void)reserved;
    if ((*vm)->GetEnv(vm, (void**)&env, JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR;
    }
    packet_local = (*env)->FindClass(env, "com/lxw112190/ppocr/NativeOcrPacket");
    bridge_local = (*env)->FindClass(env, "com/lxw112190/ppocr/NativeBridge");
    if (packet_local == NULL || bridge_local == NULL) {
        return JNI_ERR;
    }
    g_packet_class = (jclass)(*env)->NewGlobalRef(env, packet_local);
    if (g_packet_class == NULL) {
        return JNI_ERR;
    }
    g_packet_ctor = (*env)->GetMethodID(env, g_packet_class, "<init>",
                                        "(II[F[F[F[Ljava/lang/String;J)V");
    if (g_packet_ctor == NULL ||
        (*env)->RegisterNatives(env, bridge_local, g_methods,
                                (jint)(sizeof(g_methods) / sizeof(g_methods[0]))) != JNI_OK) {
        return JNI_ERR;
    }
    return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM* vm, void* reserved) {
    JNIEnv* env = NULL;
    (void)reserved;
    if (g_packet_class != NULL && (*vm)->GetEnv(vm, (void**)&env, JNI_VERSION_1_6) == JNI_OK) {
        (*env)->DeleteGlobalRef(env, g_packet_class);
    }
    g_packet_class = NULL;
    g_packet_ctor = NULL;
}
