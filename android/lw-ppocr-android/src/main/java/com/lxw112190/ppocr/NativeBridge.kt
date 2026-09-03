package com.lxw112190.ppocr

internal class NativeBridge private constructor() {
    companion object {
        init {
            System.loadLibrary("lw_ppocr_android")
        }

        @JvmStatic
        external fun nativeCreate(
            detectorPath: String,
            classifierPath: String,
            recognizerPath: String,
            dictionaryPath: String,
            useCls: Boolean,
            workerCount: Int,
            maxPixels: Long,
        ): Long

        @JvmStatic
        external fun nativeDestroy(handle: Long)

        @JvmStatic
        external fun nativeSetReadingOrder(handle: Long, readingOrder: Int): Boolean

        @JvmStatic
        external fun nativeRecognize(handle: Long, bitmap: android.graphics.Bitmap): NativeOcrPacket?

        @JvmStatic
        external fun nativeLastError(): String
    }
}
