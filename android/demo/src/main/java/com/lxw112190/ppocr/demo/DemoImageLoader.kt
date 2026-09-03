package com.lxw112190.ppocr.demo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.ceil
import kotlin.math.sqrt

object DemoImageLoader {
    private const val MAX_DECODE_PIXELS = 12_000_000L

    suspend fun load(
        context: Context,
        uri: Uri,
    ): Bitmap = withContext(Dispatchers.IO) {
        val resolver = context.contentResolver

        /*
         * Pass 1: read image dimensions only.
         *
         * BitmapFactory.decodeStream() intentionally returns null when
         * inJustDecodeBounds = true. Therefore we must check whether the
         * InputStream itself can be opened instead of using the decode result
         * as the success condition.
         */
        val bounds = BitmapFactory.Options().apply {
            inJustDecodeBounds = true
        }

        val boundsInput = resolver.openInputStream(uri)
            ?: error("无法打开所选图片")

        boundsInput.use { input ->
            BitmapFactory.decodeStream(
                input,
                null,
                bounds,
            )
        }

        require(
            bounds.outWidth > 0 &&
                bounds.outHeight > 0
        ) {
            "图片格式不受支持或图片已损坏"
        }

        /*
         * Pass 2: decode the actual Bitmap.
         *
         * Large phone photos are sampled before allocating the ARGB bitmap to
         * avoid unnecessarily large memory usage.
         */
        val sample = calculateSampleSize(
            bounds.outWidth,
            bounds.outHeight,
        )

        val decodeOptions = BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }

        val decodeInput = resolver.openInputStream(uri)
            ?: error("无法重新打开所选图片")

        val decoded = decodeInput.use { input ->
            BitmapFactory.decodeStream(
                input,
                null,
                decodeOptions,
            )
        } ?: error(
            "图片解码失败，格式可能不受当前设备支持"
        )

        var normalized = decoded

        try {
            /*
             * Read EXIF orientation independently. Photo Picker / document
             * providers return content:// URIs, so always access the selected
             * image through ContentResolver instead of assuming a filesystem
             * path.
             */
            val orientation = resolver.openInputStream(uri)?.use { input ->
                ExifInterface(input).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL,
                )
            } ?: ExifInterface.ORIENTATION_NORMAL

            normalized = applyOrientation(
                normalized,
                orientation,
            )

            if (normalized !== decoded && !decoded.isRecycled) {
                decoded.recycle()
            }

            normalized = ensureArgb8888(normalized)
            normalized = scaleToPixelLimit(normalized)

            normalized
        } catch (error: Throwable) {
            if (!normalized.isRecycled) {
                normalized.recycle()
            }

            if (!decoded.isRecycled) {
                decoded.recycle()
            }

            throw error
        }
    }

    private fun calculateSampleSize(
        width: Int,
        height: Int,
    ): Int {
        var sample = 1

        while (
            (width.toLong() / sample) *
                (height.toLong() / sample) >
            MAX_DECODE_PIXELS
        ) {
            sample *= 2
        }

        return sample
    }

    private fun applyOrientation(
        bitmap: Bitmap,
        orientation: Int,
    ): Bitmap {
        val matrix = Matrix()

        when (orientation) {
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> {
                matrix.setScale(-1f, 1f)
            }

            ExifInterface.ORIENTATION_ROTATE_180 -> {
                matrix.setRotate(180f)
            }

            ExifInterface.ORIENTATION_FLIP_VERTICAL -> {
                matrix.setScale(1f, -1f)
            }

            ExifInterface.ORIENTATION_TRANSPOSE -> {
                matrix.setRotate(90f)
                matrix.postScale(-1f, 1f)
            }

            ExifInterface.ORIENTATION_ROTATE_90 -> {
                matrix.setRotate(90f)
            }

            ExifInterface.ORIENTATION_TRANSVERSE -> {
                matrix.setRotate(-90f)
                matrix.postScale(-1f, 1f)
            }

            ExifInterface.ORIENTATION_ROTATE_270 -> {
                matrix.setRotate(-90f)
            }

            else -> {
                return bitmap
            }
        }

        return Bitmap.createBitmap(
            bitmap,
            0,
            0,
            bitmap.width,
            bitmap.height,
            matrix,
            true,
        )
    }

    private fun ensureArgb8888(
        bitmap: Bitmap,
    ): Bitmap {
        if (bitmap.config == Bitmap.Config.ARGB_8888) {
            return bitmap
        }

        val converted = bitmap.copy(
            Bitmap.Config.ARGB_8888,
            false,
        ) ?: error("无法转换图片像素格式")

        if (!bitmap.isRecycled) {
            bitmap.recycle()
        }

        return converted
    }

    private fun scaleToPixelLimit(
        bitmap: Bitmap,
    ): Bitmap {
        val pixels =
            bitmap.width.toLong() *
                bitmap.height.toLong()

        if (pixels <= MAX_DECODE_PIXELS) {
            return bitmap
        }

        val scale = sqrt(
            MAX_DECODE_PIXELS.toDouble() /
                pixels.toDouble()
        )

        val width = ceil(
            bitmap.width * scale
        ).toInt().coerceAtLeast(1)

        val height = ceil(
            bitmap.height * scale
        ).toInt().coerceAtLeast(1)

        val scaled = Bitmap.createScaledBitmap(
            bitmap,
            width,
            height,
            true,
        )

        if (scaled !== bitmap && !bitmap.isRecycled) {
            bitmap.recycle()
        }

        return scaled
    }
}