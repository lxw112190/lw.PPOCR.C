package com.lxw112190.ppocr

import android.content.Context
import android.graphics.Bitmap
import androidx.annotation.Keep
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.security.MessageDigest
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

public data class OcrOptions(
    val useCls: Boolean = false,
    val workerCount: Int = 2,
    val maxImagePixels: Long = 20_000_000L,
    val readingOrder: ReadingOrder = ReadingOrder.HORIZONTAL_LTR,
)

public enum class ReadingOrder {
    HORIZONTAL_LTR,
    VERTICAL_RTL,
    VERTICAL_LTR,
}

public data class OcrLine(
    val index: Int,
    val text: String,
    val box: FloatArray,
    val detScore: Float,
    val recScore: Float,
)

public data class OcrResult(
    val width: Int,
    val height: Int,
    val lines: List<OcrLine>,
    val elapsedMs: Long,
)

public class LwPpocrException(
    message: String,
    cause: Throwable? = null,
) : Exception(message, cause)

@Keep
public class LwPpocrEngine private constructor(
    private val handle: Long,
    private val options: OcrOptions,
) : AutoCloseable {
    private val lifecycleLock = ReentrantLock()
    private var closed = false

    public suspend fun recognize(bitmap: Bitmap): OcrResult = withContext(Dispatchers.Default) {
        lifecycleLock.withLock {
            check(!closed) { "OCR engine is closed" }
            require(bitmap.config == Bitmap.Config.ARGB_8888) {
                "Bitmap must use ARGB_8888; copy it before calling recognize"
            }
            val packet = NativeBridge.nativeRecognize(handle, bitmap)
                ?: throw LwPpocrException(NativeBridge.nativeLastError())
            val lines = packet.texts.indices.map { index ->
                OcrLine(
                    index = index,
                    text = packet.texts[index],
                    box = packet.boxes.copyOfRange(index * 8, index * 8 + 8),
                    detScore = packet.detectorScores[index],
                    recScore = packet.recognitionScores[index],
                )
            }
            OcrResult(packet.width, packet.height, lines, packet.elapsedMs)
        }
    }

    override fun close() {
        lifecycleLock.withLock {
            if (!closed) {
                closed = true
                NativeBridge.nativeDestroy(handle)
            }
        }
    }

    public companion object {
        private const val ASSET_ROOT = "lw-ppocr/models"
        private const val MODEL_VERSION = "ppocrv6-tiny-v1"

        public suspend fun create(context: Context, options: OcrOptions = OcrOptions()): LwPpocrEngine =
            withContext(Dispatchers.IO) {
                require(options.workerCount in 1..16) { "workerCount must be between 1 and 16" }
                require(options.maxImagePixels in 1..100_000_000) {
                    "maxImagePixels must be between 1 and 100000000"
                }
                val directory = File(context.noBackupFilesDir, MODEL_VERSION)
                installModels(context, directory)
                val handle = NativeBridge.nativeCreate(
                    File(directory, "det.lwm").absolutePath,
                    File(directory, "cls.lwm").absolutePath,
                    File(directory, "rec.lwm").absolutePath,
                    File(directory, "ppocr_keys.txt").absolutePath,
                    options.useCls,
                    options.workerCount,
                    options.maxImagePixels,
                )
                if (handle == 0L) {
                    throw LwPpocrException(NativeBridge.nativeLastError())
                }
                val readingOrder = when (options.readingOrder) {
                    ReadingOrder.HORIZONTAL_LTR -> 0
                    ReadingOrder.VERTICAL_RTL -> 1
                    ReadingOrder.VERTICAL_LTR -> 2
                }
                if (!NativeBridge.nativeSetReadingOrder(handle, readingOrder)) {
                    NativeBridge.nativeDestroy(handle)
                    throw LwPpocrException(NativeBridge.nativeLastError())
                }
                LwPpocrEngine(handle, options)
            }

        private fun installModels(context: Context, directory: File) {
            synchronized(modelInstallLock) {
                installModelsLocked(context, directory)
            }
        }

        private fun installModelsLocked(context: Context, directory: File) {
            val marker = File(directory, "installed.sha256")
            val names = listOf("det.lwm", "cls.lwm", "rec.lwm", "ppocr_keys.txt")
            if (marker.isFile && names.all { File(directory, it).isFile }) {
                val recorded = marker.readLines().associate { line ->
                    val parts = line.trim().split(Regex("\\s+"), limit = 2)
                    parts.first() to parts.getOrElse(1) { "" }
                }
                if (names.all { name -> recorded[sha256(File(directory, name))] == name }) return
            }
            val temporary = File(directory.parentFile, "${directory.name}.tmp-${System.nanoTime()}")
            temporary.deleteRecursively()
            check(temporary.mkdirs()) { "Unable to create model cache directory" }
            try {
                for (name in names) {
                    val target = File(temporary, name)
                    context.assets.open("$ASSET_ROOT/$name").use { input ->
                        FileOutputStream(target).use { output -> input.copyTo(output) }
                    }
                }
                FileOutputStream(File(temporary, "installed.sha256")).use { output ->
                    output.writer(Charsets.UTF_8).use { writer ->
                        for (name in names.sorted()) {
                            writer.append(sha256(File(temporary, name))).append("  ").append(name).append('\n')
                        }
                    }
                }
                directory.deleteRecursively()
                check(temporary.renameTo(directory)) { "Unable to commit model cache" }
            } catch (error: Throwable) {
                temporary.deleteRecursively()
                throw LwPpocrException("Unable to install OCR models", error)
            }
        }

        private fun sha256(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            FileInputStream(file).use { input ->
                val buffer = ByteArray(64 * 1024)
                while (true) {
                    val count = input.read(buffer)
                    if (count < 0) break
                    digest.update(buffer, 0, count)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }

        private val modelInstallLock = Any()
    }
}
