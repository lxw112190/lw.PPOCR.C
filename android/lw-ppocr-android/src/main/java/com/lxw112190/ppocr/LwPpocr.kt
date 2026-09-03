package com.lxw112190.ppocr

import android.content.Context
import android.graphics.Bitmap
import androidx.annotation.Keep
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
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

private data class ModelFile(
    val bytes: Long,
    val sha256: String,
)

private data class ModelManifest(
    val assetSetId: String,
    val files: Map<String, ModelFile>,
)

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
        private const val CACHE_ROOT = "lw-ppocr/models"
        private const val MODEL_ID = "ppocrv6-tiny"
        private const val MODEL_NAME = "PP-OCRv6 tiny"
        private const val RUNTIME_FORMAT = "LWM 0.1"
        private const val MANIFEST_NAME = "manifest.json"
        private const val MARKER_NAME = "installed.asset-set-id"
        private val MODEL_FILES = listOf("det.lwm", "cls.lwm", "rec.lwm", "ppocr_keys.txt")
        private val ASSET_ID_PATTERN = Regex("[0-9a-f]{64}")

        public suspend fun create(context: Context, options: OcrOptions = OcrOptions()): LwPpocrEngine =
            withContext(Dispatchers.IO) {
                require(options.workerCount in 1..16) { "workerCount must be between 1 and 16" }
                require(options.maxImagePixels in 1..100_000_000) {
                    "maxImagePixels must be between 1 and 100000000"
                }
                val manifest = readAssetManifest(context)
                val directory = installModels(context, manifest)
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
                pruneOldCaches(directory)
                LwPpocrEngine(handle, options)
            }

        private fun readAssetManifest(context: Context): ModelManifest {
            return try {
                val text = context.assets.open("$ASSET_ROOT/$MANIFEST_NAME").use { input ->
                    input.reader(Charsets.UTF_8).use { reader -> reader.readText() }
                }
                parseManifest(text, "asset manifest")
            } catch (error: Throwable) {
                if (error is LwPpocrException) throw error
                throw LwPpocrException("Unable to read OCR model manifest", error)
            }
        }

        private fun parseManifest(text: String, source: String): ModelManifest {
            val root = JSONObject(text)
            require(root.optInt("schema_version", -1) == 1) {
                "$source has unsupported schema_version"
            }
            require(root.optString("model", "") == MODEL_NAME) {
                "$source has unsupported model"
            }
            require(root.optString("runtime_format", "") == RUNTIME_FORMAT) {
                "$source has unsupported runtime format"
            }
            val assetSetId = root.optString("asset_set_id", "")
            require(ASSET_ID_PATTERN.matches(assetSetId)) {
                "$source has invalid asset_set_id"
            }
            val filesObject = root.optJSONObject("files")
                ?: error("$source has no files object")
            val keys = mutableSetOf<String>()
            val iterator = filesObject.keys()
            while (iterator.hasNext()) keys += iterator.next()
            require(keys == MODEL_FILES.toSet()) {
                "$source has an unexpected model file set"
            }
            val files = MODEL_FILES.associateWith { name ->
                val entry = filesObject.optJSONObject(name)
                    ?: error("$source has invalid entry for $name")
                val bytesValue = entry.opt("bytes")
                require(bytesValue is Number) { "$source has invalid byte count for $name" }
                val bytes = (bytesValue as Number).toLong()
                require(bytes > 0L) { "$source has invalid byte count for $name" }
                val sha256 = entry.optString("sha256", "")
                require(ASSET_ID_PATTERN.matches(sha256)) {
                    "$source has invalid SHA-256 for $name"
                }
                ModelFile(bytes, sha256)
            }
            return ModelManifest(assetSetId, files)
        }

        private fun installModels(context: Context, manifest: ModelManifest): File {
            synchronized(modelInstallLock) {
                return installModelsLocked(context, manifest)
            }
        }

        private fun installModelsLocked(context: Context, manifest: ModelManifest): File {
            val root = File(context.noBackupFilesDir, CACHE_ROOT)
            check(root.exists() || root.mkdirs()) { "Unable to create model cache root" }
            val directory = File(root, "$MODEL_ID-${manifest.assetSetId}")
            if (isCachedModelsValid(directory, manifest)) return directory
            if (directory.exists()) {
                check(directory.deleteRecursively()) { "Unable to replace invalid model cache" }
            }

            val temporary = File(root, ".${directory.name}.tmp-${System.nanoTime()}")
            temporary.deleteRecursively()
            check(temporary.mkdirs()) { "Unable to create model cache directory" }
            try {
                for (name in MODEL_FILES) {
                    copyAssetVerified(
                        context = context,
                        assetPath = "$ASSET_ROOT/$name",
                        target = File(temporary, name),
                        expected = manifest.files.getValue(name),
                    )
                }
                copyAsset(
                    context,
                    "$ASSET_ROOT/$MANIFEST_NAME",
                    File(temporary, MANIFEST_NAME),
                )
                File(temporary, MARKER_NAME).writeText(manifest.assetSetId + "\n", Charsets.UTF_8)
                check(temporary.renameTo(directory)) { "Unable to commit model cache" }
                return directory
            } catch (error: Throwable) {
                temporary.deleteRecursively()
                throw LwPpocrException("Unable to install OCR models", error)
            }
        }

        private fun isCachedModelsValid(directory: File, manifest: ModelManifest): Boolean {
            if (!directory.isDirectory) return false
            return runCatching {
                val marker = File(directory, MARKER_NAME)
                val cachedManifestFile = File(directory, MANIFEST_NAME)
                if (!marker.isFile || !cachedManifestFile.isFile) return false
                if (marker.readText(Charsets.UTF_8).trim() != manifest.assetSetId) return false
                val cached = parseManifest(
                    cachedManifestFile.readText(Charsets.UTF_8),
                    "cached model manifest",
                )
                if (cached != manifest) return false
                MODEL_FILES.all { name ->
                    val file = File(directory, name)
                    file.isFile && file.length() == manifest.files.getValue(name).bytes
                }
            }.getOrDefault(false)
        }

        private fun copyAssetVerified(
            context: Context,
            assetPath: String,
            target: File,
            expected: ModelFile,
        ) {
            val digest = MessageDigest.getInstance("SHA-256")
            var totalBytes = 0L
            context.assets.open(assetPath).use { input ->
                FileOutputStream(target).use { output ->
                    val buffer = ByteArray(64 * 1024)
                    while (true) {
                        val count = input.read(buffer)
                        if (count < 0) break
                        if (count == 0) continue
                        output.write(buffer, 0, count)
                        digest.update(buffer, 0, count)
                        totalBytes += count.toLong()
                    }
                }
            }
            val actualSha256 = digest.digest().joinToString("") { "%02x".format(it) }
            check(totalBytes == expected.bytes && actualSha256 == expected.sha256) {
                "Model asset verification failed for $assetPath"
            }
        }

        private fun copyAsset(context: Context, assetPath: String, target: File) {
            context.assets.open(assetPath).use { input ->
                FileOutputStream(target).use { output -> input.copyTo(output) }
            }
        }

        private fun pruneOldCaches(activeDirectory: File) {
            val root = activeDirectory.parentFile ?: return
            root.listFiles()?.forEach { candidate ->
                if (candidate.isDirectory &&
                    candidate != activeDirectory &&
                    candidate.name.startsWith("$MODEL_ID-")
                ) {
                    candidate.deleteRecursively()
                }
            }
        }

        private val modelInstallLock = Any()
    }
}
