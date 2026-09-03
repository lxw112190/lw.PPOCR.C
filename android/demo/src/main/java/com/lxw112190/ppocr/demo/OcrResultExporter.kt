package com.lxw112190.ppocr.demo

import android.content.Context
import android.content.Intent
import com.lxw112190.ppocr.OcrResult
import com.lxw112190.ppocr.ReadingOrder
import org.json.JSONArray
import org.json.JSONObject
import java.nio.charset.StandardCharsets

object OcrResultExporter {
    fun toText(result: OcrResult): String =
        result.lines.joinToString("\n") { it.text }

    fun toJson(
        sourceName: String,
        result: OcrResult,
        useCls: Boolean,
        readingOrder: ReadingOrder,
    ): String {
        val lines = JSONArray()
        result.lines.forEach { line ->
            lines.put(JSONObject().apply {
                put("index", line.index)
                put("text", line.text)
                put("box", JSONArray().apply {
                    line.box.take(8).forEach { put(it.toDouble()) }
                })
                put("det_score", line.detScore.toDouble())
                put("rec_score", line.recScore.toDouble())
            })
        }
        return JSONObject().apply {
            put("schema_version", 1)
            put("source", sourceName)
            put("image", JSONObject().apply {
                put("width", result.width)
                put("height", result.height)
            })
            put("options", JSONObject().apply {
                put("use_cls", useCls)
                put("reading_order", readingOrderValue(readingOrder))
            })
            put("elapsed_ms", result.elapsedMs)
            put("lines", lines)
        }.toString(2) + "\n"
    }

    fun shareText(context: Context, result: OcrResult) {
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_TEXT, toText(result))
        }
        context.startActivity(Intent.createChooser(intent, "分享 OCR 文本"))
    }

    fun writeText(context: Context, uri: android.net.Uri, content: String) {
        context.contentResolver.openOutputStream(uri)?.use { output ->
            output.write(content.toByteArray(StandardCharsets.UTF_8))
        } ?: error("无法打开保存位置")
    }

    private fun readingOrderValue(readingOrder: ReadingOrder): String =
        when (readingOrder) {
            ReadingOrder.HORIZONTAL_LTR -> "horizontal-ltr"
            ReadingOrder.VERTICAL_RTL -> "vertical-rtl"
            ReadingOrder.VERTICAL_LTR -> "vertical-ltr"
        }
}
