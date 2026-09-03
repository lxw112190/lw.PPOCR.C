package com.lxw112190.ppocr.demo

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.ViewGroup
import android.widget.Button
import android.widget.CheckBox
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.result.PickVisualMediaRequest
import androidx.lifecycle.lifecycleScope
import com.lxw112190.ppocr.LwPpocrEngine
import com.lxw112190.ppocr.OcrOptions
import kotlinx.coroutines.launch
import java.io.InputStream

class MainActivity : ComponentActivity() {
    private var engine: LwPpocrEngine? = null
    private lateinit var status: TextView
    private lateinit var result: TextView
    private lateinit var useCls: CheckBox
    private lateinit var choose: Button

    private val pickImage = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri == null) return@registerForActivityResult
        lifecycleScope.launch {
            status.text = "正在读取图片…"
            runCatching {
                val bitmap = contentResolver.openInputStream(uri).use(::decodeArgb8888)
                try {
                    (engine ?: error("OCR 引擎尚未准备好")).recognize(bitmap)
                } finally {
                    if (!bitmap.isRecycled) bitmap.recycle()
                }
            }.onSuccess { ocr ->
                status.text = "完成：${ocr.lines.size} 行，${ocr.elapsedMs} ms"
                result.text = ocr.lines.joinToString("\n") { line ->
                    "${line.index + 1}. ${line.text}"
                }
            }.onFailure { error ->
                status.text = "失败：${error.message ?: error.javaClass.simpleName}"
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        status = TextView(this).apply { text = "正在加载模型…" }
        result = TextView(this).apply { setPadding(16, 16, 16, 16); textSize = 16f }
        useCls = CheckBox(this).apply { text = "启用方向分类（CLS）" }
        choose = Button(this).apply {
            text = "选择图片并识别"
            isEnabled = false
            setOnClickListener {
                pickImage.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
            }
        }
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(20, 20, 20, 20)
            addView(status, ViewGroup.LayoutParams(-1, -2))
            addView(useCls, ViewGroup.LayoutParams(-1, -2))
            addView(choose, ViewGroup.LayoutParams(-1, -2))
            addView(ScrollView(this@MainActivity).apply { addView(result) },
                LinearLayout.LayoutParams(-1, 0, 1f))
        }
        setContentView(root)
        useCls.setOnCheckedChangeListener { _, checked ->
            if (engine != null) {
                engine?.close()
                engine = null
                choose.isEnabled = false
                useCls.isEnabled = false
                status.text = "正在重新创建引擎…"
                createEngine(checked)
            }
        }
        createEngine(useCls.isChecked)
    }

    private fun createEngine(enableCls: Boolean) {
        useCls.isEnabled = false
        lifecycleScope.launch {
            runCatching {
                LwPpocrEngine.create(this@MainActivity, OcrOptions(useCls = enableCls))
            }.onSuccess {
                engine = it
                choose.isEnabled = true
                useCls.isEnabled = true
                status.text = "就绪（ARM64 Native Preview）"
            }.onFailure {
                useCls.isEnabled = true
                status.text = "初始化失败：${it.message ?: it.javaClass.simpleName}"
            }
        }
    }

    override fun onDestroy() {
        engine?.close()
        engine = null
        super.onDestroy()
    }

    private fun decodeArgb8888(input: InputStream?): Bitmap {
        requireNotNull(input) { "无法读取图片" }
        val decoded = BitmapFactory.decodeStream(input) ?: error("图片格式不受支持")
        return if (decoded.config == Bitmap.Config.ARGB_8888) decoded
        else decoded.copy(Bitmap.Config.ARGB_8888, false).also { decoded.recycle() }
    }
}
