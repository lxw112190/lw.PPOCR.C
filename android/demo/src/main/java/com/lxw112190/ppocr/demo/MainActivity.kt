package com.lxw112190.ppocr.demo

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.CheckBox
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.lifecycle.lifecycleScope
import com.lxw112190.ppocr.LwPpocrEngine
import com.lxw112190.ppocr.OcrOptions
import kotlinx.coroutines.launch
import java.io.InputStream

class MainActivity : ComponentActivity() {
    private var engine: LwPpocrEngine? = null
    private lateinit var status: TextView
    private lateinit var statusDetail: TextView
    private lateinit var statusIndicator: TextView
    private lateinit var result: TextView
    private lateinit var resultMeta: TextView
    private lateinit var useCls: CheckBox
    private lateinit var choose: Button

    private val navy = Color.rgb(16, 42, 67)
    private val muted = Color.rgb(98, 125, 152)
    private val blue = Color.rgb(21, 101, 192)
    private val green = Color.rgb(0, 137, 123)
    private val red = Color.rgb(198, 40, 40)

    private val pickImage =
        registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
            if (uri == null) return@registerForActivityResult
            choose.isEnabled = false
            useCls.isEnabled = false
            setStatus("正在识别图片…", "正在运行离线 OCR 引擎", blue)
            lifecycleScope.launch {
                runCatching {
                    val bitmap = contentResolver.openInputStream(uri).use(::decodeArgb8888)
                    try {
                        (engine ?: error("OCR 引擎尚未准备好")).recognize(bitmap)
                    } finally {
                        if (!bitmap.isRecycled) bitmap.recycle()
                    }
                }.onSuccess { ocr ->
                    setStatus("识别完成", "离线处理 · ${ocr.elapsedMs} ms", green)
                    resultMeta.text = "${ocr.lines.size} 行识别结果"
                    result.text = if (ocr.lines.isEmpty()) {
                        "未检测到文字"
                    } else {
                        ocr.lines.joinToString("\n") { line ->
                            "${line.index + 1}. ${line.text}"
                        }
                    }
                    choose.isEnabled = engine != null
                    useCls.isEnabled = engine != null
                }.onFailure { error ->
                    setStatus("识别失败", error.message ?: error.javaClass.simpleName, red)
                    resultMeta.text = "请重新选择图片"
                    choose.isEnabled = engine != null
                    useCls.isEnabled = engine != null
                }
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val root = ScrollView(this).apply {
            setBackgroundColor(Color.rgb(246, 248, 252))
            isFillViewport = true
        }
        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(24), dp(20), dp(32))
        }
        root.addView(content, ViewGroup.LayoutParams(-1, -2))

        val header = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(4), 0, dp(4), dp(20))
        }
        header.addView(TextView(this).apply {
            text = "lw.PPOCR"
            textSize = 28f
            setTextColor(navy)
            typeface = Typeface.create("sans", Typeface.BOLD)
        })
        header.addView(TextView(this).apply {
            text = "Android ARM64  ·  离线文字识别"
            textSize = 14f
            setTextColor(muted)
            setPadding(0, dp(4), 0, 0)
        })
        content.addView(header)

        val statusCard = card()
        val statusRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        statusIndicator = TextView(this).apply {
            text = "●"
            textSize = 18f
            setTextColor(blue)
            gravity = Gravity.CENTER
        }
        statusRow.addView(statusIndicator, LinearLayout.LayoutParams(dp(32), dp(32)))
        val statusTextColumn = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(10), 0, 0, 0)
        }
        status = TextView(this).apply {
            textSize = 16f
            setTextColor(navy)
            typeface = Typeface.create("sans", Typeface.BOLD)
        }
        statusDetail = TextView(this).apply {
            textSize = 13f
            setTextColor(muted)
            setPadding(0, dp(3), 0, 0)
        }
        statusTextColumn.addView(status)
        statusTextColumn.addView(statusDetail)
        statusRow.addView(statusTextColumn, LinearLayout.LayoutParams(0, -2, 1f))
        statusCard.addView(statusRow)
        content.addView(statusCard, marginParams(bottom = 16))

        content.addView(sectionLabel("识别设置"), marginParams(bottom = 8))
        val settingsCard = card()
        useCls = CheckBox(this).apply {
            text = "启用方向分类（CLS）"
            textSize = 15f
            setTextColor(navy)
            buttonTintList = android.content.res.ColorStateList.valueOf(blue)
        }
        settingsCard.addView(useCls, LinearLayout.LayoutParams(-1, -2))
        settingsCard.addView(TextView(this).apply {
            text = "适合方向不固定的图片；开启后会重新初始化引擎。"
            textSize = 13f
            setTextColor(muted)
            setPadding(dp(4), dp(2), dp(4), 0)
        }, LinearLayout.LayoutParams(-1, -2))
        content.addView(settingsCard, marginParams(bottom = 16))

        choose = Button(this).apply {
            text = "选择图片并识别"
            textSize = 16f
            isAllCaps = false
            minHeight = dp(52)
            setTextColor(Color.WHITE)
            background = rounded(blue, 14)
            elevation = dp(3).toFloat()
            isEnabled = false
            setOnClickListener {
                pickImage.launch(
                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                )
            }
        }
        content.addView(choose, marginParams(bottom = 24))

        val resultCard = card()
        val resultHeader = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        resultHeader.addView(TextView(this).apply {
            text = "识别结果"
            textSize = 18f
            setTextColor(navy)
            typeface = Typeface.create("sans", Typeface.BOLD)
        }, LinearLayout.LayoutParams(0, -2, 1f))
        resultMeta = TextView(this).apply {
            text = "等待图片"
            textSize = 13f
            setTextColor(muted)
            gravity = Gravity.END
        }
        resultHeader.addView(resultMeta, LinearLayout.LayoutParams(-2, -2))
        resultCard.addView(resultHeader)
        resultCard.addView(View(this).apply {
            setBackgroundColor(Color.rgb(226, 232, 240))
        }, LinearLayout.LayoutParams(-1, dp(1)).apply {
            topMargin = dp(14)
            bottomMargin = dp(4)
        })
        result = TextView(this).apply {
            text = "选择一张图片，识别文字会显示在这里。"
            textSize = 16f
            setTextColor(Color.rgb(50, 67, 86))
            setPadding(dp(4), dp(12), dp(4), dp(16))
            setLineSpacing(dp(4).toFloat(), 1f)
            typeface = Typeface.create("sans", Typeface.NORMAL)
            textIsSelectable = true
        }
        val resultScroll = ScrollView(this).apply {
            isFillViewport = true
            addView(result, ViewGroup.LayoutParams(-1, -2))
        }
        resultCard.addView(resultScroll, LinearLayout.LayoutParams(-1, dp(280)))
        content.addView(resultCard)
        content.addView(TextView(this).apply {
            text = "模型和推理均在本机完成，不上传图片。"
            textSize = 12f
            setTextColor(muted)
            gravity = Gravity.CENTER
            setPadding(0, dp(18), 0, 0)
        }, LinearLayout.LayoutParams(-1, -2))

        setContentView(root)
        useCls.setOnCheckedChangeListener { _, checked ->
            if (engine != null) {
                engine?.close()
                engine = null
                choose.isEnabled = false
                useCls.isEnabled = false
                setStatus("正在重新创建引擎…", "正在应用 CLS 设置", blue)
                createEngine(checked)
            }
        }
        createEngine(useCls.isChecked)
    }

    private fun createEngine(enableCls: Boolean) {
        choose.isEnabled = false
        useCls.isEnabled = false
        setStatus("正在准备引擎…", "首次启动会准备离线模型", blue)
        lifecycleScope.launch {
            runCatching {
                LwPpocrEngine.create(this@MainActivity, OcrOptions(useCls = enableCls))
            }.onSuccess {
                engine = it
                choose.isEnabled = true
                useCls.isEnabled = true
                setStatus("引擎已就绪", "可以选择图片开始识别", green)
            }.onFailure {
                choose.isEnabled = false
                useCls.isEnabled = true
                setStatus("初始化失败", it.message ?: it.javaClass.simpleName, red)
            }
        }
    }

    private fun setStatus(title: String, detail: String, color: Int) {
        status.text = title
        statusDetail.text = detail
        statusIndicator.setTextColor(color)
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

    private fun card(): LinearLayout = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(18), dp(16), dp(18), dp(16))
        background = rounded(Color.WHITE, 18)
        elevation = dp(2).toFloat()
    }

    private fun sectionLabel(text: String): TextView = TextView(this).apply {
        this.text = text
        textSize = 13f
        setTextColor(muted)
        typeface = Typeface.create("sans", Typeface.BOLD)
        letterSpacing = 0.08f
    }

    private fun rounded(color: Int, radiusDp: Int): GradientDrawable =
        GradientDrawable().apply {
            setColor(color)
            cornerRadius = dp(radiusDp).toFloat()
        }

    private fun marginParams(bottom: Int = 0): LinearLayout.LayoutParams =
        LinearLayout.LayoutParams(-1, -2).apply {
            this.bottomMargin = bottom
        }

    private fun dp(value: Int): Int =
        (value * resources.displayMetrics.density + 0.5f).toInt()
}
