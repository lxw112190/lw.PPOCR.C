package com.lxw112190.ppocr.demo

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.PopupMenu
import android.widget.Spinner
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.lxw112190.ppocr.LwPpocrEngine
import com.lxw112190.ppocr.OcrOptions
import com.lxw112190.ppocr.OcrResult
import com.lxw112190.ppocr.ReadingOrder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    private enum class UiState {
        ENGINE_LOADING,
        READY,
        IMAGE_LOADING,
        IMAGE_READY,
        APPLYING_SETTINGS,
        RECOGNIZING,
        RESULT_READY,
        ENGINE_ERROR,
        OPERATION_ERROR,
    }

    private var engine: LwPpocrEngine? = null
    private var engineUseCls = false
    private var desiredUseCls = false
    private var desiredReadingOrder = ReadingOrder.HORIZONTAL_LTR
    private var currentBitmap: android.graphics.Bitmap? = null
    private var currentResult: OcrResult? = null
    private var currentSourceName = "image"
    private var state = UiState.ENGINE_LOADING
    private var pendingExportContent: String? = null

    private lateinit var status: TextView
    private lateinit var statusDetail: TextView
    private lateinit var statusIndicator: TextView
    private lateinit var chooseButton: Button
    private lateinit var runButton: Button
    private lateinit var useCls: Switch
    private lateinit var readingOrder: Spinner
    private lateinit var preview: OcrPreviewView
    private lateinit var toggleAnnotations: Button
    private lateinit var resultMeta: TextView
    private lateinit var resultEmpty: TextView
    private lateinit var copyButton: Button
    private lateinit var shareButton: Button
    private lateinit var moreButton: Button
    private lateinit var resultList: RecyclerView
    private lateinit var resultAdapter: OcrResultAdapter

    private val blue = android.graphics.Color.rgb(21, 101, 192)
    private val green = android.graphics.Color.rgb(0, 137, 123)
    private val red = android.graphics.Color.rgb(198, 40, 40)

    private val pickImage =
        registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
            if (uri != null) prepareImage(uri)
        }

    private val createTextDocument =
        registerForActivityResult(ActivityResultContracts.CreateDocument("text/plain")) { uri ->
            finishDocumentSelection(uri)
        }

    private val createJsonDocument =
        registerForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { uri ->
            finishDocumentSelection(uri)
        }

    private fun finishDocumentSelection(uri: Uri?) {
        val content = pendingExportContent
        pendingExportContent = null
        if (uri != null && content != null) saveExport(uri, content)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        status = findViewById(R.id.status)
        statusDetail = findViewById(R.id.status_detail)
        statusIndicator = findViewById(R.id.status_indicator)
        chooseButton = findViewById(R.id.choose_button)
        runButton = findViewById(R.id.run_button)
        useCls = findViewById(R.id.use_cls)
        readingOrder = findViewById(R.id.reading_order)
        preview = findViewById(R.id.preview)
        toggleAnnotations = findViewById(R.id.toggle_annotations)
        resultMeta = findViewById(R.id.result_meta)
        resultEmpty = findViewById(R.id.result_empty)
        copyButton = findViewById(R.id.copy_button)
        shareButton = findViewById(R.id.share_button)
        moreButton = findViewById(R.id.more_button)
        resultList = findViewById(R.id.result_list)

        resultAdapter = OcrResultAdapter { index ->
            preview.setSelectedLine(index)
        }
        resultList.layoutManager = LinearLayoutManager(this)
        resultList.adapter = resultAdapter
        resultList.isNestedScrollingEnabled = false

        readingOrder.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            listOf("标准横排", "古籍竖排（右→左）", "竖排（左→右）"),
        ).apply {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }

        chooseButton.setOnClickListener {
            pickImage.launch(
                PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
            )
        }
        runButton.setOnClickListener { recognizeCurrentImage() }
        toggleAnnotations.setOnClickListener {
            val visible = !preview.annotationsVisible()
            preview.setAnnotationsVisible(visible)
            toggleAnnotations.text = if (visible) "隐藏标注" else "显示标注"
        }
        copyButton.setOnClickListener { copyCurrentText() }
        shareButton.setOnClickListener { shareCurrentText() }
        moreButton.setOnClickListener { showExportMenu() }

        useCls.setOnCheckedChangeListener { _, checked ->
            if (desiredUseCls == checked) return@setOnCheckedChangeListener
            desiredUseCls = checked
            clearCurrentResult()
            if (state != UiState.ENGINE_LOADING) {
                state = if (currentBitmap == null) UiState.READY else UiState.IMAGE_READY
                renderUiState()
            }
        }
        readingOrder.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) = Unit

            override fun onItemSelected(
                parent: android.widget.AdapterView<*>?,
                view: View?,
                position: Int,
                id: Long,
            ) {
                val selected = when (position) {
                    1 -> ReadingOrder.VERTICAL_RTL
                    2 -> ReadingOrder.VERTICAL_LTR
                    else -> ReadingOrder.HORIZONTAL_LTR
                }
                if (desiredReadingOrder != selected) {
                    desiredReadingOrder = selected
                    clearCurrentResult()
                    if (state != UiState.ENGINE_LOADING) {
                        state = if (currentBitmap == null) UiState.READY else UiState.IMAGE_READY
                        renderUiState()
                    }
                }
            }
        }

        preview.setAnnotationsVisible(true)
        state = UiState.ENGINE_LOADING
        renderUiState()
        createInitialEngine()
    }

    private fun createInitialEngine() {
        lifecycleScope.launch {
            runCatching {
                LwPpocrEngine.create(
                    this@MainActivity,
                    OcrOptions(useCls = desiredUseCls, readingOrder = desiredReadingOrder),
                )
            }.onSuccess {
                engine = it
                engineUseCls = desiredUseCls
                state = if (currentBitmap == null) UiState.READY else UiState.IMAGE_READY
                renderUiState()
            }.onFailure {
                state = UiState.ENGINE_ERROR
                renderUiState(it.message ?: it.javaClass.simpleName)
            }
        }
    }

    private fun prepareImage(uri: Uri) {
        state = UiState.IMAGE_LOADING
        renderUiState()
        lifecycleScope.launch {
            runCatching { DemoImageLoader.load(this@MainActivity, uri) }
                .onSuccess { bitmap ->
                    replaceBitmap(bitmap)
                    currentSourceName = displayName(uri)
                    clearCurrentResult()
                    state = if (engine == null) UiState.ENGINE_ERROR else UiState.IMAGE_READY
                    renderUiState()
                }
                .onFailure {
                    state = UiState.OPERATION_ERROR
                    renderUiState(it.message ?: it.javaClass.simpleName)
                }
        }
    }

    private fun recognizeCurrentImage() {
        val bitmap = currentBitmap ?: return
        state = UiState.APPLYING_SETTINGS
        renderUiState()
        lifecycleScope.launch {
            runCatching {
                val active = ensureEngineForCls(desiredUseCls)
                active.setReadingOrder(desiredReadingOrder)
                state = UiState.RECOGNIZING
                renderUiState()
                active.recognize(bitmap)
            }.onSuccess { result ->
                currentResult = result
                preview.setResult(result)
                resultAdapter.submitList(result.lines)
                state = UiState.RESULT_READY
                renderUiState()
            }.onFailure {
                state = if (engine == null) UiState.ENGINE_ERROR else UiState.OPERATION_ERROR
                renderUiState(it.message ?: it.javaClass.simpleName)
            }
        }
    }

    private suspend fun ensureEngineForCls(useCls: Boolean): LwPpocrEngine {
        val existing = engine
        if (existing != null && engineUseCls == useCls) return existing
        existing?.close()
        engine = null
        val fresh = LwPpocrEngine.create(
            this@MainActivity,
            OcrOptions(useCls = useCls, readingOrder = desiredReadingOrder),
        )
        engine = fresh
        engineUseCls = useCls
        return fresh
    }

    private fun replaceBitmap(bitmap: android.graphics.Bitmap) {
        val old = currentBitmap
        preview.setBitmap(null)
        currentBitmap = null
        if (old != null && !old.isRecycled) old.recycle()
        currentBitmap = bitmap
        preview.setBitmap(bitmap)
    }

    private fun clearCurrentResult() {
        currentResult = null
        resultAdapter.submitList(emptyList())
        resultAdapter.setSelectedIndex(null)
        preview.clearResult()
        resultEmpty.visibility = View.VISIBLE
        resultMeta.text = if (currentBitmap == null) "等待图片" else "等待识别"
        renderExportEnabled()
    }

    private fun renderUiState(errorMessage: String? = null) {
        val busy = state == UiState.ENGINE_LOADING ||
            state == UiState.IMAGE_LOADING ||
            state == UiState.APPLYING_SETTINGS ||
            state == UiState.RECOGNIZING
        chooseButton.isEnabled = !busy && engine != null
        if (state == UiState.ENGINE_ERROR || state == UiState.OPERATION_ERROR) {
            chooseButton.isEnabled = !busy
        }
        runButton.isEnabled = !busy && currentBitmap != null
        useCls.isEnabled = !busy && engine != null
        readingOrder.isEnabled = !busy && engine != null
        toggleAnnotations.isEnabled = currentBitmap != null
        renderExportEnabled()

        when (state) {
            UiState.ENGINE_LOADING -> setStatus("正在准备引擎…", "首次启动会准备离线模型", blue)
            UiState.READY -> setStatus("引擎已就绪", "ARM64 Native · 可以选择图片", green)
            UiState.IMAGE_LOADING -> setStatus("正在准备图片…", "正在读取并规范化图片", blue)
            UiState.IMAGE_READY -> setStatus(
                "图片已准备好",
                "${currentBitmap?.width ?: 0} × ${currentBitmap?.height ?: 0} · 点击开始识别",
                green,
            )
            UiState.APPLYING_SETTINGS -> setStatus("正在应用识别设置…", "准备 CLS 和阅读顺序", blue)
            UiState.RECOGNIZING -> setStatus("正在识别…", "图片不会离开本机", blue)
            UiState.RESULT_READY -> {
                val result = currentResult
                setStatus(
                    "识别完成",
                    "${result?.lines?.size ?: 0} 行 · ${result?.elapsedMs ?: 0} ms",
                    green,
                )
            }
            UiState.ENGINE_ERROR -> setStatus("引擎初始化失败", errorMessage ?: "请重新打开应用重试", red)
            UiState.OPERATION_ERROR -> setStatus("处理失败", errorMessage ?: "请重新选择图片", red)
        }
    }

    private fun renderExportEnabled() {
        val enabled = currentResult != null
        copyButton.isEnabled = enabled
        shareButton.isEnabled = enabled
        moreButton.isEnabled = enabled
        resultEmpty.visibility = if (enabled) View.GONE else View.VISIBLE
        if (enabled) {
            resultMeta.text = "${currentResult?.lines?.size ?: 0} 行"
        }
    }

    private fun setStatus(title: String, detail: String, color: Int) {
        status.text = title
        statusDetail.text = detail
        statusIndicator.setTextColor(color)
    }

    private fun copyCurrentText() {
        val result = currentResult ?: return
        val text = OcrResultExporter.toText(result)
        val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        clipboard.setPrimaryClip(ClipData.newPlainText("OCR 文本", text))
        Toast.makeText(this, "已复制 ${result.lines.size} 行文本", Toast.LENGTH_SHORT).show()
    }

    private fun shareCurrentText() {
        val result = currentResult ?: return
        try {
            OcrResultExporter.shareText(this, result)
        } catch (error: Exception) {
            Toast.makeText(this, error.message ?: "系统分享不可用", Toast.LENGTH_SHORT).show()
        }
    }

    private fun showExportMenu() {
        val result = currentResult ?: return
        PopupMenu(this, moreButton).apply {
            menu.add("导出 TXT").setOnMenuItemClickListener {
                launchExport(
                    "text/plain",
                    ".txt",
                    OcrResultExporter.toText(result),
                )
                true
            }
            menu.add("导出 JSON").setOnMenuItemClickListener {
                launchExport(
                    "application/json",
                    ".json",
                    OcrResultExporter.toJson(
                        currentSourceName,
                        result,
                        desiredUseCls,
                        desiredReadingOrder,
                    ),
                )
                true
            }
        }.show()
    }

    private fun launchExport(mime: String, extension: String, content: String) {
        val base = currentSourceName.substringBeforeLast('.', currentSourceName)
            .replace(Regex("[\\\\/:*?\"<>|]"), "_")
            .ifBlank { "image" }
        pendingExportContent = content
        val fileName = "$base-ocr$extension"
        if (mime == "application/json") createJsonDocument.launch(fileName)
        else createTextDocument.launch(fileName)
    }

    private fun saveExport(uri: Uri, content: String) {
        lifecycleScope.launch(Dispatchers.IO) {
            runCatching { OcrResultExporter.writeText(this@MainActivity, uri, content) }
                .onSuccess {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@MainActivity, "已保存到所选位置", Toast.LENGTH_SHORT).show()
                    }
                }
                .onFailure {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(
                            this@MainActivity,
                            it.message ?: "保存失败",
                            Toast.LENGTH_SHORT,
                        ).show()
                    }
                }
        }
    }

    private fun displayName(uri: Uri): String {
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    if (index >= 0) return cursor.getString(index)
                }
            }
        return "image"
    }

    override fun onDestroy() {
        engine?.close()
        engine = null
        val old = currentBitmap
        currentBitmap = null
        preview.setBitmap(null)
        if (old != null && !old.isRecycled) old.recycle()
        super.onDestroy()
    }
}
