package com.lxw112190.ppocr.demo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.AttributeSet
import android.view.Gravity
import android.view.View
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import com.lxw112190.ppocr.OcrLine
import com.lxw112190.ppocr.OcrResult

class OcrPreviewView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : FrameLayout(context, attrs) {
    private val imageView = ImageView(context).apply {
        scaleType = ImageView.ScaleType.FIT_CENTER
        setBackgroundColor(Color.rgb(235, 240, 247))
        contentDescription = "图片预览"
    }
    private val placeholder = TextView(context).apply {
        text = "请选择一张图片"
        textSize = 15f
        setTextColor(Color.rgb(98, 125, 152))
        gravity = Gravity.CENTER
    }
    private val overlay = OcrOverlayView(context).apply {
        setSourceImage(imageView)
        isClickable = false
    }
    private var annotationsVisible = true

    init {
        setBackgroundColor(Color.rgb(235, 240, 247))
        addView(imageView, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT))
        addView(placeholder, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT))
        addView(overlay, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT))
    }

    fun setBitmap(bitmap: Bitmap?) {
        imageView.setImageBitmap(bitmap)
        placeholder.visibility = if (bitmap == null) View.VISIBLE else View.GONE
        overlay.setLines(emptyList())
        post { overlay.invalidate() }
    }

    fun setResult(result: OcrResult?) {
        overlay.setLines(result?.lines ?: emptyList())
        post { overlay.invalidate() }
    }

    fun setSelectedLine(index: Int?) {
        overlay.setSelectedIndex(index)
    }

    fun setAnnotationsVisible(value: Boolean) {
        annotationsVisible = value
        overlay.setAnnotationsVisible(value)
    }

    fun annotationsVisible(): Boolean = annotationsVisible

    fun clearResult() {
        overlay.setLines(emptyList())
        overlay.setSelectedIndex(null)
    }
}
