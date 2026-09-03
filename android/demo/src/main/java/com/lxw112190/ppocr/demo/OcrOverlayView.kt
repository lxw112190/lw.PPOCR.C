package com.lxw112190.ppocr.demo

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.view.View
import com.lxw112190.ppocr.OcrLine

class OcrOverlayView(context: android.content.Context) : View(context) {
    private val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.WHITE
        textAlign = Paint.Align.CENTER
        textSize = 12f * resources.displayMetrics.scaledDensity
        isFakeBoldText = true
    }
    private val labelBackgroundPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val path = Path()
    private val mappedPoints = FloatArray(8)
    private val labelRect = RectF()
    private var sourceImage: android.widget.ImageView? = null
    private var lines: List<OcrLine> = emptyList()
    private var selectedIndex: Int? = null
    private var annotationsVisible = true

    fun setSourceImage(imageView: android.widget.ImageView) {
        sourceImage = imageView
        invalidate()
    }

    fun setLines(value: List<OcrLine>) {
        lines = value
        invalidate()
    }

    fun setSelectedIndex(value: Int?) {
        selectedIndex = value
        invalidate()
    }

    fun setAnnotationsVisible(value: Boolean) {
        annotationsVisible = value
        visibility = if (value) VISIBLE else INVISIBLE
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (!annotationsVisible || lines.isEmpty()) return
        val image = sourceImage ?: return
        val matrix = Matrix(image.imageMatrix)
        val density = resources.displayMetrics.density
        lines.forEachIndexed { position, line ->
            if (line.box.size < 8) return@forEachIndexed
            for (index in 0 until 8) mappedPoints[index] = line.box[index]
            matrix.mapPoints(mappedPoints)
            path.reset()
            path.moveTo(mappedPoints[0], mappedPoints[1])
            path.lineTo(mappedPoints[2], mappedPoints[3])
            path.lineTo(mappedPoints[4], mappedPoints[5])
            path.lineTo(mappedPoints[6], mappedPoints[7])
            path.close()

            val selected = selectedIndex == position || selectedIndex == line.index
            val color = if (selected) Color.rgb(21, 101, 192) else Color.rgb(198, 40, 40)
            if (selected) {
                fillPaint.color = Color.argb(42, 21, 101, 192)
                canvas.drawPath(path, fillPaint)
            }
            strokePaint.color = color
            strokePaint.strokeWidth = (if (selected) 3f else 2f) * density
            canvas.drawPath(path, strokePaint)

            val labelX = mappedPoints[0]
            val labelY = mappedPoints[1]
            val label = "%02d".format(line.index + 1)
            val halfWidth = labelPaint.measureText(label) / 2f + 5f * density
            val labelHeight = 18f * density
            labelRect.set(
                labelX - halfWidth,
                labelY - labelHeight,
                labelX + halfWidth,
                labelY,
            )
            labelBackgroundPaint.color = color
            canvas.drawRoundRect(labelRect, 4f * density, 4f * density, labelBackgroundPaint)
            val baseline = labelRect.centerY() - (labelPaint.ascent() + labelPaint.descent()) / 2f
            canvas.drawText(label, labelRect.centerX(), baseline, labelPaint)
        }
    }
}
