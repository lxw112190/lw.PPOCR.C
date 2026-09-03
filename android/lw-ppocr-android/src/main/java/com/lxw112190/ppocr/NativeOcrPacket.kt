package com.lxw112190.ppocr

internal class NativeOcrPacket(
    val width: Int,
    val height: Int,
    val boxes: FloatArray,
    val detectorScores: FloatArray,
    val recognitionScores: FloatArray,
    val texts: Array<String>,
    val elapsedMs: Long,
)
