package io.t6x.dust.onnx

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

object ImagePreprocessor {
    private val defaultMean = listOf(0.485, 0.456, 0.406)
    private val defaultStd = listOf(0.229, 0.224, 0.225)

    fun preprocess(
        imageData: ByteArray,
        targetWidth: Int,
        targetHeight: Int,
        resize: String,
        normalization: String,
        customMean: List<Double>?,
        customStd: List<Double>?,
    ): TensorData {
        if (targetWidth <= 0 || targetHeight <= 0) {
            throw ONNXError.PreprocessError("Target dimensions must be greater than zero")
        }

        if (!looksLikeSupportedImage(imageData)) {
            throw ONNXError.PreprocessError("Unable to decode image data")
        }

        val source = BitmapFactory.decodeByteArray(imageData, 0, imageData.size)
            ?: throw ONNXError.PreprocessError("Unable to decode image data")
        val processed = renderBitmap(source, targetWidth, targetHeight, resize)

        try {
            return extractTensor(
                bitmap = processed,
                targetWidth = targetWidth,
                targetHeight = targetHeight,
                normalization = normalization,
                customMean = customMean,
                customStd = customStd,
            )
        } finally {
            if (processed !== source && !processed.isRecycled) {
                processed.recycle()
            }
            if (!source.isRecycled) {
                source.recycle()
            }
        }
    }

    private fun renderBitmap(
        source: Bitmap,
        targetWidth: Int,
        targetHeight: Int,
        resize: String,
    ): Bitmap {
        val plan = calculateResizePlan(source.width, source.height, targetWidth, targetHeight, resize)

        return when (resize) {
            "stretch" -> Bitmap.createScaledBitmap(source, targetWidth, targetHeight, true)
            "letterbox" -> {
                val output = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
                val canvas = Canvas(output)
                val paint = Paint(Paint.FILTER_BITMAP_FLAG)
                canvas.drawColor(Color.rgb(114, 114, 114))
                val scaled = Bitmap.createScaledBitmap(source, plan.scaledWidth, plan.scaledHeight, true)
                canvas.drawBitmap(scaled, plan.offsetX.toFloat(), plan.offsetY.toFloat(), paint)
                if (scaled !== source && !scaled.isRecycled) {
                    scaled.recycle()
                }
                output
            }
            "crop_center" -> {
                val scaled = Bitmap.createScaledBitmap(source, plan.scaledWidth, plan.scaledHeight, true)
                val offsetX = -plan.offsetX
                val offsetY = -plan.offsetY
                val cropped = Bitmap.createBitmap(scaled, offsetX, offsetY, targetWidth, targetHeight)
                if (scaled !== source && scaled !== cropped && !scaled.isRecycled) {
                    scaled.recycle()
                }
                cropped
            }
            else -> throw ONNXError.PreprocessError("Unsupported resize mode: $resize")
        }
    }

    private fun extractTensor(
        bitmap: Bitmap,
        targetWidth: Int,
        targetHeight: Int,
        normalization: String,
        customMean: List<Double>?,
        customStd: List<Double>?,
    ): TensorData {
        if (customMean != null && customMean.size != 3) {
            throw ONNXError.PreprocessError("Custom mean must contain three values")
        }
        if (customStd != null && customStd.size != 3) {
            throw ONNXError.PreprocessError("Custom std must contain three values")
        }

        val useCustomStatistics = customMean != null || customStd != null
        val mean = if (useCustomStatistics) {
            customMean ?: listOf(0.0, 0.0, 0.0)
        } else {
            defaultMean
        }
        val std = if (useCustomStatistics) {
            customStd ?: listOf(1.0, 1.0, 1.0)
        } else {
            defaultStd
        }

        if (useCustomStatistics && std.any { it == 0.0 }) {
            throw ONNXError.PreprocessError("Custom std values must be non-zero")
        }

        val pixels = IntArray(targetWidth * targetHeight)
        bitmap.getPixels(pixels, 0, targetWidth, 0, 0, targetWidth, targetHeight)

        val planeSize = targetWidth * targetHeight
        val data = MutableList(planeSize * 3) { 0.0 }

        for ((index, pixel) in pixels.withIndex()) {
            val channels = listOf(
                Color.red(pixel).toDouble(),
                Color.green(pixel).toDouble(),
                Color.blue(pixel).toDouble(),
            )

            for (channel in 0 until 3) {
                data[(channel * planeSize) + index] = normalize(
                    pixel = channels[channel],
                    channel = channel,
                    normalization = normalization,
                    mean = mean,
                    std = std,
                    useCustomStatistics = useCustomStatistics,
                )
            }
        }

        return TensorData(
            name = "image",
            dtype = "float32",
            shape = listOf(1, 3, targetHeight, targetWidth),
            data = data,
        )
    }

    private fun normalize(
        pixel: Double,
        channel: Int,
        normalization: String,
        mean: List<Double>,
        std: List<Double>,
        useCustomStatistics: Boolean,
    ): Double {
        if (useCustomStatistics) {
            return ((pixel / 255.0) - mean[channel]) / std[channel]
        }

        return when (normalization) {
            "imagenet" -> ((pixel / 255.0) - mean[channel]) / std[channel]
            "minus1_plus1" -> (pixel / 127.5) - 1.0
            "zero_to_1" -> pixel / 255.0
            "none" -> pixel
            else -> throw ONNXError.PreprocessError("Unsupported normalization mode: $normalization")
        }
    }

    internal fun calculateResizePlan(
        sourceWidth: Int,
        sourceHeight: Int,
        targetWidth: Int,
        targetHeight: Int,
        resize: String,
    ): ResizePlan {
        return when (resize) {
            "stretch" -> ResizePlan(
                scaledWidth = targetWidth,
                scaledHeight = targetHeight,
                offsetX = 0,
                offsetY = 0,
            )
            "letterbox" -> {
                val scale = min(
                    targetWidth.toDouble() / sourceWidth.toDouble(),
                    targetHeight.toDouble() / sourceHeight.toDouble(),
                )
                val scaledWidth = max(1, (sourceWidth * scale).roundToInt()).coerceAtMost(targetWidth)
                val scaledHeight = max(1, (sourceHeight * scale).roundToInt()).coerceAtMost(targetHeight)
                ResizePlan(
                    scaledWidth = scaledWidth,
                    scaledHeight = scaledHeight,
                    offsetX = (targetWidth - scaledWidth) / 2,
                    offsetY = (targetHeight - scaledHeight) / 2,
                )
            }
            "crop_center" -> {
                val scale = max(
                    targetWidth.toDouble() / sourceWidth.toDouble(),
                    targetHeight.toDouble() / sourceHeight.toDouble(),
                )
                val scaledWidth = max(targetWidth, ceil(sourceWidth * scale).toInt())
                val scaledHeight = max(targetHeight, ceil(sourceHeight * scale).toInt())
                ResizePlan(
                    scaledWidth = scaledWidth,
                    scaledHeight = scaledHeight,
                    offsetX = (targetWidth - scaledWidth) / 2,
                    offsetY = (targetHeight - scaledHeight) / 2,
                )
            }
            else -> throw ONNXError.PreprocessError("Unsupported resize mode: $resize")
        }
    }

    private fun looksLikeSupportedImage(imageData: ByteArray): Boolean {
        if (imageData.size >= 8 &&
            imageData[0] == 0x89.toByte() &&
            imageData[1] == 0x50.toByte() &&
            imageData[2] == 0x4E.toByte() &&
            imageData[3] == 0x47.toByte() &&
            imageData[4] == 0x0D.toByte() &&
            imageData[5] == 0x0A.toByte() &&
            imageData[6] == 0x1A.toByte() &&
            imageData[7] == 0x0A.toByte()
        ) {
            return true
        }

        return imageData.size >= 3 &&
            imageData[0] == 0xFF.toByte() &&
            imageData[1] == 0xD8.toByte() &&
            imageData[2] == 0xFF.toByte()
    }
}

internal data class ResizePlan(
    val scaledWidth: Int,
    val scaledHeight: Int,
    val offsetX: Int,
    val offsetY: Int,
)
