package io.t6x.dust.onnx

import android.graphics.Bitmap
import android.graphics.Color
import org.junit.Assert.assertEquals
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.io.ByteArrayOutputStream

@RunWith(RobolectricTestRunner::class)
class ONNXPreprocessTest {
    @Test
    fun o3T1SolidRedStretchImagenetNormalization() {
        val tensor = ImagePreprocessor.preprocess(
            imageData = createSolidColorPNG(width = 8, height = 8, r = 255, g = 0, b = 0),
            targetWidth = 8,
            targetHeight = 8,
            resize = "stretch",
            normalization = "imagenet",
            customMean = null,
            customStd = null,
        )

        assertEquals(listOf(1, 3, 8, 8), tensor.shape)
        assertEquals("float32", tensor.dtype)
        assertEquals(2.2489082969, value(tensor, channel = 0, x = 0, y = 0), 0.001)
        assertEquals(-2.0357142857, value(tensor, channel = 1, x = 0, y = 0), 0.001)
        assertEquals(-1.8044444444, value(tensor, channel = 2, x = 0, y = 0), 0.001)
    }

    @Test
    fun o3T2LetterboxAppliesPaddingAndCentersImage() {
        val plan = ImagePreprocessor.calculateResizePlan(
            sourceWidth = 8,
            sourceHeight = 16,
            targetWidth = 8,
            targetHeight = 8,
            resize = "letterbox",
        )
        val tensor = ImagePreprocessor.preprocess(
            imageData = createSolidColorPNG(width = 8, height = 16, r = 0, g = 0, b = 255),
            targetWidth = 8,
            targetHeight = 8,
            resize = "letterbox",
            normalization = "imagenet",
            customMean = null,
            customStd = null,
        )

        assertEquals(4, plan.scaledWidth)
        assertEquals(8, plan.scaledHeight)
        assertEquals(2, plan.offsetX)
        assertEquals(0, plan.offsetY)
        assertEquals(listOf(1, 3, 8, 8), tensor.shape)
        assertEquals(2.6399999999, value(tensor, channel = 2, x = 4, y = 4), 0.001)
    }

    @Test
    fun o3T3StretchUpscalesSmallImage() {
        val tensor = ImagePreprocessor.preprocess(
            imageData = createSolidColorPNG(width = 4, height = 4, r = 0, g = 255, b = 0),
            targetWidth = 8,
            targetHeight = 8,
            resize = "stretch",
            normalization = "imagenet",
            customMean = null,
            customStd = null,
        )

        assertEquals(2.4285714286, value(tensor, channel = 1, x = 3, y = 3), 0.001)
    }

    @Test
    fun o3T4MinusOneToOneNormalization() {
        val tensor = ImagePreprocessor.preprocess(
            imageData = createSolidColorPNG(width = 8, height = 8, r = 255, g = 255, b = 255),
            targetWidth = 8,
            targetHeight = 8,
            resize = "stretch",
            normalization = "minus1_plus1",
            customMean = null,
            customStd = null,
        )

        assertEquals(1.0, value(tensor, channel = 0, x = 0, y = 0), 0.0001)
        assertEquals(1.0, value(tensor, channel = 1, x = 0, y = 0), 0.0001)
        assertEquals(1.0, value(tensor, channel = 2, x = 0, y = 0), 0.0001)
    }

    @Test
    fun o3T5ZeroToOneNormalization() {
        val tensor = ImagePreprocessor.preprocess(
            imageData = createSolidColorPNG(width = 8, height = 8, r = 0, g = 0, b = 0),
            targetWidth = 8,
            targetHeight = 8,
            resize = "stretch",
            normalization = "zero_to_1",
            customMean = null,
            customStd = null,
        )

        assertEquals(0.0, value(tensor, channel = 0, x = 2, y = 2), 0.0001)
        assertEquals(0.0, value(tensor, channel = 1, x = 2, y = 2), 0.0001)
        assertEquals(0.0, value(tensor, channel = 2, x = 2, y = 2), 0.0001)
    }

    @Test
    fun o3T6NoneNormalizationPreservesByteValues() {
        val tensor = ImagePreprocessor.preprocess(
            imageData = createSolidColorPNG(width = 8, height = 8, r = 255, g = 0, b = 0),
            targetWidth = 8,
            targetHeight = 8,
            resize = "stretch",
            normalization = "none",
            customMean = null,
            customStd = null,
        )

        assertEquals(255.0, value(tensor, channel = 0, x = 6, y = 6), 0.0001)
        assertEquals(0.0, value(tensor, channel = 1, x = 6, y = 6), 0.0001)
        assertEquals(0.0, value(tensor, channel = 2, x = 6, y = 6), 0.0001)
    }

    @Test
    fun o3T7InvalidImageDataThrowsPreprocessError() {
        try {
            ImagePreprocessor.preprocess(
                imageData = "not-an-image".toByteArray(),
                targetWidth = 8,
                targetHeight = 8,
                resize = "stretch",
                normalization = "imagenet",
                customMean = null,
                customStd = null,
            )
            fail("Expected PreprocessError")
        } catch (error: ONNXError) {
            val preprocessError = error as? ONNXError.PreprocessError
            if (preprocessError == null) {
                fail("Expected PreprocessError, got $error")
                return
            }
            assertEquals("Unable to decode image data", preprocessError.detail)
        }
    }

    @Test
    fun o3T8CustomMeanAndStdOverrideNormalization() {
        val tensor = ImagePreprocessor.preprocess(
            imageData = createSolidColorPNG(width = 8, height = 8, r = 128, g = 128, b = 128),
            targetWidth = 8,
            targetHeight = 8,
            resize = "stretch",
            normalization = "imagenet",
            customMean = listOf(0.5, 0.5, 0.5),
            customStd = listOf(0.5, 0.5, 0.5),
        )

        assertEquals(0.0039215686, value(tensor, channel = 0, x = 1, y = 1), 0.001)
        assertEquals(0.0039215686, value(tensor, channel = 1, x = 1, y = 1), 0.001)
        assertEquals(0.0039215686, value(tensor, channel = 2, x = 1, y = 1), 0.001)
    }

    private fun value(tensor: TensorData, channel: Int, x: Int, y: Int): Double {
        val width = tensor.shape[3]
        val height = tensor.shape[2]
        val planeSize = width * height
        val index = (channel * planeSize) + (y * width) + x
        return tensor.data[index]
    }

    private fun createSolidColorPNG(width: Int, height: Int, r: Int, g: Int, b: Int): ByteArray {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.eraseColor(Color.rgb(r, g, b))

        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
        bitmap.recycle()
        return stream.toByteArray()
    }
}
