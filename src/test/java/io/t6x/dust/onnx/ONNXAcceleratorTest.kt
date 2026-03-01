package io.t6x.dust.onnx

import io.t6x.dust.core.SessionPriority
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.io.File

class ONNXAcceleratorTest {
    @Test
    fun o4T1SelectAutoPrefersNnapi() {
        val result = AcceleratorSelector.select("auto")

        assertEquals("nnapi", result.resolvedAccelerator)
    }

    @Test
    fun o4T2SelectCpuStaysOnCpu() {
        val result = AcceleratorSelector.select("cpu")

        assertEquals("cpu", result.resolvedAccelerator)
    }

    @Test
    fun o4T3SelectExplicitNnapiUsesNnapi() {
        val result = AcceleratorSelector.select("nnapi")

        assertEquals("nnapi", result.resolvedAccelerator)
    }

    @Test
    fun o4T4SelectXnnpackUsesXnnpack() {
        val result = AcceleratorSelector.select("xnnpack")

        assertEquals("xnnpack", result.resolvedAccelerator)
    }

    @Test
    fun o4T5SelectCoremlFallsBackToCpuOnAndroid() {
        val result = AcceleratorSelector.select("coreml")

        assertEquals("cpu", result.resolvedAccelerator)
    }

    @Test
    fun o4T6LoadRetriesOnAcceleratorFailureAndFallsBackToCpu() = runTest {
        val file = tempModelFile()
        var callCount = 0
        val manager = ONNXSessionManager(
            defaultSessionFactory = { _, modelId, _, priority, acceleratorResult ->
                callCount += 1
                if (callCount == 1) {
                    assertEquals("nnapi", acceleratorResult.resolvedAccelerator)
                    throw IllegalStateException("NNAPI unavailable")
                }

                assertEquals("cpu", acceleratorResult.resolvedAccelerator)
                ONNXSession(modelId, fakeMetadata(acceleratorResult.resolvedAccelerator), priority)
            },
        )

        val session = manager.loadModel(
            file.path,
            "fallback-model",
            ONNXConfig(accelerator = "auto"),
            SessionPriority.INTERACTIVE,
        )

        assertEquals(2, callCount)
        assertEquals("cpu", session.metadata.accelerator)
    }

    @Test
    fun o4T7CpuConfigurationLoadsWithoutRetry() = runTest {
        val file = tempModelFile()
        var callCount = 0
        val manager = ONNXSessionManager(
            defaultSessionFactory = { _, modelId, _, priority, acceleratorResult ->
                callCount += 1
                assertEquals("cpu", acceleratorResult.resolvedAccelerator)
                ONNXSession(modelId, fakeMetadata(acceleratorResult.resolvedAccelerator), priority)
            },
        )

        val session = manager.loadModel(
            file.path,
            "cpu-model",
            ONNXConfig(accelerator = "cpu"),
            SessionPriority.INTERACTIVE,
        )

        assertEquals(1, callCount)
        assertEquals("cpu", session.metadata.accelerator)
    }

    @Test
    fun o4T8FallbackFailureReturnsLoadFailed() = runTest {
        val file = tempModelFile()
        var callCount = 0
        val manager = ONNXSessionManager(
            defaultSessionFactory = { _, _, _, _, _ ->
                callCount += 1
                throw IllegalStateException("session creation failed")
            },
        )

        try {
            manager.loadModel(
                file.path,
                "failing-model",
                ONNXConfig(accelerator = "nnapi"),
                SessionPriority.INTERACTIVE,
            )
            fail("Expected LoadFailed")
        } catch (error: ONNXError) {
            assertTrue(error is ONNXError.LoadFailed)
            assertEquals(2, callCount)
        }
    }

    @Test
    fun o4T9SessionMetadataUsesResolvedAccelerator() = runTest {
        val file = tempModelFile()
        val manager = ONNXSessionManager(
            defaultSessionFactory = { _, modelId, _, priority, acceleratorResult ->
                ONNXSession(modelId, fakeMetadata(acceleratorResult.resolvedAccelerator), priority)
            },
        )

        val session = manager.loadModel(
            file.path,
            "nnapi-model",
            ONNXConfig(accelerator = "nnapi"),
            SessionPriority.INTERACTIVE,
        )

        assertEquals("nnapi", session.metadata.accelerator)
        assertEquals("nnapi", manager.session("nnapi-model")?.metadata?.accelerator)
    }
}

private fun tempModelFile(): File {
    return File.createTempFile("tiny-test-", ".onnx").apply {
        writeText("fixture placeholder")
        deleteOnExit()
    }
}

private fun fakeMetadata(accelerator: String): ONNXModelMetadataValue {
    return ONNXModelMetadataValue(
        inputs = listOf(
            ONNXTensorMetadata("input_a", listOf(1, 3), "float32"),
        ),
        outputs = listOf(
            ONNXTensorMetadata("output", listOf(1, 3), "float32"),
        ),
        accelerator = accelerator,
        opset = 13,
    )
}
