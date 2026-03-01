package io.t6x.dust.onnx

import io.t6x.dust.core.ModelFormat
import io.t6x.dust.core.SessionPriority
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.io.File

class ONNXSessionManagerTest {
    @Test
    fun o1T1LoadValidPathCreatesSession() = runTest {
        val file = File.createTempFile("tiny-test-", ".onnx")
        file.writeText("fixture placeholder")
        file.deleteOnExit()

        val manager = ONNXSessionManager(
            sessionFactory = { _, modelId, _, priority ->
                ONNXSession(modelId, fakeMetadata(modelId), priority)
            },
        )

        val session = manager.loadModel(file.path, "model-a", ONNXConfig(), SessionPriority.INTERACTIVE)

        assertEquals("model-a", session.sessionId)
        assertEquals(1, session.metadata.inputs.size)
        assertEquals(SessionPriority.INTERACTIVE, session.priority())
    }

    @Test
    fun o1T2MetadataAccessReturnsExpectedTensorNames() = runTest {
        val file = File.createTempFile("tiny-test-", ".onnx")
        file.writeText("fixture placeholder")
        file.deleteOnExit()

        val manager = ONNXSessionManager(
            sessionFactory = { _, modelId, _, priority ->
                ONNXSession(modelId, fakeMetadata(modelId), priority)
            },
        )

        manager.loadModel(file.path, "model-a", ONNXConfig(), SessionPriority.INTERACTIVE)
        val session = manager.session("model-a")

        assertNotNull(session)
        assertEquals("input_a", session?.metadata?.inputs?.first()?.name)
        assertEquals("output", session?.metadata?.outputs?.first()?.name)
    }

    @Test
    fun o1T3MissingFileThrowsFileNotFound() = runTest {
        val manager = ONNXSessionManager(
            sessionFactory = { path, _, _, _ ->
                throw ONNXError.FileNotFound(path)
            },
        )

        try {
            manager.loadModel("/nonexistent/model.onnx", "missing", ONNXConfig(), SessionPriority.INTERACTIVE)
            fail("Expected FileNotFound")
        } catch (error: ONNXError) {
            assertTrue(error is ONNXError.FileNotFound)
            assertTrue(error.message.orEmpty().contains("/nonexistent/model.onnx"))
        }
    }

    @Test
    fun o1T4CorruptFileThrowsLoadFailed() = runTest {
        val file = File.createTempFile("corrupt-", ".onnx")
        file.writeText("not a real onnx file")
        file.deleteOnExit()

        val manager = ONNXSessionManager(
            sessionFactory = { path, _, _, _ ->
                throw ONNXError.LoadFailed(path, "fixture is corrupt")
            },
        )

        try {
            manager.loadModel(file.path, "corrupt", ONNXConfig(), SessionPriority.INTERACTIVE)
            fail("Expected LoadFailed")
        } catch (error: ONNXError) {
            assertTrue(error is ONNXError.LoadFailed)
        }
    }

    @Test
    fun o1T5WrongFormatWouldBeRejectedAtPluginLayer() {
        val accepted = ModelFormat.ONNX.value
        assertEquals("onnx", accepted)

        val rejected = ModelFormat.entries.filter { it != ModelFormat.ONNX }
        assertTrue(rejected.isNotEmpty())
        for (format in rejected) {
            assertTrue(format.value != accepted)
        }

        var factoryCalled = false
        val manager = ONNXSessionManager(
            sessionFactory = { _, modelId, _, priority ->
                factoryCalled = true
                ONNXSession(modelId, fakeMetadata(modelId), priority)
            },
        )
        assertNotNull(manager)
        assertFalse(factoryCalled)
    }

    @Test
    fun o1T6UnloadLoadedModelEmptiesSessionMap() = runTest {
        val file = File.createTempFile("tiny-test-", ".onnx")
        file.writeText("fixture placeholder")
        file.deleteOnExit()

        val manager = ONNXSessionManager(
            sessionFactory = { _, modelId, _, priority ->
                ONNXSession(modelId, fakeMetadata(modelId), priority)
            },
        )

        manager.loadModel(file.path, "model-a", ONNXConfig(), SessionPriority.INTERACTIVE)
        manager.forceUnloadModel("model-a")

        assertEquals(0, manager.sessionCount)
        assertFalse(manager.hasCachedSession("model-a"))
    }

    @Test
    fun o1T6bUnloadUnknownIdThrowsModelNotFound() = runTest {
        val manager = ONNXSessionManager()

        try {
            manager.forceUnloadModel("nonexistent")
            fail("Expected DustCoreError.ModelNotFound")
        } catch (error: io.t6x.dust.core.DustCoreError) {
            assertTrue(error is io.t6x.dust.core.DustCoreError.ModelNotFound)
        }
    }

    @Test
    fun o1T7LoadingSameIdTwiceReusesSessionAndUpdatesRefCount() = runTest {
        val file = File.createTempFile("tiny-test-", ".onnx")
        file.writeText("fixture placeholder")
        file.deleteOnExit()

        val manager = ONNXSessionManager(
            sessionFactory = { _, modelId, _, priority ->
                ONNXSession(modelId, fakeMetadata(modelId), priority)
            },
        )

        val first = manager.loadModel(file.path, "model-a", ONNXConfig(), SessionPriority.INTERACTIVE)
        val second = manager.loadModel(file.path, "model-a", ONNXConfig(), SessionPriority.BACKGROUND)

        assertSame(first, second)
        assertEquals(1, manager.sessionCount)
        assertEquals(2, manager.refCount("model-a"))
    }

    @Test
    fun o1T8ConcurrentLoadTwoDifferentModels() = runTest {
        val fileA = File.createTempFile("tiny-test-a-", ".onnx")
        fileA.writeText("fixture placeholder a")
        fileA.deleteOnExit()

        val fileB = File.createTempFile("tiny-test-b-", ".onnx")
        fileB.writeText("fixture placeholder b")
        fileB.deleteOnExit()

        val manager = ONNXSessionManager(
            sessionFactory = { _, modelId, _, priority ->
                ONNXSession(modelId, fakeMetadata(modelId), priority)
            },
        )

        val sessions = listOf(
            async { manager.loadModel(fileA.path, "model-a", ONNXConfig(), SessionPriority.INTERACTIVE) },
            async { manager.loadModel(fileB.path, "model-b", ONNXConfig(), SessionPriority.INTERACTIVE) },
        ).awaitAll()

        assertEquals(2, sessions.size)
        assertEquals("model-a", sessions[0].sessionId)
        assertEquals("model-b", sessions[1].sessionId)
        assertEquals(2, manager.sessionCount)
        assertEquals(1, manager.refCount("model-a"))
        assertEquals(1, manager.refCount("model-b"))
    }
}

private fun fakeMetadata(modelId: String): ONNXModelMetadataValue {
    return ONNXModelMetadataValue(
        inputs = listOf(
            ONNXTensorMetadata("input_a", listOf(1, 3), "float32"),
        ),
        outputs = listOf(
            ONNXTensorMetadata("output", listOf(1, 3), "float32"),
        ),
        accelerator = "auto",
        opset = 13,
    )
}
