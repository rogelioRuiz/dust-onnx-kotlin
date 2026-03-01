package io.t6x.dust.onnx

import io.t6x.dust.core.DustCoreError
import io.t6x.dust.core.DustCoreRegistry
import io.t6x.dust.core.ModelDescriptor
import io.t6x.dust.core.ModelFormat
import io.t6x.dust.core.ModelStatus
import io.t6x.dust.core.SessionPriority
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import java.io.File

class ONNXRegistryTest {

    @Before
    fun setUp() {
        DustCoreRegistry.resetForTesting()
    }

    @After
    fun tearDown() {
        DustCoreRegistry.resetForTesting()
    }

    @Test
    fun o5T1RegistryRegistrationMakesManagerResolvable() {
        val manager = makeManager()

        DustCoreRegistry.getInstance().registerModelServer(manager)

        assertSame(manager, DustCoreRegistry.getInstance().resolveModelServer())
    }

    @Test
    fun o5T2LoadModelForReadyDescriptorCreatesSessionAndRefCount() = runTest {
        val file = makeTempModelFile()
        val manager = makeManager()
        val descriptor = makeDescriptor("model-a", file.path)
        manager.register(descriptor)

        val session = manager.loadModel(descriptor, SessionPriority.INTERACTIVE)

        assertEquals(ModelStatus.Ready, session.status())
        assertEquals(1, manager.refCount("model-a"))

        file.delete()
    }

    @Test
    fun o5T3LoadModelForNotLoadedDescriptorThrowsModelNotReady() = runTest {
        val manager = makeManager()
        val descriptor = makeDescriptor("model-a", "/tmp/missing-model.onnx")
        manager.register(descriptor)

        try {
            manager.loadModel(descriptor, SessionPriority.INTERACTIVE)
            fail("Expected ModelNotReady")
        } catch (error: DustCoreError) {
            assertTrue(error is DustCoreError.ModelNotReady)
        }
    }

    @Test
    fun o5T4LoadModelForUnregisteredIdThrowsModelNotFound() = runTest {
        val manager = makeManager()
        val descriptor = makeDescriptor("ghost", "/tmp/ghost.onnx")

        try {
            manager.loadModel(descriptor, SessionPriority.INTERACTIVE)
            fail("Expected ModelNotFound")
        } catch (error: DustCoreError) {
            assertTrue(error is DustCoreError.ModelNotFound)
        }
    }

    @Test
    fun o5T5UnloadModelDecrementsRefCountAndKeepsSessionCached() = runTest {
        val file = makeTempModelFile()
        val manager = makeManager()
        val descriptor = makeDescriptor("model-a", file.path)
        manager.register(descriptor)

        manager.loadModel(descriptor, SessionPriority.INTERACTIVE)
        manager.unloadModel("model-a")

        assertEquals(0, manager.refCount("model-a"))
        assertTrue(manager.hasCachedSession("model-a"))

        file.delete()
    }

    @Test
    fun o5T6LoadModelTwiceReusesSameSessionAndIncrementsRefCount() = runTest {
        val file = makeTempModelFile()
        val manager = makeManager()
        val descriptor = makeDescriptor("model-a", file.path)
        manager.register(descriptor)

        val first = manager.loadModel(descriptor, SessionPriority.INTERACTIVE)
        val second = manager.loadModel(descriptor, SessionPriority.BACKGROUND)

        assertSame(first, second)
        assertEquals(2, manager.refCount("model-a"))

        file.delete()
    }

    @Test
    fun o5T7EvictUnderPressureStandardRemovesBackgroundZeroRefSessions() = runTest {
        val fileA = makeTempModelFile()
        val fileB = makeTempModelFile()
        val manager = makeManager()
        val descriptorA = makeDescriptor("model-a", fileA.path)
        val descriptorB = makeDescriptor("model-b", fileB.path)
        manager.register(descriptorA)
        manager.register(descriptorB)

        val sessionA = manager.loadModel(descriptorA, SessionPriority.BACKGROUND) as ONNXSession
        val sessionB = manager.loadModel(descriptorB, SessionPriority.INTERACTIVE) as ONNXSession

        manager.unloadModel("model-a")
        manager.unloadModel("model-b")
        manager.evictUnderPressure(MemoryPressureLevel.STANDARD)

        assertTrue(sessionA.isModelEvicted)
        assertFalse(manager.hasCachedSession("model-a"))
        assertFalse(sessionB.isModelEvicted)
        assertTrue(manager.hasCachedSession("model-b"))

        fileA.delete()
        fileB.delete()
    }

    @Test
    fun o5T8EvictUnderPressureCriticalRemovesAllZeroRefSessions() = runTest {
        val file = makeTempModelFile()
        val manager = makeManager()
        val descriptor = makeDescriptor("model-a", file.path)
        manager.register(descriptor)

        val session = manager.loadModel(descriptor, SessionPriority.INTERACTIVE) as ONNXSession
        manager.unloadModel("model-a")
        manager.evictUnderPressure(MemoryPressureLevel.CRITICAL)

        assertTrue(session.isModelEvicted)
        assertFalse(manager.hasCachedSession("model-a"))

        file.delete()
    }

    @Test
    fun o5T9AllModelIdsReturnsOnlyLiveSessionsAfterEviction() = runTest {
        val fileA = makeTempModelFile()
        val fileB = makeTempModelFile()
        val manager = makeManager()
        val descriptorA = makeDescriptor("model-a", fileA.path)
        val descriptorB = makeDescriptor("model-b", fileB.path)
        manager.register(descriptorA)
        manager.register(descriptorB)

        manager.loadModel(descriptorA, SessionPriority.INTERACTIVE)
        manager.loadModel(descriptorB, SessionPriority.INTERACTIVE)

        manager.unloadModel("model-a")
        manager.evict("model-a")

        assertEquals(listOf("model-b"), manager.allModelIds())

        fileA.delete()
        fileB.delete()
    }

    private fun makeManager(): ONNXSessionManager {
        return ONNXSessionManager(
            sessionFactory = { _, modelId, _, priority ->
                ONNXSession(modelId, registryMetadata(), priority)
            },
        )
    }

    private fun makeDescriptor(id: String, path: String): ModelDescriptor {
        return ModelDescriptor(
            id = id,
            name = id,
            format = ModelFormat.ONNX,
            sizeBytes = 1,
            version = "1.0.0",
            metadata = mapOf("localPath" to path),
        )
    }

    private fun makeTempModelFile(): File {
        val file = File.createTempFile("onnx-registry-", ".onnx")
        file.writeBytes(byteArrayOf(0x08, 0x01, 0x12, 0x00))
        file.deleteOnExit()
        return file
    }
}

private fun registryMetadata(): ONNXModelMetadataValue {
    return ONNXModelMetadataValue(
        inputs = listOf(
            ONNXTensorMetadata("input", listOf(1, 3), "float32"),
        ),
        outputs = listOf(
            ONNXTensorMetadata("output", listOf(1, 3), "float32"),
        ),
        accelerator = "cpu",
        opset = 13,
    )
}
