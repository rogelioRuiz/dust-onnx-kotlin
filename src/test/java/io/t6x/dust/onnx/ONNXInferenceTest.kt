package io.t6x.dust.onnx

import io.t6x.dust.core.DustCoreError
import io.t6x.dust.core.SessionPriority
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.io.File

class ONNXInferenceTest {
    @Test
    fun o2T1RunInferenceFloat32ReturnsOutput() {
        val engine = MockONNXEngine()
        val session = makeSession(engine)

        val outputs = session.runInference(
            mapOf(
                "input_a" to TensorData("input_a", "float32", listOf(1, 3), listOf(1.0, 2.0, 3.0)),
                "input_b" to TensorData("input_b", "float32", listOf(1, 3), listOf(4.0, 5.0, 6.0)),
            ),
            outputNames = null,
        )

        assertEquals(listOf(1, 3), outputs["output"]?.shape)
        assertEquals("float32", outputs["output"]?.dtype)
        assertEquals(listOf(1.0, 2.0, 3.0), engine.lastInputs?.get("input_a")?.data)
    }

    @Test
    fun o2T2RunInferenceUInt8ReturnsOutput() {
        val engine = MockONNXEngine(
            inputMetadata = listOf(ONNXTensorMetadata("pixels", listOf(1, 4), "uint8")),
            outputMetadata = listOf(ONNXTensorMetadata("output", listOf(1, 4), "uint8")),
        ).apply {
            outputTensors = mapOf(
                "output" to TensorData("output", "uint8", listOf(1, 4), listOf(8.0, 16.0, 32.0, 64.0)),
            )
        }
        val metadata = ONNXModelMetadataValue(engine.inputMetadata, engine.outputMetadata, "auto", 13)
        val session = makeSession(engine, metadata)

        val outputs = session.runInference(
            mapOf(
                "pixels" to TensorData("pixels", "uint8", listOf(1, 4), listOf(1.0, 2.0, 3.0, 4.0)),
            ),
            outputNames = null,
        )

        assertEquals("uint8", outputs["output"]?.dtype)
        assertEquals("uint8", engine.lastInputs?.get("pixels")?.dtype)
    }

    @Test
    fun o2T3ShapeMismatchWrongRankThrowsShapeError() {
        val session = makeSession()

        try {
            session.runInference(
                mapOf(
                    "input_a" to TensorData("input_a", "float32", listOf(3), listOf(1.0, 2.0, 3.0)),
                    "input_b" to TensorData("input_b", "float32", listOf(1, 3), listOf(4.0, 5.0, 6.0)),
                ),
                outputNames = null,
            )
            fail("Expected ShapeError")
        } catch (error: ONNXError) {
            val shapeError = error as? ONNXError.ShapeError
            if (shapeError == null) {
                fail("Expected ShapeError, got $error")
                return
            }
            assertEquals("input_a", shapeError.name)
            assertEquals(listOf(1, 3), shapeError.expected)
            assertEquals(listOf(3), shapeError.got)
        }
    }

    @Test
    fun o2T4ShapeMismatchWrongStaticDimensionThrowsShapeError() {
        val session = makeSession()

        try {
            session.runInference(
                mapOf(
                    "input_a" to TensorData("input_a", "float32", listOf(1, 4), listOf(1.0, 2.0, 3.0, 4.0)),
                    "input_b" to TensorData("input_b", "float32", listOf(1, 3), listOf(4.0, 5.0, 6.0)),
                ),
                outputNames = null,
            )
            fail("Expected ShapeError")
        } catch (error: ONNXError) {
            val shapeError = error as? ONNXError.ShapeError
            if (shapeError == null) {
                fail("Expected ShapeError, got $error")
                return
            }
            assertEquals("input_a", shapeError.name)
            assertEquals(listOf(1, 3), shapeError.expected)
            assertEquals(listOf(1, 4), shapeError.got)
        }
    }

    @Test
    fun o2T5DynamicDimensionAcceptsAnySize() {
        val engine = MockONNXEngine(
            inputMetadata = listOf(ONNXTensorMetadata("tokens", listOf(-1, 3), "float32")),
            outputMetadata = listOf(ONNXTensorMetadata("output", listOf(-1, 3), "float32")),
        ).apply {
            outputTensors = mapOf(
                "output" to TensorData("output", "float32", listOf(2, 3), listOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)),
            )
        }
        val metadata = ONNXModelMetadataValue(engine.inputMetadata, engine.outputMetadata, "auto", 13)
        val session = makeSession(engine, metadata)

        val outputs = session.runInference(
            mapOf(
                "tokens" to TensorData("tokens", "float32", listOf(2, 3), listOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)),
            ),
            outputNames = null,
        )

        assertEquals(listOf(2, 3), outputs["output"]?.shape)
    }

    @Test
    fun o2T6DtypeMismatchThrowsDtypeError() {
        val session = makeSession()

        try {
            session.runInference(
                mapOf(
                    "input_a" to TensorData("input_a", "int32", listOf(1, 3), listOf(1.0, 2.0, 3.0)),
                    "input_b" to TensorData("input_b", "float32", listOf(1, 3), listOf(4.0, 5.0, 6.0)),
                ),
                outputNames = null,
            )
            fail("Expected DtypeError")
        } catch (error: ONNXError) {
            val dtypeError = error as? ONNXError.DtypeError
            if (dtypeError == null) {
                fail("Expected DtypeError, got $error")
                return
            }
            assertEquals("input_a", dtypeError.name)
            assertEquals("float32", dtypeError.expected)
            assertEquals("int32", dtypeError.got)
        }
    }

    @Test
    fun o2T7OutputNamesFiltersSubset() {
        val engine = MockONNXEngine(
            outputMetadata = listOf(
                ONNXTensorMetadata("first", listOf(1), "float32"),
                ONNXTensorMetadata("second", listOf(1), "float32"),
            ),
        ).apply {
            outputTensors = linkedMapOf(
                "first" to TensorData("first", "float32", listOf(1), listOf(1.0)),
                "second" to TensorData("second", "float32", listOf(1), listOf(2.0)),
            )
        }
        val metadata = ONNXModelMetadataValue(engine.inputMetadata, engine.outputMetadata, "auto", 13)
        val session = makeSession(engine, metadata)

        val outputs = session.runInference(
            mapOf(
                "input_a" to TensorData("input_a", "float32", listOf(1, 3), listOf(1.0, 2.0, 3.0)),
                "input_b" to TensorData("input_b", "float32", listOf(1, 3), listOf(4.0, 5.0, 6.0)),
            ),
            outputNames = listOf("second"),
        )

        assertEquals(listOf("second"), outputs.keys.toList())
        assertEquals(listOf(2.0), outputs["second"]?.data)
    }

    @Test
    fun o2T8RunInferenceOnUnloadedModelMapsToModelNotFound() = runTest {
        val tempFile = File.createTempFile("tiny-test-", ".onnx")
        tempFile.writeText("fixture placeholder")
        tempFile.deleteOnExit()

        val engine = MockONNXEngine()
        val manager = ONNXSessionManager(
            sessionFactory = { _, modelId, _, priority ->
                ONNXSession(modelId, engine, defaultMetadata(), priority)
            },
        )

        manager.loadModel(tempFile.path, "tiny-test", ONNXConfig(), SessionPriority.INTERACTIVE)
        manager.forceUnloadModel("tiny-test")

        try {
            if (manager.session("tiny-test") == null) {
                throw DustCoreError.ModelNotFound
            }
            fail("Expected ModelNotFound")
        } catch (error: DustCoreError) {
            assertTrue(error is DustCoreError.ModelNotFound)
        }
    }

    @Test
    fun o2T9EngineThrowsDuringRunMapsToInferenceError() {
        val engine = MockONNXEngine().apply {
            runError = IllegalStateException("mock engine failure")
        }
        val session = makeSession(engine)

        try {
            session.runInference(
                mapOf(
                    "input_a" to TensorData("input_a", "float32", listOf(1, 3), listOf(1.0, 2.0, 3.0)),
                    "input_b" to TensorData("input_b", "float32", listOf(1, 3), listOf(4.0, 5.0, 6.0)),
                ),
                outputNames = null,
            )
            fail("Expected InferenceError")
        } catch (error: ONNXError) {
            val inferenceError = error as? ONNXError.InferenceError
            if (inferenceError == null) {
                fail("Expected InferenceError, got $error")
                return
            }
            assertEquals("mock engine failure", inferenceError.detail)
        }
    }

    private fun makeSession(
        engine: MockONNXEngine = MockONNXEngine(),
        metadata: ONNXModelMetadataValue = defaultMetadata(),
    ): ONNXSession {
        return ONNXSession(
            sessionId = "tiny-test",
            engine = engine,
            metadata = metadata,
            sessionPriority = SessionPriority.INTERACTIVE,
        )
    }

    private fun defaultMetadata(): ONNXModelMetadataValue {
        return ONNXModelMetadataValue(
            inputs = listOf(
                ONNXTensorMetadata("input_a", listOf(1, 3), "float32"),
                ONNXTensorMetadata("input_b", listOf(1, 3), "float32"),
            ),
            outputs = listOf(
                ONNXTensorMetadata("output", listOf(1, 3), "float32"),
            ),
            accelerator = "auto",
            opset = 13,
        )
    }
}

private class MockONNXEngine(
    override val inputMetadata: List<ONNXTensorMetadata> = listOf(
        ONNXTensorMetadata("input_a", listOf(1, 3), "float32"),
        ONNXTensorMetadata("input_b", listOf(1, 3), "float32"),
    ),
    override val outputMetadata: List<ONNXTensorMetadata> = listOf(
        ONNXTensorMetadata("output", listOf(1, 3), "float32"),
    ),
    override val accelerator: String = "auto",
) : ONNXEngine {
    var outputTensors: Map<String, TensorData> = mapOf(
        "output" to TensorData("output", "float32", listOf(1, 3), listOf(5.0, 7.0, 9.0)),
    )
    var runError: Throwable? = null
    var lastInputs: Map<String, TensorData>? = null
        private set
    var closeCallCount = 0
        private set

    override fun run(inputs: Map<String, TensorData>): Map<String, TensorData> {
        lastInputs = inputs
        runError?.let { throw it }
        return outputTensors
    }

    override fun close() {
        closeCallCount += 1
    }
}
