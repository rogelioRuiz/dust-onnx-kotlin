package io.t6x.dust.onnx

import io.t6x.dust.core.SessionPriority
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class ONNXPipelineTest {
    @Test
    fun o6T1TwoStepPipelineBothResultsReturned() {
        val engine = ScriptedMockONNXEngine().apply {
            scriptedOutputs = listOf(
                mapOf("output" to TensorData("output", "float32", listOf(1, 3), listOf(1.0, 2.0, 3.0))),
                mapOf("output" to TensorData("output", "float32", listOf(1, 2), listOf(4.0, 5.0))),
            )
        }
        val session = makeSession(engine)

        val results = session.runPipeline(
            listOf(
                literalStep(data = listOf(1.0, 2.0, 3.0)),
                literalStep(data = listOf(9.0, 8.0, 7.0)),
            ),
        )

        assertEquals(2, results.size)
        assertEquals(listOf(1, 3), results[0]["output"]?.shape)
        assertEquals(listOf(1, 2), results[1]["output"]?.shape)
        assertEquals(2, engine.callCount)
    }

    @Test
    fun o6T2PreviousOutputChainingSubstitutesCorrectly() {
        val engine = ScriptedMockONNXEngine(
            inputMetadata = listOf(ONNXTensorMetadata("output", listOf(1, 3), "float32")),
        ).apply {
            scriptedOutputs = listOf(
                mapOf("output" to TensorData("output", "float32", listOf(1, 3), listOf(9.0, 8.0, 7.0))),
                mapOf("output" to TensorData("output", "float32", listOf(1, 3), listOf(1.0, 1.0, 1.0))),
            )
        }
        val session = makeSession(engine, metadataFor(engine))

        session.runPipeline(
            listOf(
                literalStep(inputName = "output", data = listOf(1.0, 2.0, 3.0)),
                PipelineStep(inputs = listOf(PipelineInputValue.PreviousOutput("output")), outputNames = null),
            ),
        )

        assertEquals(listOf(9.0, 8.0, 7.0), engine.allInputs[1]["output"]?.data)
    }

    @Test
    fun o6T3ExplicitFromStepChainingRoutesCorrectTensor() {
        val engine = ScriptedMockONNXEngine().apply {
            scriptedOutputs = listOf(
                mapOf("output" to TensorData("output", "float32", listOf(1, 3), listOf(3.0, 2.0, 1.0))),
                mapOf("output" to TensorData("output", "float32", listOf(1, 3), listOf(0.0, 0.0, 0.0))),
            )
        }
        val session = makeSession(engine)

        session.runPipeline(
            listOf(
                literalStep(data = listOf(1.0, 2.0, 3.0)),
                PipelineStep(
                    inputs = listOf(
                        PipelineInputValue.StepReference(
                            name = "input",
                            fromStep = 0,
                            outputName = "output",
                        ),
                    ),
                    outputNames = null,
                ),
            ),
        )

        assertEquals(listOf(3.0, 2.0, 1.0), engine.allInputs[1]["input"]?.data)
    }

    @Test
    fun o6T4Step1FailsPipelineHaltsWithStepIndex0() {
        val engine = ScriptedMockONNXEngine().apply {
            scriptedErrors = listOf(RuntimeException("step 0 boom"))
        }
        val session = makeSession(engine)

        try {
            session.runPipeline(listOf(literalStep(data = listOf(1.0, 2.0, 3.0))))
            fail("Expected InferenceError")
        } catch (error: ONNXError) {
            val inferenceError = error as? ONNXError.InferenceError
            if (inferenceError == null) {
                fail("Expected InferenceError, got $error")
                return
            }
            assertTrue(inferenceError.detail.contains("step 0"))
            assertEquals(1, engine.callCount)
        }
    }

    @Test
    fun o6T5Step2FailsErrorReportsStepIndex1() {
        val engine = ScriptedMockONNXEngine().apply {
            scriptedErrors = listOf(null, RuntimeException("step 1 boom"))
        }
        val session = makeSession(engine)

        try {
            session.runPipeline(
                listOf(
                    literalStep(data = listOf(1.0, 2.0, 3.0)),
                    literalStep(data = listOf(4.0, 5.0, 6.0)),
                ),
            )
            fail("Expected InferenceError")
        } catch (error: ONNXError) {
            val inferenceError = error as? ONNXError.InferenceError
            if (inferenceError == null) {
                fail("Expected InferenceError, got $error")
                return
            }
            assertTrue(inferenceError.detail.contains("step 1"))
            assertEquals(2, engine.callCount)
        }
    }

    @Test
    fun o6T6SingleStepPipelineMatchesRunInference() {
        val expectedOutput = mapOf(
            "output" to TensorData("output", "float32", listOf(1, 3), listOf(5.0, 7.0, 9.0)),
        )
        val pipelineSession = makeSession(
            ScriptedMockONNXEngine().apply {
                scriptedOutputs = listOf(expectedOutput)
            },
        )
        val directSession = makeSession(
            ScriptedMockONNXEngine().apply {
                scriptedOutputs = listOf(expectedOutput)
            },
        )
        val inputs = mapOf(
            "input" to TensorData("input", "float32", listOf(1, 3), listOf(1.0, 2.0, 3.0)),
        )

        val pipelineOutputs = pipelineSession.runPipeline(
            listOf(
                PipelineStep(
                    inputs = listOf(PipelineInputValue.Literal(inputs.getValue("input"))),
                    outputNames = null,
                ),
            ),
        ).single()
        val directOutputs = directSession.runInference(inputs, outputNames = null)

        assertEquals(directOutputs, pipelineOutputs)
    }

    @Test
    fun o6T7PipelineOnEvictedSessionThrowsAtFirstStep() {
        val engine = ScriptedMockONNXEngine()
        val session = makeSession(engine)
        session.evict()

        try {
            session.runPipeline(listOf(literalStep(data = listOf(1.0, 2.0, 3.0))))
            fail("Expected ModelEvicted")
        } catch (error: ONNXError) {
            assertTrue(error is ONNXError.ModelEvicted)
            assertEquals(0, engine.callCount)
        }
    }

    private fun makeSession(
        engine: ScriptedMockONNXEngine = ScriptedMockONNXEngine(),
        metadata: ONNXModelMetadataValue? = null,
    ): ONNXSession {
        return ONNXSession(
            sessionId = "tiny-test",
            engine = engine,
            metadata = metadata ?: metadataFor(engine),
            sessionPriority = SessionPriority.INTERACTIVE,
        )
    }

    private fun metadataFor(engine: ScriptedMockONNXEngine): ONNXModelMetadataValue {
        return ONNXModelMetadataValue(
            inputs = engine.inputMetadata,
            outputs = engine.outputMetadata,
            accelerator = "auto",
            opset = 13,
        )
    }

    private fun literalStep(
        inputName: String = "input",
        data: List<Double>,
        shape: List<Int>? = null,
        outputNames: List<String>? = null,
    ): PipelineStep {
        return PipelineStep(
            inputs = listOf(
                PipelineInputValue.Literal(
                    TensorData(
                        name = inputName,
                        dtype = "float32",
                        shape = shape ?: listOf(1, data.size),
                        data = data,
                    ),
                ),
            ),
            outputNames = outputNames,
        )
    }
}

private class ScriptedMockONNXEngine(
    override val inputMetadata: List<ONNXTensorMetadata> = listOf(
        ONNXTensorMetadata("input", listOf(1, 3), "float32"),
    ),
    override val outputMetadata: List<ONNXTensorMetadata> = listOf(
        ONNXTensorMetadata("output", listOf(1, 3), "float32"),
    ),
    override val accelerator: String = "auto",
) : ONNXEngine {
    private val defaultOutput = mapOf(
        "output" to TensorData("output", "float32", listOf(1, 3), listOf(5.0, 7.0, 9.0)),
    )

    var scriptedOutputs: List<Map<String, TensorData>> = emptyList()
    var scriptedErrors: List<Throwable?> = emptyList()
    val allInputs = mutableListOf<Map<String, TensorData>>()
    var callCount = 0
        private set

    override fun run(inputs: Map<String, TensorData>): Map<String, TensorData> {
        val index = callCount++
        allInputs.add(inputs)
        scriptedErrors.getOrNull(index)?.let { throw it }
        return scriptedOutputs.getOrElse(index) { defaultOutput }
    }

    override fun close() = Unit
}
