package io.t6x.dust.onnx

import org.json.JSONArray
import org.json.JSONObject
import io.t6x.dust.core.DustInputTensor
import io.t6x.dust.core.DustOutputTensor
import io.t6x.dust.core.ModelSession
import io.t6x.dust.core.ModelStatus
import io.t6x.dust.core.SessionPriority
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

data class ONNXTensorMetadata(
    val name: String,
    val shape: List<Int>,
    val dtype: String,
) {
    fun toJSONObject(): JSONObject {
        val shapeArray = JSONArray()
        for (dimension in shape) {
            shapeArray.put(dimension)
        }

        return JSONObject()
            .put("name", name)
            .put("shape", shapeArray)
            .put("dtype", dtype)
    }
}

data class ONNXModelMetadataValue(
    val inputs: List<ONNXTensorMetadata>,
    val outputs: List<ONNXTensorMetadata>,
    val accelerator: String,
    val opset: Int? = null,
) {
    fun toJSONObject(): JSONObject {
        val inputArray = JSONArray()
        val outputArray = JSONArray()

        for (input in inputs) {
            inputArray.put(input.toJSONObject())
        }
        for (output in outputs) {
            outputArray.put(output.toJSONObject())
        }

        return JSONObject()
            .put("inputs", inputArray)
            .put("outputs", outputArray)
            .put("accelerator", accelerator)
            .apply {
                opset?.let { put("opset", it) }
            }
    }
}

data class TensorData(
    val name: String,
    val dtype: String,
    val shape: List<Int>,
    val data: List<Double>,
) {
    fun toJSONObject(): JSONObject {
        val shapeArray = JSONArray()
        val dataArray = JSONArray()

        for (dimension in shape) {
            shapeArray.put(dimension)
        }
        for (value in data) {
            dataArray.put(value)
        }

        return JSONObject()
            .put("name", name)
            .put("dtype", dtype)
            .put("shape", shapeArray)
            .put("data", dataArray)
    }
}

sealed class PipelineInputValue {
    abstract val name: String

    data class Literal(val tensor: TensorData) : PipelineInputValue() {
        override val name: String
            get() = tensor.name
    }

    data class PreviousOutput(override val name: String) : PipelineInputValue()

    data class StepReference(
        override val name: String,
        val fromStep: Int,
        val outputName: String,
    ) : PipelineInputValue()
}

data class PipelineStep(
    val inputs: List<PipelineInputValue>,
    val outputNames: List<String>?,
)

class ONNXSession(
    val sessionId: String,
    private var engine: ONNXEngine?,
    val metadata: ONNXModelMetadataValue,
    private val sessionPriority: SessionPriority,
) : ModelSession {
    private val lock = ReentrantLock()
    private var currentStatus: ModelStatus = ModelStatus.Ready
    private var evicted = false

    constructor(
        sessionId: String,
        metadata: ONNXModelMetadataValue,
        priority: SessionPriority,
    ) : this(sessionId, null, metadata, priority)

    override suspend fun predict(inputs: List<DustInputTensor>): List<DustOutputTensor> {
        val tensors = inputs.associate { input ->
            input.name to TensorData(
                name = input.name,
                dtype = "float32",
                shape = input.shape,
                data = input.data.map { it.toDouble() },
            )
        }
        val outputs = runInference(tensors, outputNames = null)

        return metadata.outputs.mapNotNull { outputMetadata ->
            outputs[outputMetadata.name]?.let { tensor ->
                DustOutputTensor(
                    name = tensor.name,
                    data = tensor.data.map { it.toFloat() },
                    shape = tensor.shape,
                )
            }
        }
    }

    fun runInference(
        inputs: Map<String, TensorData>,
        outputNames: List<String>?,
    ): Map<String, TensorData> {
        val activeEngine = requireEngine()
        validate(inputs, activeEngine.inputMetadata)

        val rawOutputs = try {
            activeEngine.run(inputs)
        } catch (error: ONNXError) {
            when (error) {
                is ONNXError.InferenceError -> throw error
                else -> throw ONNXError.InferenceError(error.message ?: error.toString())
            }
        } catch (error: Throwable) {
            throw ONNXError.InferenceError(error.message ?: error.toString())
        }

        return filterOutputs(rawOutputs, outputNames)
    }

    fun runPipeline(steps: List<PipelineStep>): List<Map<String, TensorData>> {
        val activeEngine = requireEngine()
        val resolvedResults = arrayOfNulls<Map<String, TensorData>>(steps.size)
        val cachedResults = arrayOfNulls<Map<String, TensorData>>(steps.size)

        for ((stepIndex, step) in steps.withIndex()) {
            val resolvedInputs = linkedMapOf<String, TensorData>()
            for (inputValue in step.inputs) {
                val tensor = when (inputValue) {
                    is PipelineInputValue.Literal -> inputValue.tensor
                    is PipelineInputValue.PreviousOutput -> {
                        val previousIndex = stepIndex - 1
                        if (previousIndex < 0) {
                            throw pipelineResolutionError(
                                stepIndex,
                                "previous_output requires a previous step",
                            )
                        }

                        val previousOutputs = cachedResults[previousIndex]
                            ?: throw pipelineResolutionError(
                                stepIndex,
                                "previous step $previousIndex outputs are unavailable",
                            )
                        val previousTensor = previousOutputs[inputValue.name]
                            ?: throw pipelineResolutionError(
                                stepIndex,
                                "previous step output '${inputValue.name}' was not found",
                            )
                        previousTensor.copy(name = inputValue.name)
                    }
                    is PipelineInputValue.StepReference -> {
                        if (inputValue.fromStep !in 0 until stepIndex) {
                            throw pipelineResolutionError(
                                stepIndex,
                                "fromStep ${inputValue.fromStep} must reference an earlier step",
                            )
                        }

                        val referencedOutputs = cachedResults[inputValue.fromStep]
                            ?: throw pipelineResolutionError(
                                stepIndex,
                                "step ${inputValue.fromStep} outputs are unavailable",
                            )
                        val referencedTensor = referencedOutputs[inputValue.outputName]
                            ?: throw pipelineResolutionError(
                                stepIndex,
                                "step ${inputValue.fromStep} output '${inputValue.outputName}' was not found",
                            )
                        referencedTensor.copy(name = inputValue.name)
                    }
                }
                resolvedInputs[inputValue.name] = tensor
            }

            val filteredOutputs = try {
                validate(resolvedInputs, activeEngine.inputMetadata)
                filterOutputs(activeEngine.run(resolvedInputs), step.outputNames)
            } catch (error: ONNXError) {
                throw ONNXError.InferenceError(
                    "Pipeline step $stepIndex failed: ${error.message ?: error.toString()}",
                )
            } catch (error: Throwable) {
                throw ONNXError.InferenceError(
                    "Pipeline step $stepIndex failed: ${error.message ?: error.toString()}",
                )
            }

            resolvedResults[stepIndex] = filteredOutputs
            cachedResults[stepIndex] = filteredOutputs

            if (stepIndex > 0) {
                val previousIndex = stepIndex - 1
                val referencedLater = (stepIndex + 1 until steps.size).any { futureIndex ->
                    steps[futureIndex].inputs.any { input ->
                        input is PipelineInputValue.StepReference && input.fromStep == previousIndex
                    }
                }
                if (!referencedLater) {
                    cachedResults[previousIndex] = null
                }
            }
        }

        return resolvedResults.map { it ?: emptyMap() }
    }

    override fun status(): ModelStatus = lock.withLock { currentStatus }

    val isModelEvicted: Boolean
        get() = lock.withLock { evicted }

    override fun priority(): SessionPriority = sessionPriority

    override suspend fun close() {
        closeContext(evicted = false)
    }

    fun closeContext() {
        closeContext(evicted = false)
    }

    fun evict() {
        closeContext(evicted = true)
    }

    private fun closeContext(evicted: Boolean) {
        val activeEngine = lock.withLock {
            val current = engine
            engine = null
            currentStatus = ModelStatus.NotLoaded
            this.evicted = evicted
            current
        }
        activeEngine?.close()
    }

    fun requireEngine(): ONNXEngine = lock.withLock {
        engine ?: throw if (evicted) {
            ONNXError.ModelEvicted
        } else {
            ONNXError.SessionClosed
        }
    }

    private fun validate(
        inputs: Map<String, TensorData>,
        metadata: List<ONNXTensorMetadata>,
    ) {
        val metadataByName = metadata.associateBy { it.name }

        for ((name, tensor) in inputs) {
            val expected = metadataByName[name] ?: continue

            if (expected.dtype != "unknown" && expected.dtype != tensor.dtype) {
                throw ONNXError.DtypeError(name, expected.dtype, tensor.dtype)
            }

            if (expected.shape.size != tensor.shape.size) {
                throw ONNXError.ShapeError(name, expected.shape, tensor.shape)
            }

            for ((expectedDimension, actualDimension) in expected.shape.zip(tensor.shape)) {
                if (expectedDimension != -1 && expectedDimension != actualDimension) {
                    throw ONNXError.ShapeError(name, expected.shape, tensor.shape)
                }
            }
        }
    }

    private fun filterOutputs(
        outputs: Map<String, TensorData>,
        outputNames: List<String>?,
    ): Map<String, TensorData> {
        if (outputNames == null) {
            return outputs
        }

        val filtered = linkedMapOf<String, TensorData>()
        for (outputName in outputNames) {
            val tensor = outputs[outputName] ?: continue
            filtered[outputName] = tensor
        }
        return filtered
    }

    private fun pipelineResolutionError(
        stepIndex: Int,
        detail: String,
    ): ONNXError.InferenceError {
        return ONNXError.InferenceError("Pipeline step $stepIndex: $detail")
    }
}
