package io.t6x.dust.onnx

interface ONNXEngine {
    val inputMetadata: List<ONNXTensorMetadata>
    val outputMetadata: List<ONNXTensorMetadata>
    val accelerator: String

    fun run(inputs: Map<String, TensorData>): Map<String, TensorData>
    fun close()
}
