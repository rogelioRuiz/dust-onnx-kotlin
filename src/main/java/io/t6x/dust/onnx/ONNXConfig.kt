package io.t6x.dust.onnx

data class ONNXConfig(
    val accelerator: String = "auto",
    val interOpNumThreads: Int = 1,
    val intraOpNumThreads: Int = maxOf(1, Runtime.getRuntime().availableProcessors() - 1),
    val graphOptimizationLevel: String = "all",
    val enableMemoryPattern: Boolean = true,
)
