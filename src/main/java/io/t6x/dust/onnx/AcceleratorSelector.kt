package io.t6x.dust.onnx

import ai.onnxruntime.OrtSession

data class AcceleratorResult(
    val configureOptions: (OrtSession.SessionOptions) -> Unit,
    val resolvedAccelerator: String,
)

object AcceleratorSelector {
    fun select(accelerator: String): AcceleratorResult = when (accelerator.lowercase()) {
        "auto",
        "nnapi",
        -> AcceleratorResult(
            configureOptions = { options -> options.addNnapi() },
            resolvedAccelerator = "nnapi",
        )
        "xnnpack" -> AcceleratorResult(
            configureOptions = { options -> options.addXnnpack(emptyMap<String, String>()) },
            resolvedAccelerator = "xnnpack",
        )
        else -> AcceleratorResult(
            configureOptions = {},
            resolvedAccelerator = "cpu",
        )
    }
}
