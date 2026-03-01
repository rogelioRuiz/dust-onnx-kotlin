package io.t6x.dust.onnx

sealed class ONNXError(message: String? = null) : Exception(message) {
    data class FileNotFound(val path: String) : ONNXError("Model file not found: $path")
    data class LoadFailed(val path: String, val detail: String? = null) : ONNXError(
        buildString {
            append("Failed to load ONNX model: ")
            append(path)
            if (!detail.isNullOrBlank()) {
                append(" (")
                append(detail)
                append(')')
            }
        },
    )
    data class FormatUnsupported(val format: String) : ONNXError("Unsupported model format: $format")
    data object SessionClosed : ONNXError("Session is closed")
    data object ModelEvicted : ONNXError("Model was evicted from memory")
    data class ShapeError(val name: String, val expected: List<Int>, val got: List<Int>) : ONNXError(
        "Shape mismatch for $name: expected $expected, got $got",
    )
    data class DtypeError(val name: String, val expected: String, val got: String) : ONNXError(
        "Dtype mismatch for $name: expected $expected, got $got",
    )
    data class InferenceError(val detail: String) : ONNXError(detail)
    data class PreprocessError(val detail: String) : ONNXError("Preprocessing failed: $detail")

    fun code(): String = when (this) {
        is FileNotFound, is LoadFailed, is InferenceError -> "inferenceFailed"
        is FormatUnsupported -> "formatUnsupported"
        is SessionClosed -> "sessionClosed"
        is ModelEvicted -> "modelEvicted"
        is ShapeError -> "shapeError"
        is DtypeError -> "dtypeError"
        is PreprocessError -> "preprocessError"
    }
}
