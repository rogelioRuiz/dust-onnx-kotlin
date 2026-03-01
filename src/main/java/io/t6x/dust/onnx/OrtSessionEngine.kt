package io.t6x.dust.onnx

import ai.onnxruntime.NodeInfo
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxValue
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.nio.ByteBuffer
import java.nio.DoubleBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.nio.ShortBuffer

class OrtSessionEngine(
    private val environment: OrtEnvironment,
    private val session: OrtSession,
    override val accelerator: String,
) : ONNXEngine {
    override val inputMetadata: List<ONNXTensorMetadata> = readTensorMetadata(session.inputInfo)
    override val outputMetadata: List<ONNXTensorMetadata> = readTensorMetadata(session.outputInfo)

    override fun run(inputs: Map<String, TensorData>): Map<String, TensorData> {
        val ortInputs = linkedMapOf<String, OnnxTensor>()

        try {
            for ((name, tensor) in inputs) {
                ortInputs[name] = createOnnxTensor(tensor)
            }

            session.run(ortInputs).use { result ->
                val outputs = linkedMapOf<String, TensorData>()
                for ((name, value) in result) {
                    val tensor = value as? OnnxTensor
                        ?: throw ONNXError.InferenceError("Only tensor outputs are supported")
                    outputs[name] = readTensorData(name, tensor)
                }
                return outputs
            }
        } finally {
            OnnxValue.close(ortInputs)
        }
    }

    override fun close() {
        session.close()
    }

    private fun createOnnxTensor(tensor: TensorData): OnnxTensor {
        val shape = tensor.shape.map(Int::toLong).toLongArray()

        return when (tensor.dtype) {
            "float16" -> OnnxTensor.createTensor(
                environment,
                ShortBuffer.wrap(tensor.data.map { floatToHalfBits(it.toFloat()) }.toShortArray()),
                shape,
                OnnxJavaType.FLOAT16,
            )
            "float32" -> OnnxTensor.createTensor(
                environment,
                FloatBuffer.wrap(tensor.data.map(Double::toFloat).toFloatArray()),
                shape,
            )
            "float64" -> OnnxTensor.createTensor(
                environment,
                DoubleBuffer.wrap(tensor.data.toDoubleArray()),
                shape,
            )
            "int8" -> OnnxTensor.createTensor(
                environment,
                ByteBuffer.wrap(tensor.data.map { it.toLong().toByte() }.toByteArray()),
                shape,
                OnnxJavaType.INT8,
            )
            "int16" -> OnnxTensor.createTensor(
                environment,
                ShortBuffer.wrap(tensor.data.map { it.toLong().toShort() }.toShortArray()),
                shape,
                OnnxJavaType.INT16,
            )
            "int32" -> OnnxTensor.createTensor(
                environment,
                IntBuffer.wrap(tensor.data.map { it.toLong().toInt() }.toIntArray()),
                shape,
            )
            "int64" -> OnnxTensor.createTensor(
                environment,
                LongBuffer.wrap(tensor.data.map { it.toLong() }.toLongArray()),
                shape,
            )
            "uint8" -> OnnxTensor.createTensor(
                environment,
                ByteBuffer.wrap(tensor.data.map { it.toLong().toInt().coerceIn(0, 255).toByte() }.toByteArray()),
                shape,
                OnnxJavaType.UINT8,
            )
            "bool" -> OnnxTensor.createTensor(
                environment,
                ByteBuffer.wrap(tensor.data.map { if (it == 0.0) 0.toByte() else 1.toByte() }.toByteArray()),
                shape,
                OnnxJavaType.BOOL,
            )
            else -> throw ONNXError.InferenceError("Unsupported tensor dtype: ${tensor.dtype}")
        }
    }

    private fun readTensorData(
        name: String,
        tensor: OnnxTensor,
    ): TensorData {
        val info = tensor.info
        val dtype = mapOrtType(info.type)
        val shape = info.shape.map(Long::toInt)

        val data = when (info.type) {
            OnnxJavaType.FLOAT -> {
                val buffer = tensor.floatBuffer.duplicate()
                val values = FloatArray(buffer.remaining())
                buffer.get(values)
                values.map(Float::toDouble)
            }
            OnnxJavaType.DOUBLE -> {
                val buffer = tensor.doubleBuffer.duplicate()
                val values = DoubleArray(buffer.remaining())
                buffer.get(values)
                values.toList()
            }
            OnnxJavaType.FLOAT16 -> {
                val buffer = tensor.shortBuffer.duplicate()
                val values = ShortArray(buffer.remaining())
                buffer.get(values)
                values.map { halfBitsToFloat(it).toDouble() }
            }
            OnnxJavaType.INT8 -> {
                val buffer = tensor.byteBuffer.duplicate()
                val values = ByteArray(buffer.remaining())
                buffer.get(values)
                values.map(Byte::toDouble)
            }
            OnnxJavaType.INT16 -> {
                val buffer = tensor.shortBuffer.duplicate()
                val values = ShortArray(buffer.remaining())
                buffer.get(values)
                values.map(Short::toDouble)
            }
            OnnxJavaType.INT32 -> {
                val buffer = tensor.intBuffer.duplicate()
                val values = IntArray(buffer.remaining())
                buffer.get(values)
                values.map(Int::toDouble)
            }
            OnnxJavaType.INT64 -> {
                val buffer = tensor.longBuffer.duplicate()
                val values = LongArray(buffer.remaining())
                buffer.get(values)
                values.map(Long::toDouble)
            }
            OnnxJavaType.UINT8 -> {
                val buffer = tensor.byteBuffer.duplicate()
                val values = ByteArray(buffer.remaining())
                buffer.get(values)
                values.map { (it.toInt() and 0xFF).toDouble() }
            }
            OnnxJavaType.BOOL -> {
                val buffer = tensor.byteBuffer.duplicate()
                val values = ByteArray(buffer.remaining())
                buffer.get(values)
                values.map { if (it.toInt() == 0) 0.0 else 1.0 }
            }
            else -> throw ONNXError.InferenceError("Unsupported output tensor dtype: ${info.type}")
        }

        return TensorData(name = name, dtype = dtype, shape = shape, data = data)
    }

    companion object {
        fun readMetadata(
            session: OrtSession,
            accelerator: String,
        ): ONNXModelMetadataValue {
            return ONNXModelMetadataValue(
                inputs = readTensorMetadata(session.inputInfo),
                outputs = readTensorMetadata(session.outputInfo),
                accelerator = accelerator,
                opset = null,
            )
        }

        fun readTensorMetadata(entries: Map<String, NodeInfo>): List<ONNXTensorMetadata> {
            return entries.entries.map { (name, nodeInfo) ->
                val info = nodeInfo.info
                if (info is TensorInfo) {
                    ONNXTensorMetadata(
                        name = name,
                        shape = info.shape.map(Long::toInt),
                        dtype = mapOrtType(info.type),
                    )
                } else {
                    ONNXTensorMetadata(
                        name = name,
                        shape = emptyList(),
                        dtype = "unknown",
                    )
                }
            }
        }

        fun mapOrtType(type: OnnxJavaType): String = when (type) {
            OnnxJavaType.FLOAT -> "float32"
            OnnxJavaType.DOUBLE -> "float64"
            OnnxJavaType.FLOAT16 -> "float16"
            OnnxJavaType.INT8 -> "int8"
            OnnxJavaType.INT16 -> "int16"
            OnnxJavaType.INT32 -> "int32"
            OnnxJavaType.INT64 -> "int64"
            OnnxJavaType.UINT8 -> "uint8"
            OnnxJavaType.BOOL -> "bool"
            OnnxJavaType.STRING -> "string"
            else -> "unknown"
        }

        private fun floatToHalfBits(value: Float): Short {
            val bits = value.toRawBits()
            val sign = (bits ushr 16) and 0x8000
            var exponent = ((bits ushr 23) and 0xff) - 127 + 15
            var mantissa = bits and 0x7fffff

            if (exponent <= 0) {
                if (exponent < -10) {
                    return sign.toShort()
                }

                mantissa = (mantissa or 0x800000) shr (1 - exponent)
                return (sign or ((mantissa + 0x1000) shr 13)).toShort()
            }

            if (exponent >= 31) {
                return (sign or 0x7c00).toShort()
            }

            return (sign or (exponent shl 10) or ((mantissa + 0x1000) shr 13)).toShort()
        }

        private fun halfBitsToFloat(value: Short): Float {
            val bits = value.toInt() and 0xffff
            val sign = (bits and 0x8000) shl 16
            var exponent = (bits ushr 10) and 0x1f
            var mantissa = bits and 0x03ff

            val floatBits = when {
                exponent == 0 -> {
                    if (mantissa == 0) {
                        sign
                    } else {
                        exponent = 1
                        while ((mantissa and 0x0400) == 0) {
                            mantissa = mantissa shl 1
                            exponent -= 1
                        }
                        mantissa = mantissa and 0x03ff
                        sign or ((exponent + 127 - 15) shl 23) or (mantissa shl 13)
                    }
                }
                exponent == 0x1f -> sign or 0x7f800000 or (mantissa shl 13)
                else -> sign or ((exponent + 127 - 15) shl 23) or (mantissa shl 13)
            }

            return Float.fromBits(floatBits)
        }
    }
}
