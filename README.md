<p align="center">
  <img alt="dust" src="assets/dust_banner.png" width="400">
</p>

<p align="center">
  <strong>Device Unified Serving Toolkit</strong><br>
  <a href="https://github.com/rogelioRuiz/dust">dust ecosystem</a> · v0.1.0 · Apache 2.0
</p>

<p align="center">
  <a href="https://github.com/rogelioRuiz/dust/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <img alt="Version" src="https://img.shields.io/badge/version-0.1.0-informational">
  <img alt="Maven" src="https://img.shields.io/badge/Maven-io.t6x.dust%3Adust--onnx-blue">
  <a href="https://developer.android.com/studio/releases/platforms"><img alt="API" src="https://img.shields.io/badge/API-26+-green.svg"></a>
  <a href="https://kotlinlang.org"><img alt="Kotlin" src="https://img.shields.io/badge/Kotlin-2.1-purple.svg"></a>
  <img alt="ONNX" src="https://img.shields.io/badge/ONNX_Runtime-1.20-005CED">
</p>

---

<p align="center">
<strong>dust ecosystem</strong> —
<a href="../capacitor-core/README.md">capacitor-core</a> ·
<a href="../capacitor-llm/README.md">capacitor-llm</a> ·
<a href="../capacitor-onnx/README.md">capacitor-onnx</a> ·
<a href="../capacitor-serve/README.md">capacitor-serve</a> ·
<a href="../capacitor-embeddings/README.md">capacitor-embeddings</a>
<br>
<a href="../dust-core-kotlin/README.md">dust-core-kotlin</a> ·
<a href="../dust-llm-kotlin/README.md">dust-llm-kotlin</a> ·
<strong>dust-onnx-kotlin</strong> ·
<a href="../dust-embeddings-kotlin/README.md">dust-embeddings-kotlin</a> ·
<a href="../dust-serve-kotlin/README.md">dust-serve-kotlin</a>
<br>
<a href="../dust-core-swift/README.md">dust-core-swift</a> ·
<a href="../dust-llm-swift/README.md">dust-llm-swift</a> ·
<a href="../dust-onnx-swift/README.md">dust-onnx-swift</a> ·
<a href="../dust-embeddings-swift/README.md">dust-embeddings-swift</a> ·
<a href="../dust-serve-swift/README.md">dust-serve-swift</a>
</p>

---

# dust-onnx-kotlin

Android ONNX runtime session management and preprocessing for on-device inference.

**Version: 0.1.0**

## Overview

`dust-onnx-kotlin` wraps the ONNX Runtime Android SDK behind the [dust-core-kotlin](../dust-core-kotlin) contract interfaces. It handles session lifecycle, image preprocessing, hardware accelerator selection, and inference pipeline orchestration:

- **ONNXSession** — load/close ONNX models with configurable execution providers
- **ONNXInferenceEngine** — run inference with automatic input/output tensor mapping
- **ImagePreprocessor** — resize, normalize, and convert images to float tensors
- **AcceleratorSelector** — pick NNAPI, GPU, or CPU based on device capabilities
- **ONNXRegistry** — thread-safe model registration and lookup
- **ONNXPipeline** — chain preprocessing, inference, and postprocessing steps

## Architecture

```
src/main/kotlin/io/t6x/dust/onnx/
├── ONNXSession.kt
├── ONNXInferenceEngine.kt
├── ImagePreprocessor.kt
├── AcceleratorSelector.kt
├── ONNXRegistry.kt
└── ONNXPipeline.kt
```

## Install

### Gradle — local project dependency

```groovy
// settings.gradle
include ':dust-onnx-kotlin'
project(':dust-onnx-kotlin').projectDir = new File('../dust-onnx-kotlin')

// Also include the contract library
include ':dust-core-kotlin'
project(':dust-core-kotlin').projectDir = new File('../dust-core-kotlin')

// build.gradle
dependencies {
    implementation project(':dust-onnx-kotlin')
}
```

### Gradle — Maven (when published)

```groovy
dependencies {
    implementation 'io.t6x.dust:dust-onnx:0.1.0'
    // transitive: com.microsoft.onnxruntime:onnxruntime-android:1.20.0
}
```

## Usage

```kotlin
import io.t6x.dust.onnx.*

// 1. Select accelerator
val accelerator = AcceleratorSelector.best(context)

// 2. Open a session
val session = ONNXSession(modelPath = "/data/model.onnx", accelerator = accelerator)

// 3. Preprocess an image
val tensor = ImagePreprocessor.bitmapToTensor(bitmap, width = 224, height = 224)

// 4. Run inference
val engine = ONNXInferenceEngine(session)
val outputs = engine.run(listOf(tensor))

// 5. Clean up
session.close()
```

## Test

```bash
./gradlew test    # 51 JUnit tests (6 suites)
```

| Suite | Tests | Coverage |
|-------|-------|----------|
| `AcceleratorSelectorTest` | 9 | Accelerator ranking, fallback, caching |
| `ONNXInferenceEngineTest` | 9 | Tensor mapping, error handling, multi-output |
| `ImagePreprocessorTest` | 9 | Resize, normalize, color conversion |
| `ONNXRegistryTest` | 9 | Register/resolve, thread safety |
| `ONNXSessionTest` | 9 | Load/close lifecycle, provider config |
| `ONNXPipelineTest` | 6 | End-to-end pipeline, step chaining |

No emulator needed — all tests run on the JVM with mocks.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding conventions, and PR guidelines.

## License

Copyright 2026 Rogelio Ruiz Perez. Licensed under the [Apache License 2.0](LICENSE).
