package io.t6x.dust.onnx

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import io.t6x.dust.core.DustCoreError
import io.t6x.dust.core.ModelDescriptor
import io.t6x.dust.core.ModelFormat
import io.t6x.dust.core.ModelServer
import io.t6x.dust.core.ModelSession
import io.t6x.dust.core.ModelStatus
import io.t6x.dust.core.SessionPriority
import java.io.File
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

enum class MemoryPressureLevel {
    STANDARD,
    CRITICAL,
}

object ORTEnvironmentSingleton {
    val environment: OrtEnvironment by lazy { OrtEnvironment.getEnvironment() }
}

private typealias DefaultOrtSessionFactory = (
    path: String,
    modelId: String,
    config: ONNXConfig,
    priority: SessionPriority,
    acceleratorResult: AcceleratorResult,
) -> ONNXSession

class ONNXSessionManager(
    private val sessionFactory: ((path: String, modelId: String, config: ONNXConfig, priority: SessionPriority) -> ONNXSession)? = null,
    private val defaultSessionFactory: DefaultOrtSessionFactory? = null,
) : ModelServer {
    private val lock = ReentrantLock()
    private val descriptors = mutableMapOf<String, ModelDescriptor>()
    private val statuses = mutableMapOf<String, ModelStatus>()
    private val configs = mutableMapOf<String, ONNXConfig>()
    private val cachedSessions = mutableMapOf<String, CachedONNXSession>()

    fun register(descriptor: ModelDescriptor, config: ONNXConfig? = null) {
        val status = initialStatus(descriptor)

        lock.withLock {
            descriptors[descriptor.id] = descriptor
            statuses[descriptor.id] = status
            configs[descriptor.id] = config ?: configs[descriptor.id] ?: ONNXConfig()
        }
    }

    fun setStatus(status: ModelStatus, id: String) {
        lock.withLock {
            statuses[id] = status
        }
    }

    override suspend fun loadModel(descriptor: ModelDescriptor, priority: SessionPriority): ModelSession {
        val config = lock.withLock {
            if (!descriptors.containsKey(descriptor.id)) {
                throw DustCoreError.ModelNotFound
            }
            configs[descriptor.id] ?: ONNXConfig()
        }

        return loadModelWithConfig(descriptor, config, priority)
    }

    fun loadModelWithConfig(
        descriptor: ModelDescriptor,
        config: ONNXConfig,
        priority: SessionPriority,
    ): ONNXSession {
        incrementCachedRefCount(descriptor.id)?.let { return it }

        val (registeredDescriptor, status) = lock.withLock {
            descriptors[descriptor.id] to (statuses[descriptor.id] ?: ModelStatus.NotLoaded)
        }

        val storedDescriptor = registeredDescriptor ?: throw DustCoreError.ModelNotFound
        if (status != ModelStatus.Ready) {
            throw DustCoreError.ModelNotReady
        }

        val path = resolvedPath(storedDescriptor)
            ?: throw DustCoreError.InvalidInput("descriptor.url or descriptor.metadata.localPath is required")

        val created = createSession(path, descriptor.id, config, priority)

        val winner = lock.withLock {
            val existing = cachedSessions[descriptor.id]
            if (existing != null) {
                existing.refCount += 1
                existing.lastAccessTime = System.nanoTime()
                existing.session
            } else {
                cachedSessions[descriptor.id] = CachedONNXSession(
                    session = created,
                    priority = priority,
                    refCount = 1,
                    lastAccessTime = System.nanoTime(),
                )
                created
            }.also {
                statuses[descriptor.id] = ModelStatus.Ready
                configs[descriptor.id] = config
            }
        }

        if (winner !== created) {
            created.evict()
        }

        return winner
    }

    fun loadModel(
        path: String,
        modelId: String,
        config: ONNXConfig,
        priority: SessionPriority,
    ): ONNXSession {
        val descriptor = legacyDescriptor(path, modelId)
        register(descriptor, config)
        setStatus(ModelStatus.Ready, modelId)
        return loadModelWithConfig(descriptor, config, priority)
    }

    override suspend fun unloadModel(id: String) {
        val didDecrement = lock.withLock {
            val cached = cachedSessions[id]
            if (cached == null || cached.refCount == 0) {
                false
            } else {
                cached.refCount -= 1
                cached.lastAccessTime = System.nanoTime()
                true
            }
        }

        if (!didDecrement) {
            throw DustCoreError.ModelNotFound
        }
    }

    suspend fun forceUnloadModel(id: String) {
        val session = lock.withLock { cachedSessions.remove(id)?.session } ?: throw DustCoreError.ModelNotFound
        session.close()
    }

    suspend fun evict(modelId: String): ONNXSession? {
        val session = lock.withLock { cachedSessions.remove(modelId)?.session }
        session?.evict()
        return session
    }

    suspend fun evictUnderPressure(level: MemoryPressureLevel) {
        val evicted = lock.withLock {
            val eligible = cachedSessions.filter { (_, cached) ->
                cached.refCount == 0 && when (level) {
                    MemoryPressureLevel.STANDARD -> cached.priority == SessionPriority.BACKGROUND
                    MemoryPressureLevel.CRITICAL -> true
                }
            }
            val sorted = eligible.entries.sortedBy { it.value.lastAccessTime }
            val sessions = sorted.map { it.value.session }
            for ((id, _) in sorted) {
                cachedSessions.remove(id)
            }
            sessions
        }

        for (session in evicted) {
            session.evict()
        }
    }

    override suspend fun listModels(): List<ModelDescriptor> = allDescriptors()

    override suspend fun modelStatus(id: String): ModelStatus = lock.withLock {
        statuses[id] ?: ModelStatus.NotLoaded
    }

    fun refCount(id: String): Int = lock.withLock {
        cachedSessions[id]?.refCount ?: 0
    }

    fun hasCachedSession(id: String): Boolean = lock.withLock {
        cachedSessions.containsKey(id)
    }

    fun session(id: String): ONNXSession? = lock.withLock {
        cachedSessions[id]?.session
    }

    fun allModelIds(): List<String> = lock.withLock { cachedSessions.keys.sorted() }

    fun allDescriptors(): List<ModelDescriptor> = lock.withLock {
        descriptors.values.sortedBy { it.id }
    }

    val sessionCount: Int
        get() = lock.withLock { cachedSessions.size }

    private fun incrementCachedRefCount(id: String): ONNXSession? {
        return lock.withLock {
            val cached = cachedSessions[id] ?: return null
            cached.refCount += 1
            cached.lastAccessTime = System.nanoTime()
            cached.session
        }
    }

    private fun initialStatus(descriptor: ModelDescriptor): ModelStatus {
        val path = resolvedPath(descriptor) ?: return ModelStatus.NotLoaded
        return if (File(path).exists()) {
            ModelStatus.Ready
        } else {
            ModelStatus.NotLoaded
        }
    }

    private fun resolvedPath(descriptor: ModelDescriptor): String? {
        val localPath = descriptor.metadata?.get("localPath")
        if (!localPath.isNullOrEmpty()) {
            return localPath
        }

        if (!descriptor.url.isNullOrEmpty()) {
            return descriptor.url
        }

        return null
    }

    private fun legacyDescriptor(path: String, modelId: String): ModelDescriptor {
        val file = File(path)
        val sizeBytes = if (file.exists()) file.length() else 0L

        return ModelDescriptor(
            id = modelId,
            name = modelId,
            format = ModelFormat.ONNX,
            sizeBytes = sizeBytes,
            version = "legacy",
            url = path,
        )
    }

    private fun createSession(
        path: String,
        modelId: String,
        config: ONNXConfig,
        priority: SessionPriority,
    ): ONNXSession {
        sessionFactory?.let { factory ->
            return factory(path, modelId, config, priority)
        }

        if (!File(path).exists()) {
            throw ONNXError.FileNotFound(path)
        }

        val accelResult = AcceleratorSelector.select(config.accelerator)
        val factory = defaultSessionFactory ?: ::createOrtSession

        try {
            return attemptCreateSession(path, modelId, config, priority, accelResult, factory)
        } catch (error: ONNXError) {
            if (error !is ONNXError.LoadFailed || accelResult.resolvedAccelerator == "cpu") {
                throw error
            }
        }

        return attemptCreateSession(
            path = path,
            modelId = modelId,
            config = config,
            priority = priority,
            acceleratorResult = AcceleratorResult(configureOptions = {}, resolvedAccelerator = "cpu"),
            factory = factory,
        )
    }

    private fun resolveOptLevel(value: String): OrtSession.SessionOptions.OptLevel = when (value) {
        "disable" -> OrtSession.SessionOptions.OptLevel.NO_OPT
        "basic" -> OrtSession.SessionOptions.OptLevel.BASIC_OPT
        "extended" -> OrtSession.SessionOptions.OptLevel.EXTENDED_OPT
        else -> OrtSession.SessionOptions.OptLevel.ALL_OPT
    }

    private fun attemptCreateSession(
        path: String,
        modelId: String,
        config: ONNXConfig,
        priority: SessionPriority,
        acceleratorResult: AcceleratorResult,
        factory: DefaultOrtSessionFactory,
    ): ONNXSession {
        return try {
            factory(path, modelId, config, priority, acceleratorResult)
        } catch (error: ONNXError) {
            throw error
        } catch (error: Throwable) {
            throw ONNXError.LoadFailed(path, error.message)
        }
    }

    private fun createOrtSession(
        path: String,
        modelId: String,
        config: ONNXConfig,
        priority: SessionPriority,
        acceleratorResult: AcceleratorResult,
    ): ONNXSession {
        OrtSession.SessionOptions().use { options ->
            options.setInterOpNumThreads(config.interOpNumThreads)
            options.setIntraOpNumThreads(config.intraOpNumThreads)
            options.setOptimizationLevel(resolveOptLevel(config.graphOptimizationLevel))
            options.setMemoryPatternOptimization(config.enableMemoryPattern)
            acceleratorResult.configureOptions(options)

            val environment = ORTEnvironmentSingleton.environment
            val ortSession = environment.createSession(path, options)
            val resolvedAccelerator = acceleratorResult.resolvedAccelerator
            val metadata = OrtSessionEngine.readMetadata(ortSession, resolvedAccelerator)
            val engine = OrtSessionEngine(environment, ortSession, resolvedAccelerator)
            return ONNXSession(modelId, engine, metadata, priority)
        }
    }
}

private data class CachedONNXSession(
    val session: ONNXSession,
    val priority: SessionPriority,
    var refCount: Int,
    var lastAccessTime: Long,
)
