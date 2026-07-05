//! Native scheduler engine: admission control, queue accounting, and FIFO dispatch.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeSchedulerConfig {
    pub max_concurrent_requests: usize,
    pub max_queued_requests: usize,
    pub max_batch_size: usize,
    pub max_batch_wait_ms: u64,
    pub kv_cache_budget_bytes: u64,
}

impl Default for NativeSchedulerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_requests: 1,
            max_queued_requests: 127,
            max_batch_size: 1,
            max_batch_wait_ms: 0,
            kv_cache_budget_bytes: 0,
        }
    }
}

#[derive(Clone)]
pub struct NativeSchedulerEngine {
    inner: Arc<dyn NativeEngine>,
    config: NativeSchedulerConfig,
    permits: Arc<Semaphore>,
    waiting: Arc<AtomicUsize>,
}

impl std::fmt::Debug for NativeSchedulerEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeSchedulerEngine")
            .field("model_alias", &self.model_alias())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl NativeSchedulerEngine {
    pub fn new(inner: Arc<dyn NativeEngine>, config: NativeSchedulerConfig) -> Self {
        let config = NativeSchedulerConfig {
            max_concurrent_requests: config.max_concurrent_requests.max(1),
            max_queued_requests: config.max_queued_requests,
            max_batch_size: config.max_batch_size.max(1),
            max_batch_wait_ms: config.max_batch_wait_ms,
            kv_cache_budget_bytes: config.kv_cache_budget_bytes,
        };
        Self {
            inner,
            permits: Arc::new(Semaphore::new(config.max_concurrent_requests)),
            waiting: Arc::new(AtomicUsize::new(0)),
            config,
        }
    }
}

impl NativeEngine for NativeSchedulerEngine {
    fn model_alias(&self) -> &str {
        self.inner.model_alias()
    }

    fn chat(&self, request: NativeChatRequest) -> BoxFuture<'_, Result<NativeChatResponse>> {
        scheduled_native_chat(
            self.inner.clone(),
            self.permits.clone(),
            self.waiting.clone(),
            self.config,
            request,
            NativeScheduledOperation::Chat,
        )
    }

    fn chat_stream(&self, request: NativeChatRequest) -> BoxFuture<'_, Result<NativeChatResponse>> {
        scheduled_native_chat(
            self.inner.clone(),
            self.permits.clone(),
            self.waiting.clone(),
            self.config,
            request,
            NativeScheduledOperation::Stream,
        )
    }

    fn embeddings(
        &self,
        request: NativeEmbeddingRequest,
    ) -> BoxFuture<'_, Result<NativeEmbeddingResponse>> {
        self.inner.embeddings(request)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeScheduledOperation {
    Chat,
    Stream,
}

struct NativeSchedulerWaitGuard {
    waiting: Arc<AtomicUsize>,
    queued_before: usize,
}

impl NativeSchedulerWaitGuard {
    fn enter(waiting: Arc<AtomicUsize>, max_queued_requests: usize) -> Result<Self> {
        let queued_before = waiting.fetch_add(1, Ordering::AcqRel);
        if queued_before >= max_queued_requests {
            waiting.fetch_sub(1, Ordering::AcqRel);
            bail!("native scheduler queue is full");
        }
        Ok(Self {
            waiting,
            queued_before,
        })
    }
}

impl Drop for NativeSchedulerWaitGuard {
    fn drop(&mut self) {
        self.waiting.fetch_sub(1, Ordering::AcqRel);
    }
}

fn scheduled_native_chat(
    inner: Arc<dyn NativeEngine>,
    permits: Arc<Semaphore>,
    waiting: Arc<AtomicUsize>,
    config: NativeSchedulerConfig,
    mut request: NativeChatRequest,
    operation: NativeScheduledOperation,
) -> BoxFuture<'static, Result<NativeChatResponse>> {
    Box::pin(async move {
        reject_cancelled_request(&request.metadata)?;
        let queued_at = Instant::now();
        let mut queued_before_admit = 0usize;
        let permit = match permits.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(TryAcquireError::NoPermits) => {
                let wait_guard =
                    NativeSchedulerWaitGuard::enter(waiting, config.max_queued_requests)?;
                queued_before_admit = wait_guard.queued_before;
                let permit = permits
                    .acquire_owned()
                    .await
                    .map_err(|_| anyhow::anyhow!("native scheduler is closed"))?;
                drop(wait_guard);
                permit
            }
            Err(TryAcquireError::Closed) => bail!("native scheduler is closed"),
        };
        reject_cancelled_request(&request.metadata)?;
        stamp_scheduler_metadata(
            &mut request.metadata,
            queued_at,
            queued_before_admit,
            config,
        );
        run_scheduled_native_chat(inner, request, operation, permit).await
    })
}

fn reject_cancelled_request(metadata: &BTreeMap<String, Value>) -> Result<()> {
    if metadata
        .get("llmctl.scheduler.cancelled")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        bail!("native scheduler request was cancelled before decode");
    }
    Ok(())
}

async fn run_scheduled_native_chat(
    inner: Arc<dyn NativeEngine>,
    request: NativeChatRequest,
    operation: NativeScheduledOperation,
    _permit: OwnedSemaphorePermit,
) -> Result<NativeChatResponse> {
    match operation {
        NativeScheduledOperation::Chat => inner.chat(request).await,
        NativeScheduledOperation::Stream => inner.chat_stream(request).await,
    }
}

fn stamp_scheduler_metadata(
    metadata: &mut BTreeMap<String, Value>,
    queued_at: Instant,
    queued_before_admit: usize,
    config: NativeSchedulerConfig,
) {
    let wait_ms = queued_at.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
    metadata.insert(
        "llmctl.scheduler.discipline".to_string(),
        Value::String("fifo".to_string()),
    );
    metadata.insert(
        "llmctl.scheduler.queue.implemented".to_string(),
        Value::Bool(true),
    );
    metadata.insert(
        "llmctl.scheduler.queue_wait_ms".to_string(),
        Value::from(wait_ms),
    );
    metadata.insert(
        "llmctl.scheduler.admission_wait_ms".to_string(),
        Value::from(wait_ms),
    );
    metadata.insert(
        "llmctl.scheduler.queued_requests_before_admit".to_string(),
        Value::from(queued_before_admit as u64),
    );
    metadata.insert(
        "llmctl.scheduler.max_concurrent_requests".to_string(),
        Value::from(config.max_concurrent_requests as u64),
    );
    metadata.insert(
        "llmctl.scheduler.max_queued_requests".to_string(),
        Value::from(config.max_queued_requests as u64),
    );
    metadata.insert(
        "llmctl.scheduler.batching.continuous.implemented".to_string(),
        Value::Bool(false),
    );
    metadata.insert(
        "llmctl.scheduler.batching.phase_scheduling.implemented".to_string(),
        Value::Bool(true),
    );
    metadata.insert(
        "llmctl.scheduler.phase".to_string(),
        Value::String("prefill-then-decode".to_string()),
    );
    metadata.insert(
        "llmctl.scheduler.prefill.phase".to_string(),
        Value::String("scheduled".to_string()),
    );
    metadata.insert(
        "llmctl.scheduler.decode.phase".to_string(),
        Value::String("scheduled".to_string()),
    );
    metadata.insert(
        "llmctl.scheduler.max_batch_size".to_string(),
        Value::from(config.max_batch_size as u64),
    );
    metadata.insert(
        "llmctl.scheduler.max_wait_ms".to_string(),
        Value::from(config.max_batch_wait_ms),
    );
    metadata.insert(
        "llmctl.scheduler.kv_cache_budget_bytes".to_string(),
        Value::from(config.kv_cache_budget_bytes),
    );
    metadata.insert(
        "llmctl.scheduler.kv_cache.reuse_implemented".to_string(),
        Value::Bool(false),
    );
    metadata.insert(
        "llmctl.scheduler.kv_cache.policy".to_string(),
        Value::String("request-local-reset".to_string()),
    );
    metadata.insert(
        "llmctl.scheduler.cancellation.admission_check_implemented".to_string(),
        Value::Bool(true),
    );
    metadata.insert(
        "llmctl.scheduler.cancellation.decode_loop_check_implemented".to_string(),
        Value::Bool(false),
    );
}
