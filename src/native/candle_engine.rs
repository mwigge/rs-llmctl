//! Candle engine factory, load plan, and artifact-backed engine wiring.
use super::*;

#[derive(Debug, Clone)]
pub struct NativeCandleEngineFactory {
    registry: BTreeMap<CandleModelFamily, CandleFamilySupportMetadata>,
}

impl Default for NativeCandleEngineFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeCandleEngineFactory {
    pub fn new() -> Self {
        let registry = CandleModelFamily::all()
            .iter()
            .copied()
            .map(|family| (family, CandleFamilySupportMetadata::for_family(family)))
            .collect();
        Self { registry }
    }

    pub fn support_metadata(
        &self,
        family: CandleModelFamily,
    ) -> Option<&CandleFamilySupportMetadata> {
        self.registry.get(&family)
    }

    pub fn registered_families(&self) -> Vec<CandleModelFamily> {
        self.registry.keys().copied().collect()
    }

    pub fn plan(
        &self,
        family: CandleModelFamily,
        model: &ModelConfig,
        resources: &ResourceConfig,
    ) -> Result<NativeEngineLoadPlan> {
        let support = self.support_metadata(family).cloned().ok_or_else(|| {
            anyhow::anyhow!(
                "Candle model family '{}' is not registered",
                family.as_str()
            )
        })?;
        let format = NativeModelFormat::from_path(&model.path);
        let acceleration = NativeAcceleration::from_resources(resources);
        let candle = CandleEngineConfig::for_family(family, format, acceleration);
        let device_selection = candle.load_contract.device_selection.clone();

        let plan = NativeEngineLoadPlan {
            runtime: RuntimeBackend::CandleNative,
            engine: candle.engine.clone(),
            alias: model.alias.clone(),
            role: normalize_role(&model.role).to_string(),
            family: candle.load_contract.model_family.as_str().to_string(),
            format,
            acceleration,
            candle,
            support,
            device_selection,
            scheduler: NativeSchedulerContract::fifo_runtime(),
            model_path: model.path.clone(),
            budget_fraction: resources.budget,
            implemented: true,
            token_accounting: "native-tokenizer-or-deterministic-estimator".to_string(),
            observability: vec![
                "emit load, request, token, and error telemetry with safe attributes".to_string(),
                "never include prompt content, bearer tokens, API keys, or local paths".to_string(),
            ],
            security: vec![
                "load only configured model aliases".to_string(),
                "validate model artifacts before constructing a native engine".to_string(),
            ],
        };
        validate_native_engine_load_plan(&plan)?;
        Ok(plan)
    }

    pub fn load(&self, plan: &NativeEngineLoadPlan) -> Result<Box<dyn NativeEngine>> {
        validate_native_engine_load_plan(plan)?;
        if !matches!(
            plan.acceleration,
            NativeAcceleration::Cpu | NativeAcceleration::Auto
        ) {
            bail!(
                "native Candle decoding currently supports CPU execution only; requested {} acceleration for model {}",
                plan.acceleration.as_str(),
                plan.alias
            );
        }
        let model = ModelConfig {
            alias: plan.alias.clone(),
            path: plan.model_path.clone(),
            role: plan.role.clone(),
            family: Some(plan.family.clone()),
            weight: 1,
        };
        let artifacts =
            validate_candle_model_artifacts(plan.candle.load_contract.model_family, &model)?;
        verify_candle_artifacts_can_load(&plan.model_path, &artifacts)?;
        let decoder = NativeCandleDecoder::load(
            plan.candle.load_contract.model_family,
            &plan.model_path,
            &artifacts,
        )?;

        Ok(Box::new(ArtifactBackedCandleEngine {
            alias: plan.alias.clone(),
            decoder: Arc::new(decoder),
        }))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeEngineLoadPlan {
    pub runtime: RuntimeBackend,
    pub engine: String,
    pub alias: String,
    pub role: String,
    pub family: String,
    pub format: NativeModelFormat,
    pub acceleration: NativeAcceleration,
    pub candle: CandleEngineConfig,
    pub support: CandleFamilySupportMetadata,
    pub device_selection: CandleDeviceSelectionContract,
    pub scheduler: NativeSchedulerContract,
    #[serde(skip)]
    pub model_path: PathBuf,
    pub budget_fraction: f64,
    pub implemented: bool,
    pub token_accounting: String,
    pub observability: Vec<String>,
    pub security: Vec<String>,
}

#[derive(Debug)]
pub struct ArtifactBackedCandleEngine {
    alias: String,
    // Behind an `Arc` so a clone can be moved into a `spawn_blocking` closure:
    // the candle decode loop is fully synchronous and must not run on a Tokio
    // worker thread (see `chat`).
    decoder: Arc<NativeCandleDecoder>,
}

impl NativeEngine for ArtifactBackedCandleEngine {
    fn model_alias(&self) -> &str {
        &self.alias
    }

    fn chat(&self, request: NativeChatRequest) -> BoxFuture<'_, Result<NativeChatResponse>> {
        let decoder = self.decoder.clone();
        Box::pin(async move { spawn_blocking_decode(decoder, request, None).await })
    }

    fn chat_stream_tokens(
        &self,
        request: NativeChatRequest,
        token_tx: NativeTokenSender,
    ) -> BoxFuture<'_, Result<NativeChatResponse>> {
        let decoder = self.decoder.clone();
        Box::pin(async move { spawn_blocking_decode(decoder, request, Some(token_tx)).await })
    }
}

/// Runs the synchronous candle decode loop on the blocking thread pool so the
/// Tokio executor is never stalled by a multi-thousand-step forward pass held
/// under a `std::sync::Mutex` (Bug 6). When `token_tx` is `Some`, each decoded
/// content delta is forwarded as it is produced (Bug 10); the returned response
/// carries the exact prompt/completion token counts from the decode loop (Bug
/// 12). All `!Send` state (the model mutex guard, the tokenizer) stays inside
/// the closure — only the `Arc<NativeCandleDecoder>`, the owned request, and the
/// `Send` channel sender cross the boundary.
async fn spawn_blocking_decode(
    decoder: Arc<NativeCandleDecoder>,
    request: NativeChatRequest,
    token_tx: Option<NativeTokenSender>,
) -> Result<NativeChatResponse> {
    tokio::task::spawn_blocking(move || {
        let model = request.model.clone();
        let mut sink = |piece: &str| {
            if let Some(tx) = token_tx.as_ref() {
                // A closed receiver (client disconnected) is not an error here;
                // generation continues so terminal accounting still runs.
                let _ = tx.send(piece.to_string());
            }
        };
        let generation = decoder.generate_streaming(&request, &mut sink)?;
        Ok(NativeChatResponse {
            model,
            content: generation.text,
            tool_calls: None,
            finish_reason: generation.finish_reason,
            usage: native_exact_usage(generation.prompt_tokens, generation.completion_tokens),
        })
    })
    .await
    .map_err(|err| anyhow::anyhow!("native decode task failed to join: {err}"))?
}

impl NativeEngineLoadPlan {
    pub fn safe_telemetry_attributes(&self) -> BTreeMap<String, Value> {
        BTreeMap::from([
            (
                "runtime.backend".to_string(),
                Value::String("candle-native".to_string()),
            ),
            (
                "runtime.engine".to_string(),
                Value::String(self.engine.clone()),
            ),
            ("model.alias".to_string(), Value::String(self.alias.clone())),
            ("model.role".to_string(), Value::String(self.role.clone())),
            (
                "model.family".to_string(),
                Value::String(self.candle.load_contract.model_family.as_str().to_string()),
            ),
            (
                "model.format".to_string(),
                json_value(self.candle.load_contract.model_format),
            ),
            (
                "runtime.accelerator".to_string(),
                json_value(self.candle.load_contract.accelerator),
            ),
            (
                "runtime.tokenizer_requirement".to_string(),
                json_value(&self.candle.load_contract.tokenizer),
            ),
            (
                "runtime.implemented".to_string(),
                Value::Bool(self.implemented),
            ),
            (
                "runtime.scheduler.contract_only".to_string(),
                Value::Bool(self.scheduler.contract_only),
            ),
            (
                "runtime.fail_closed".to_string(),
                Value::Bool(self.candle.load_contract.fail_closed),
            ),
        ])
    }
}

#[derive(Debug, Clone, Default)]
pub struct Qwen3CandleEngineLoader;

impl Qwen3CandleEngineLoader {
    pub fn plan(model: &ModelConfig, resources: &ResourceConfig) -> Result<NativeEngineLoadPlan> {
        NativeCandleEngineFactory::default().plan(CandleModelFamily::Qwen3, model, resources)
    }

    pub fn load(&self, plan: &NativeEngineLoadPlan) -> Result<Box<dyn NativeEngine>> {
        NativeCandleEngineFactory::default().load(plan)
    }
}
