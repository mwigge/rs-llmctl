//! Native embeddings: Candle BERT semantic path and deterministic fallback vectors.
use super::*;

pub const DETERMINISTIC_EMBEDDING_DIMENSIONS: usize = 64;

#[derive(Debug)]
pub struct NativeBertEmbeddingEngine {
    alias: String,
    encoder: NativeBertEmbeddingEncoder,
}

impl NativeBertEmbeddingEngine {
    pub fn load(alias: impl Into<String>, model_path: impl AsRef<Path>) -> Result<Self> {
        let alias = alias.into();
        let encoder = NativeBertEmbeddingEncoder::load(model_path.as_ref())?;
        Ok(Self { alias, encoder })
    }
}

impl NativeEngine for NativeBertEmbeddingEngine {
    fn model_alias(&self) -> &str {
        &self.alias
    }

    fn chat(&self, request: NativeChatRequest) -> BoxFuture<'_, Result<NativeChatResponse>> {
        Box::pin(async move {
            bail!(
                "native BERT embedding model '{}' does not serve chat completions",
                request.model
            )
        })
    }

    fn embeddings(
        &self,
        request: NativeEmbeddingRequest,
    ) -> BoxFuture<'_, Result<NativeEmbeddingResponse>> {
        Box::pin(async move {
            let (embeddings, input_tokens) = self.encoder.embed(&request.input)?;
            Ok(NativeEmbeddingResponse {
                model: request.model,
                embeddings,
                usage: NativeTokenUsage::with_mode(
                    input_tokens,
                    0,
                    TokenAccountingMode::NativeExact,
                ),
                backend: "candle-bert-embeddings".to_string(),
                status: "semantic-native".to_string(),
                semantic: true,
            })
        })
    }
}

#[derive(Debug)]
enum NativeBertEmbeddingEncoder {
    #[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
    Real(RealBertEmbeddingEncoder),
    #[cfg(not(all(feature = "native-candle", feature = "native-tokenizers")))]
    Unavailable,
}

impl NativeBertEmbeddingEncoder {
    fn load(model_path: &Path) -> Result<Self> {
        load_real_bert_embedding_encoder(model_path)
    }

    fn embed(&self, input: &[String]) -> Result<(Vec<Vec<f32>>, u64)> {
        match self {
            #[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
            Self::Real(encoder) => encoder.embed(input),
            #[cfg(not(all(feature = "native-candle", feature = "native-tokenizers")))]
            Self::Unavailable => {
                let _ = input;
                bail!(
                    "semantic native embeddings require the native-candle and native-tokenizers features"
                )
            }
        }
    }
}

#[cfg(not(all(feature = "native-candle", feature = "native-tokenizers")))]
fn load_real_bert_embedding_encoder(_model_path: &Path) -> Result<NativeBertEmbeddingEncoder> {
    Ok(NativeBertEmbeddingEncoder::Unavailable)
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
struct RealBertEmbeddingEncoder {
    tokenizer: tokenizers::tokenizer::Tokenizer,
    model: Mutex<candle_transformers::models::bert::BertModel>,
    pad_token_id: u32,
    max_position_embeddings: usize,
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
impl std::fmt::Debug for RealBertEmbeddingEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RealBertEmbeddingEncoder")
            .field("pad_token_id", &self.pad_token_id)
            .field("max_position_embeddings", &self.max_position_embeddings)
            .finish_non_exhaustive()
    }
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
fn load_real_bert_embedding_encoder(model_path: &Path) -> Result<NativeBertEmbeddingEncoder> {
    let artifact_dir = safetensors_artifact_dir(model_path);
    let model = ModelConfig {
        alias: "semantic-embedding".to_string(),
        path: model_path.to_path_buf(),
        role: "embedding".to_string(),
        family: Some("qwen3".to_string()),
        weight: 1,
    };
    let artifacts = validate_semantic_embedding_artifacts(&model)?;
    let paths = artifacts
        .weight_files
        .iter()
        .map(|name| artifact_dir.join(name))
        .collect::<Vec<_>>();
    let config_path = artifact_dir.join("config.json");
    let tokenizer_path = artifact_dir.join("tokenizer.json");
    let config: candle_transformers::models::bert::Config = read_json_config(&config_path)?;
    let tokenizer = tokenizers::tokenizer::Tokenizer::from_file(&tokenizer_path)
        .map_err(|err| anyhow::anyhow!("failed to load tokenizer.json: {err}"))?;
    let device = candle_core::Device::Cpu;
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&paths, candle_core::DType::F32, &device)
    }
    .with_context(|| "failed to mmap BERT embedding safetensors with Candle")?;
    let bert = candle_transformers::models::bert::BertModel::load(vb, &config)
        .with_context(|| "failed to construct BERT embedding model")?;
    Ok(NativeBertEmbeddingEncoder::Real(RealBertEmbeddingEncoder {
        tokenizer,
        model: Mutex::new(bert),
        pad_token_id: config.pad_token_id as u32,
        max_position_embeddings: config.max_position_embeddings,
    }))
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
impl RealBertEmbeddingEncoder {
    fn embed(&self, input: &[String]) -> Result<(Vec<Vec<f32>>, u64)> {
        if input.is_empty() {
            return Ok((Vec::new(), 0));
        }

        let mut encoded = Vec::with_capacity(input.len());
        let mut max_len = 0usize;
        let mut input_tokens = 0u64;
        for text in input {
            let encoding = self
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|err| anyhow::anyhow!("failed to tokenize embedding input: {err}"))?;
            let mut ids = encoding.get_ids().to_vec();
            if ids.is_empty() {
                ids.push(self.pad_token_id);
            }
            ids.truncate(self.max_position_embeddings.max(1));
            input_tokens = input_tokens.saturating_add(ids.len() as u64);
            max_len = max_len.max(ids.len());
            encoded.push(ids);
        }

        let mut input_ids = Vec::with_capacity(encoded.len());
        let mut token_type_ids = Vec::with_capacity(encoded.len());
        let mut attention_mask = Vec::with_capacity(encoded.len());
        for ids in &encoded {
            let mut padded = ids.clone();
            let mut mask = vec![1u32; ids.len()];
            padded.resize(max_len, self.pad_token_id);
            mask.resize(max_len, 0);
            input_ids.push(padded);
            token_type_ids.push(vec![0u32; max_len]);
            attention_mask.push(mask);
        }

        let device = candle_core::Device::Cpu;
        let input_ids = candle_core::Tensor::new(input_ids, &device)
            .with_context(|| "failed to build BERT input_ids tensor")?;
        let token_type_ids = candle_core::Tensor::new(token_type_ids, &device)
            .with_context(|| "failed to build BERT token_type_ids tensor")?;
        let attention = candle_core::Tensor::new(attention_mask.clone(), &device)
            .with_context(|| "failed to build BERT attention_mask tensor")?;
        let model = self
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("native BERT embedding model lock is poisoned"))?;
        let sequence = model
            .forward(&input_ids, &token_type_ids, Some(&attention))
            .with_context(|| "native BERT embedding forward pass failed")?;
        let hidden = sequence
            .to_vec3::<f32>()
            .with_context(|| "failed to read BERT embedding tensor")?;
        let embeddings = mean_pool_hidden(hidden, &attention_mask);
        Ok((embeddings, input_tokens))
    }
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
fn mean_pool_hidden(hidden: Vec<Vec<Vec<f32>>>, masks: &[Vec<u32>]) -> Vec<Vec<f32>> {
    hidden
        .into_iter()
        .zip(masks.iter())
        .map(|(tokens, mask)| {
            let dimensions = tokens.first().map(Vec::len).unwrap_or(0);
            let mut pooled = vec![0.0f32; dimensions];
            let mut count = 0f32;
            for (token, active) in tokens.into_iter().zip(mask.iter()) {
                if *active == 0 {
                    continue;
                }
                for (slot, value) in pooled.iter_mut().zip(token) {
                    *slot += value;
                }
                count += 1.0;
            }
            if count > 0.0 {
                for value in &mut pooled {
                    *value /= count;
                }
            }
            normalize_vector(pooled)
        })
        .collect()
}

#[cfg(all(feature = "native-candle", feature = "native-tokenizers"))]
fn validate_semantic_embedding_artifacts(model: &ModelConfig) -> Result<CandleArtifactValidation> {
    let format = infer_native_artifact_format(&model.path);
    if format != NativeModelFormat::Safetensors {
        bail!(
            "candle-bert-embeddings cannot load model alias '{}' because semantic embeddings require safetensors weights with tokenizer.json and config.json",
            model.alias
        );
    }
    let layout = CandleArtifactLayout::for_format(NativeModelFormat::Safetensors);
    validate_safetensors_artifacts(CandleModelFamily::Qwen3, model, layout).map_err(|err| {
        anyhow::anyhow!(
            "{}",
            err.to_string()
                .replace("candle-native-qwen3", "candle-bert-embeddings")
        )
    })
}

pub fn deterministic_native_embeddings(
    request: NativeEmbeddingRequest,
) -> Result<NativeEmbeddingResponse> {
    let input_tokens = request
        .input
        .iter()
        .map(|input| EstimatedNativeTokenCounter::estimate_text_tokens(input))
        .sum();
    let embeddings = request
        .input
        .iter()
        .map(|input| deterministic_embedding_vector(&request.model, input))
        .collect();
    Ok(NativeEmbeddingResponse {
        model: request.model,
        embeddings,
        usage: NativeTokenUsage::with_mode(input_tokens, 0, TokenAccountingMode::Estimated),
        backend: "deterministic-local-fallback".to_string(),
        status: "non-semantic-dev-fallback".to_string(),
        semantic: false,
    })
}

fn deterministic_embedding_vector(model: &str, input: &str) -> Vec<f32> {
    let mut vector = Vec::with_capacity(DETERMINISTIC_EMBEDDING_DIMENSIONS);
    let mut counter = 0u64;
    while vector.len() < DETERMINISTIC_EMBEDDING_DIMENSIONS {
        let mut hasher = Sha256::new();
        hasher.update(model.as_bytes());
        hasher.update(b"\0");
        hasher.update(input.as_bytes());
        hasher.update(b"\0");
        hasher.update(counter.to_le_bytes());
        let digest = hasher.finalize();
        for chunk in digest.chunks_exact(4) {
            let raw = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let unit = (raw as f32) / (u32::MAX as f32);
            vector.push(unit.mul_add(2.0, -1.0));
            if vector.len() == DETERMINISTIC_EMBEDDING_DIMENSIONS {
                break;
            }
        }
        counter = counter.saturating_add(1);
    }
    normalize_vector(vector)
}

fn normalize_vector(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}
