//! Candle native-engine load-plan and model-artifact validation.
use super::*;

pub fn validate_native_engine_load_plan(plan: &NativeEngineLoadPlan) -> Result<()> {
    if plan.runtime != RuntimeBackend::CandleNative {
        bail!("native load plan runtime must be candle-native");
    }
    if plan.alias.trim().is_empty() {
        bail!("native load plan has an empty model alias");
    }
    if !(0.0..=1.0).contains(&plan.budget_fraction) {
        bail!(
            "native load plan budget_fraction must be between 0.0 and 1.0, got {}",
            plan.budget_fraction
        );
    }
    if plan.engine != plan.candle.engine || plan.engine != plan.support.engine {
        bail!("native load plan engine does not match Candle support metadata");
    }
    if plan.candle.load_contract.model_family != plan.support.model_family {
        bail!("native load plan family does not match Candle support metadata");
    }
    if plan.family != plan.candle.load_contract.model_family.as_str() {
        bail!("native load plan family string does not match Candle load contract");
    }
    if plan.format != plan.candle.load_contract.model_format {
        bail!("native load plan format does not match Candle load contract");
    }
    if plan.acceleration != plan.candle.load_contract.accelerator {
        bail!("native load plan acceleration does not match Candle load contract");
    }
    if plan.device_selection != plan.candle.load_contract.device_selection {
        bail!("native load plan device selection does not match Candle load contract");
    }
    if !plan.support.supported_formats.contains(&plan.format) {
        bail!(
            "{} does not support model format {:?}",
            plan.engine,
            plan.format
        );
    }
    if !plan
        .support
        .supported_accelerators
        .contains(&plan.acceleration)
    {
        bail!(
            "{} does not support acceleration {:?}",
            plan.engine,
            plan.acceleration
        );
    }
    if plan.candle.load_contract.tokenizer == CandleTokenizerRequirement::UnsupportedFormat {
        bail!("native load plan has no tokenizer contract for model format");
    }
    if plan.candle.load_contract.supported_operations.is_empty() {
        bail!("native load plan has no supported Candle operations");
    }
    if !plan.implemented && !plan.candle.load_contract.fail_closed {
        bail!("unimplemented native load plan must fail closed");
    }
    if plan.scheduler.contract_only {
        bail!("native scheduler contract must report implemented FIFO queue runtime");
    }
    if plan.scheduler.queue.discipline != NativeQueueDiscipline::Fifo
        || !plan.scheduler.queue.implemented
        || !plan.scheduler.batching.prefill_decode_phase_scheduling
        || plan.scheduler.batching.implemented
        || plan.scheduler.kv_cache.reuse_implemented
        || plan.scheduler.kv_cache.implemented
        || !plan.scheduler.cancellation.admission_check_implemented
        || plan.scheduler.cancellation.decode_loop_check_implemented
        || plan.scheduler.cancellation.implemented
    {
        bail!(
            "native scheduler must implement FIFO queue and phase metadata while continuous batching, KV-cache reuse, and decode cancellation remain explicit unsupported runtime boundaries"
        );
    }

    Ok(())
}

pub fn validate_candle_model_artifacts(
    family: CandleModelFamily,
    model: &ModelConfig,
) -> Result<CandleArtifactValidation> {
    let support = CandleFamilySupportMetadata::for_family(family);
    let format = infer_native_artifact_format(&model.path);
    if !support.supported_formats.contains(&format) {
        bail!(
            "{} cannot load model alias '{}' because the artifact format is unsupported; expected a .gguf file or safetensors weights with tokenizer.json and config.json",
            family.engine_name(),
            model.alias
        );
    }

    let layout = CandleArtifactLayout::for_format(format);
    match format {
        NativeModelFormat::Gguf => validate_gguf_artifacts(family, model, layout),
        NativeModelFormat::Safetensors => validate_safetensors_artifacts(family, model, layout),
        NativeModelFormat::Unknown => unreachable!("unsupported formats are rejected above"),
    }
}

fn validate_gguf_artifacts(
    family: CandleModelFamily,
    model: &ModelConfig,
    layout: CandleArtifactLayout,
) -> Result<CandleArtifactValidation> {
    let mut missing = Vec::new();
    if !model.path.is_file() || NativeModelFormat::from_path(&model.path) != NativeModelFormat::Gguf
    {
        missing.push("GGUF weights (*.gguf)".to_string());
    }

    fail_missing_artifacts(family, model, NativeModelFormat::Gguf, &missing)?;
    Ok(CandleArtifactValidation {
        model_family: family,
        model_format: NativeModelFormat::Gguf,
        layout,
        weight_files: vec![artifact_file_name(&model.path)],
        tokenizer_file: None,
        config_file: None,
    })
}

pub(crate) fn validate_safetensors_artifacts(
    family: CandleModelFamily,
    model: &ModelConfig,
    layout: CandleArtifactLayout,
) -> Result<CandleArtifactValidation> {
    let artifact_dir = safetensors_artifact_dir(&model.path);
    let weights = safetensors_weight_files(&model.path, artifact_dir);
    let tokenizer = artifact_dir.join("tokenizer.json");
    let config = artifact_dir.join("config.json");

    let mut missing = Vec::new();
    if weights.is_empty() {
        missing.push("safetensors weights (*.safetensors)".to_string());
    }
    if !tokenizer.is_file() {
        missing.push("tokenizer.json".to_string());
    }
    if !config.is_file() {
        missing.push("config.json".to_string());
    }

    fail_missing_artifacts(family, model, NativeModelFormat::Safetensors, &missing)?;
    Ok(CandleArtifactValidation {
        model_family: family,
        model_format: NativeModelFormat::Safetensors,
        layout,
        weight_files: weights,
        tokenizer_file: Some("tokenizer.json".to_string()),
        config_file: Some("config.json".to_string()),
    })
}

pub(crate) fn verify_candle_artifacts_can_load(
    model_path: &Path,
    artifacts: &CandleArtifactValidation,
) -> Result<()> {
    match artifacts.model_format {
        NativeModelFormat::Gguf => verify_gguf_can_load(model_path),
        NativeModelFormat::Safetensors => verify_safetensors_can_load(model_path, artifacts),
        NativeModelFormat::Unknown => bail!("native artifact format is unsupported"),
    }
}

#[cfg(feature = "native-candle")]
fn verify_gguf_can_load(model_path: &Path) -> Result<()> {
    let device = candle_core::Device::Cpu;
    candle_transformers::quantized_var_builder::VarBuilder::from_gguf(model_path, &device)
        .map(|_| ())
        .with_context(|| "failed to load GGUF weights with Candle")
}

#[cfg(not(feature = "native-candle"))]
fn verify_gguf_can_load(_model_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(feature = "native-candle")]
fn verify_safetensors_can_load(
    model_path: &Path,
    artifacts: &CandleArtifactValidation,
) -> Result<()> {
    let artifact_dir = safetensors_artifact_dir(model_path);
    let paths = artifacts
        .weight_files
        .iter()
        .map(|name| artifact_dir.join(name))
        .collect::<Vec<_>>();
    let device = candle_core::Device::Cpu;
    // Candle exposes safetensors mmap loading as unsafe because it relies on OS mmap invariants.
    // The paths come from validation immediately above and are only used for read-only weight access.
    unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&paths, candle_core::DType::F32, &device)
    }
    .map(|_| ())
    .with_context(|| "failed to load safetensors weights with Candle")
}

#[cfg(not(feature = "native-candle"))]
fn verify_safetensors_can_load(
    _model_path: &Path,
    _artifacts: &CandleArtifactValidation,
) -> Result<()> {
    Ok(())
}

fn fail_missing_artifacts(
    family: CandleModelFamily,
    model: &ModelConfig,
    format: NativeModelFormat,
    missing: &[String],
) -> Result<()> {
    if missing.is_empty() {
        return Ok(());
    }

    bail!(
        "{} cannot load model alias '{}' as {:?}: missing required artifact(s): {}",
        family.engine_name(),
        model.alias,
        format,
        missing.join(", ")
    )
}

pub(crate) fn infer_native_artifact_format(path: &Path) -> NativeModelFormat {
    let format = NativeModelFormat::from_path(path);
    if format != NativeModelFormat::Unknown {
        return format;
    }

    if path.is_dir() && !safetensors_weight_files(path, path).is_empty() {
        return NativeModelFormat::Safetensors;
    }

    NativeModelFormat::Unknown
}

pub(crate) fn safetensors_artifact_dir(path: &Path) -> &Path {
    if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    }
}

fn safetensors_weight_files(path: &Path, artifact_dir: &Path) -> Vec<String> {
    if path.is_file() && NativeModelFormat::from_path(path) == NativeModelFormat::Safetensors {
        return vec![artifact_file_name(path)];
    }

    let mut weights = fs::read_dir(artifact_dir)
        .ok()
        .into_iter()
        .flat_map(|entries| entries.filter_map(Result::ok))
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file() && NativeModelFormat::from_path(path) == NativeModelFormat::Safetensors
        })
        .map(|path| artifact_file_name(&path))
        .collect::<Vec<_>>();
    weights.sort();
    weights
}

pub(crate) fn artifact_file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("<unnamed>")
        .to_string()
}
