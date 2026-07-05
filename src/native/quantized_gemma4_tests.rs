
use super::quantized_gemma4;
use candle_core::quantized::gguf_file::{Content, TensorInfo, Value as GgufValue, VersionedMagic};
use candle_core::quantized::GgmlDType;
use candle_core::{Device, Shape};
use std::collections::HashMap;
use std::io::Cursor;

/// Synthetic gemma4 GGUF config: 2 layers, layer 0 sliding (local) and
/// layer 1 global, mirroring the real model's alternating attention
/// pattern but at a tiny scale.
const EMBEDDING_LENGTH: usize = 8;
const HEAD_COUNT: usize = 2;
const KEY_LENGTH: usize = 4; // global head_dim
const KEY_LENGTH_SWA: usize = 2; // sliding head_dim
const FFN_DIM: usize = 6;
const VOCAB_SIZE: usize = 10;
const BLOCK_COUNT: usize = 2;

/// Appends an F32 tensor (raw little-endian bytes) to `data` and records
/// a matching [`TensorInfo`] entry in `tensor_infos`.
fn push_tensor(
    data: &mut Vec<u8>,
    tensor_infos: &mut HashMap<String, TensorInfo>,
    name: &str,
    shape: &[usize],
) {
    let elem_count: usize = shape.iter().product();
    let offset = data.len() as u64;
    for i in 0..elem_count {
        // Small deterministic values keep RmsNorm/softmax well-behaved.
        let value = 0.01 * (i as f32 + 1.0);
        data.extend_from_slice(&value.to_le_bytes());
    }
    tensor_infos.insert(
        name.to_string(),
        TensorInfo {
            ggml_dtype: GgmlDType::F32,
            shape: Shape::from(shape.to_vec()),
            offset,
        },
    );
}

/// Builds a synthetic `gemma4` GGUF [`Content`] plus its backing tensor
/// data, with `block_count` layers alternating sliding/global per
/// `sliding_window_pattern`.
fn gemma4_content_and_data(
    head_count_kv: &[u32],
    sliding_window_pattern: &[u32],
) -> (Content, Cursor<Vec<u8>>) {
    let mut metadata = HashMap::new();
    metadata.insert(
        "gemma4.block_count".to_string(),
        GgufValue::U32(BLOCK_COUNT as u32),
    );
    metadata.insert(
        "gemma4.embedding_length".to_string(),
        GgufValue::U32(EMBEDDING_LENGTH as u32),
    );
    metadata.insert(
        "gemma4.attention.head_count".to_string(),
        GgufValue::U32(HEAD_COUNT as u32),
    );
    metadata.insert(
        "gemma4.attention.key_length".to_string(),
        GgufValue::U32(KEY_LENGTH as u32),
    );
    metadata.insert(
        "gemma4.attention.value_length".to_string(),
        GgufValue::U32(KEY_LENGTH as u32),
    );
    metadata.insert(
        "gemma4.attention.key_length_swa".to_string(),
        GgufValue::U32(KEY_LENGTH_SWA as u32),
    );
    metadata.insert(
        "gemma4.attention.value_length_swa".to_string(),
        GgufValue::U32(KEY_LENGTH_SWA as u32),
    );
    metadata.insert(
        "gemma4.attention.layer_norm_rms_epsilon".to_string(),
        GgufValue::F32(1e-6),
    );
    metadata.insert(
        "gemma4.attention.sliding_window".to_string(),
        GgufValue::U32(4),
    );
    metadata.insert(
        "gemma4.rope.freq_base".to_string(),
        GgufValue::F32(1_000_000.0),
    );
    metadata.insert(
        "gemma4.rope.freq_base_swa".to_string(),
        GgufValue::F32(10_000.0),
    );
    metadata.insert(
        "gemma4.attention.head_count_kv".to_string(),
        GgufValue::Array(head_count_kv.iter().copied().map(GgufValue::U32).collect()),
    );
    metadata.insert(
        "gemma4.attention.sliding_window_pattern".to_string(),
        GgufValue::Array(
            sliding_window_pattern
                .iter()
                .copied()
                .map(GgufValue::U32)
                .collect(),
        ),
    );

    let mut data = Vec::new();
    let mut tensor_infos = HashMap::new();

    push_tensor(
        &mut data,
        &mut tensor_infos,
        "token_embd.weight",
        &[VOCAB_SIZE, EMBEDDING_LENGTH],
    );
    push_tensor(
        &mut data,
        &mut tensor_infos,
        "output_norm.weight",
        &[EMBEDDING_LENGTH],
    );
    push_tensor(
        &mut data,
        &mut tensor_infos,
        "output.weight",
        &[VOCAB_SIZE, EMBEDDING_LENGTH],
    );

    for (layer_idx, &pattern) in sliding_window_pattern.iter().enumerate() {
        let head_dim = if pattern == 1 {
            KEY_LENGTH_SWA
        } else {
            KEY_LENGTH
        };
        let n_kv_head = head_count_kv[layer_idx] as usize;
        let q_dim = HEAD_COUNT * head_dim;
        let kv_dim = n_kv_head * head_dim;
        let prefix = format!("blk.{layer_idx}");

        push_tensor(
            &mut data,
            &mut tensor_infos,
            &format!("{prefix}.attn_q.weight"),
            &[q_dim, EMBEDDING_LENGTH],
        );
        push_tensor(
            &mut data,
            &mut tensor_infos,
            &format!("{prefix}.attn_k.weight"),
            &[kv_dim, EMBEDDING_LENGTH],
        );
        // Global (non-sliding) layers have no `attn_v.weight` in the real
        // GGUF; mirror that here so the fixture exercises the
        // `Vcur = Kcur` fallback path.
        if pattern == 1 {
            push_tensor(
                &mut data,
                &mut tensor_infos,
                &format!("{prefix}.attn_v.weight"),
                &[kv_dim, EMBEDDING_LENGTH],
            );
        }
        push_tensor(
            &mut data,
            &mut tensor_infos,
            &format!("{prefix}.attn_output.weight"),
            &[EMBEDDING_LENGTH, q_dim],
        );
        push_tensor(
            &mut data,
            &mut tensor_infos,
            &format!("{prefix}.attn_q_norm.weight"),
            &[head_dim],
        );
        push_tensor(
            &mut data,
            &mut tensor_infos,
            &format!("{prefix}.attn_k_norm.weight"),
            &[head_dim],
        );
        push_tensor(
            &mut data,
            &mut tensor_infos,
            &format!("{prefix}.attn_norm.weight"),
            &[EMBEDDING_LENGTH],
        );
        push_tensor(
            &mut data,
            &mut tensor_infos,
            &format!("{prefix}.post_attention_norm.weight"),
            &[EMBEDDING_LENGTH],
        );
        push_tensor(
            &mut data,
            &mut tensor_infos,
            &format!("{prefix}.ffn_norm.weight"),
            &[EMBEDDING_LENGTH],
        );
        push_tensor(
            &mut data,
            &mut tensor_infos,
            &format!("{prefix}.post_ffw_norm.weight"),
            &[EMBEDDING_LENGTH],
        );
        push_tensor(
            &mut data,
            &mut tensor_infos,
            &format!("{prefix}.layer_output_scale.weight"),
            &[1],
        );
        push_tensor(
            &mut data,
            &mut tensor_infos,
            &format!("{prefix}.ffn_gate.weight"),
            &[FFN_DIM, EMBEDDING_LENGTH],
        );
        push_tensor(
            &mut data,
            &mut tensor_infos,
            &format!("{prefix}.ffn_up.weight"),
            &[FFN_DIM, EMBEDDING_LENGTH],
        );
        push_tensor(
            &mut data,
            &mut tensor_infos,
            &format!("{prefix}.ffn_down.weight"),
            &[EMBEDDING_LENGTH, FFN_DIM],
        );
    }

    let content = Content {
        magic: VersionedMagic::GgufV3,
        metadata,
        tensor_infos,
        tensor_data_offset: 0,
    };
    (content, Cursor::new(data))
}

#[test]
fn from_gguf_builds_model_with_alternating_sliding_and_global_layers() {
    let (content, mut reader) = gemma4_content_and_data(&[1, 1], &[1, 0]);
    let device = Device::Cpu;

    let mut model = quantized_gemma4::ModelWeights::from_gguf(content, &mut reader, &device)
        .expect("synthetic gemma4 GGUF should build successfully");

    let input = candle_core::Tensor::new(&[1u32, 2u32, 3u32], &device)
        .and_then(|t| t.reshape((1, 3)))
        .expect("input tensor");

    let logits = model
        .forward(&input, 0)
        .expect("forward pass on synthetic gemma4 model should succeed");

    assert_eq!(logits.dims(), &[1, VOCAB_SIZE]);
}

#[test]
fn from_gguf_rejects_missing_head_count_kv_array() {
    let (mut content, mut reader) = gemma4_content_and_data(&[1, 1], &[1, 0]);
    content.metadata.remove("gemma4.attention.head_count_kv");
    let device = Device::Cpu;

    let err = quantized_gemma4::ModelWeights::from_gguf(content, &mut reader, &device)
        .expect_err("missing head_count_kv array must be rejected");
    let message = err.to_string();
    assert!(
        message.contains("attention.head_count_kv"),
        "error message `{message}` should mention the missing key"
    );
}

#[test]
fn from_gguf_rejects_wrong_length_sliding_window_pattern() {
    let (mut content, mut reader) = gemma4_content_and_data(&[1, 1], &[1, 0]);
    // Replace with an array of the wrong length (1 element instead of 2).
    content.metadata.insert(
        "gemma4.attention.sliding_window_pattern".to_string(),
        GgufValue::Array(vec![GgufValue::U32(1)]),
    );
    let device = Device::Cpu;

    let err = quantized_gemma4::ModelWeights::from_gguf(content, &mut reader, &device)
        .expect_err("wrong-length sliding_window_pattern array must be rejected");
    let message = err.to_string();
    assert!(
        message.contains("sliding_window_pattern"),
        "error message `{message}` should mention the offending key"
    );
    assert!(
        message.contains("expected 2"),
        "error message `{message}` should mention the expected length"
    );
}

#[test]
fn from_gguf_rejects_non_array_head_count_kv() {
    let (mut content, mut reader) = gemma4_content_and_data(&[1, 1], &[1, 0]);
    content.metadata.insert(
        "gemma4.attention.head_count_kv".to_string(),
        GgufValue::U32(1),
    );
    let device = Device::Cpu;

    let err = quantized_gemma4::ModelWeights::from_gguf(content, &mut reader, &device)
        .expect_err("non-array head_count_kv must be rejected");
    let message = err.to_string();
    assert!(
        message.contains("head_count_kv"),
        "error message `{message}` should mention the offending key"
    );
    assert!(
        message.contains("not an array"),
        "error message `{message}` should explain it is not an array"
    );
}

#[cfg(feature = "llama-cpp-native")]
mod llama_cpp_tests {
    use super::super::{LlamaCppNativeEngine, NativeEngine};

    #[test]
    fn load_rejects_missing_path() {
        let result = LlamaCppNativeEngine::load(
            "test".to_string(),
            std::path::Path::new("/nonexistent/model.gguf"),
            32,
        );
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("not found") || msg.contains("nonexistent"),
            "error message `{msg}` should mention missing path"
        );
    }

    #[test]
    fn load_stores_alias_and_gpu_layers() {
        use tempfile::NamedTempFile;
        let tmp = NamedTempFile::new().expect("create temp file");
        let engine = LlamaCppNativeEngine {
            alias: "my-model".to_string(),
            model_path: tmp.path().to_owned(),
            gpu_layers: 32,
        };
        assert_eq!(engine.model_alias(), "my-model");
        assert_eq!(engine.gpu_layers, 32);
    }

    #[test]
    fn llama_cpp_native_engine_implements_native_engine_trait() {
        fn assert_native_engine<T: crate::native::NativeEngine>() {}
        assert_native_engine::<LlamaCppNativeEngine>();
    }

    #[test]
    fn arc_from_box_dyn_engine_preserves_alias() {
        use std::sync::Arc;
        use tempfile::NamedTempFile;
        let tmp = NamedTempFile::new().expect("temp file");
        let engine =
            LlamaCppNativeEngine::load("registry-alias".to_string(), tmp.path(), 0).unwrap();
        let boxed: Box<dyn NativeEngine> = Box::new(engine);
        let arc: Arc<dyn NativeEngine> = Arc::from(boxed);
        assert_eq!(arc.model_alias(), "registry-alias");
    }
}

#[test]
#[ignore = "requires the real gemma-4-12b-it-Q4_K_M.gguf model file on disk"]
fn real_gemma4_gguf_constructs_model_and_runs_forward() {
    let path = std::path::Path::new(
        "/home/morgan/.local/share/milliways/models/gemma-4-12b-it-Q4_K_M.gguf",
    );
    if !path.exists() {
        eprintln!("skipping: {} not present", path.display());
        return;
    }

    let mut file = std::fs::File::open(path).expect("open real gemma4 gguf");
    let content = candle_core::quantized::gguf_file::Content::read(&mut file)
        .expect("read real gemma4 gguf metadata");
    let device = Device::Cpu;

    let mut model = quantized_gemma4::ModelWeights::from_gguf(content, &mut file, &device)
        .expect("construct quantized gemma4 model from real GGUF weights");

    let input = candle_core::Tensor::new(&[2u32, 3u32], &device)
        .and_then(|t| t.reshape((1, 2)))
        .expect("input tensor");

    let logits = model
        .forward(&input, 0)
        .expect("forward pass on real gemma4 model should succeed");

    eprintln!("logits dims: {:?}", logits.dims());
    assert_eq!(logits.dims().len(), 2);
    assert_eq!(logits.dims()[0], 1);
    assert!(logits.dims()[1] > 0);
}
