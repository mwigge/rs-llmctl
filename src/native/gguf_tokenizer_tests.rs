
use super::*;
use candle_core::quantized::gguf_file::{Content, Value as GgufValue, VersionedMagic};
use std::collections::HashMap;
use std::path::Path;

/// Builds a tiny synthetic GGUF `Content` with a `gemma4`-style SentencePiece
/// metaspace vocabulary, suitable for exercising the tokenizer builder without
/// reading a real model file.
fn gemma4_content() -> Content {
    let tokens = vec![
        "<pad>".to_string(),  // 0
        "<eos>".to_string(),  // 1
        "<bos>".to_string(),  // 2
        "<unk>".to_string(),  // 3
        "<mask>".to_string(), // 4
        "▁".to_string(),      // 5
        "h".to_string(),      // 6
        "i".to_string(),      // 7
        "t".to_string(),      // 8
        "e".to_string(),      // 9
        "r".to_string(),      // 10
        "hi".to_string(),     // 11
        "▁hi".to_string(),    // 12
        "th".to_string(),     // 13
        "the".to_string(),    // 14
        "ther".to_string(),   // 15
        "there".to_string(),  // 16
        "▁there".to_string(), // 17
    ];
    let token_type: Vec<GgufValue> = vec![
        GgufValue::U32(3), // <pad> -> control
        GgufValue::U32(3), // <eos> -> control
        GgufValue::U32(3), // <bos> -> control
        GgufValue::U32(2), // <unk> -> unknown
        GgufValue::U32(3), // <mask> -> control
        GgufValue::U32(1), // ▁ -> normal
        GgufValue::U32(1), // h -> normal
        GgufValue::U32(1), // i -> normal
        GgufValue::U32(1), // t -> normal
        GgufValue::U32(1), // e -> normal
        GgufValue::U32(1), // r -> normal
        GgufValue::U32(1), // hi -> normal
        GgufValue::U32(1), // ▁hi -> normal
        GgufValue::U32(1), // th -> normal
        GgufValue::U32(1), // the -> normal
        GgufValue::U32(1), // ther -> normal
        GgufValue::U32(1), // there -> normal
        GgufValue::U32(1), // ▁there -> normal
    ];
    let merges = vec![
        "h i".to_string(),
        "▁ hi".to_string(),
        "t h".to_string(),
        "th e".to_string(),
        "the r".to_string(),
        "ther e".to_string(),
        "▁ there".to_string(),
    ];

    let mut metadata = HashMap::new();
    metadata.insert(
        "tokenizer.ggml.model".to_string(),
        GgufValue::String("gemma4".to_string()),
    );
    metadata.insert(
        "tokenizer.ggml.tokens".to_string(),
        GgufValue::Array(tokens.into_iter().map(GgufValue::String).collect()),
    );
    metadata.insert(
        "tokenizer.ggml.merges".to_string(),
        GgufValue::Array(merges.into_iter().map(GgufValue::String).collect()),
    );
    metadata.insert(
        "tokenizer.ggml.token_type".to_string(),
        GgufValue::Array(token_type),
    );
    metadata.insert("tokenizer.ggml.bos_token_id".to_string(), GgufValue::U32(2));
    metadata.insert("tokenizer.ggml.eos_token_id".to_string(), GgufValue::U32(1));
    metadata.insert(
        "tokenizer.ggml.unknown_token_id".to_string(),
        GgufValue::U32(3),
    );
    metadata.insert(
        "tokenizer.ggml.padding_token_id".to_string(),
        GgufValue::U32(0),
    );
    metadata.insert(
        "tokenizer.ggml.add_space_prefix".to_string(),
        GgufValue::Bool(false),
    );
    metadata.insert(
        "tokenizer.ggml.add_bos_token".to_string(),
        GgufValue::Bool(true),
    );

    Content {
        magic: VersionedMagic::GgufV3,
        metadata,
        tensor_infos: HashMap::new(),
        tensor_data_offset: 0,
    }
}

/// Builds a minimal gpt2-style GGUF `Content`, just enough metadata for
/// candle's `TokenizerFromGguf::from_gguf` to be reached and attempted.
fn gpt2_content() -> Content {
    let tokens = vec![
        "<|endoftext|>".to_string(),
        "h".to_string(),
        "i".to_string(),
        "Ġthere".to_string(),
        "hi".to_string(),
    ];
    let merges = vec!["h i".to_string()];

    let mut metadata = HashMap::new();
    metadata.insert(
        "tokenizer.ggml.model".to_string(),
        GgufValue::String("gpt2".to_string()),
    );
    metadata.insert(
        "tokenizer.ggml.tokens".to_string(),
        GgufValue::Array(tokens.into_iter().map(GgufValue::String).collect()),
    );
    metadata.insert(
        "tokenizer.ggml.merges".to_string(),
        GgufValue::Array(merges.into_iter().map(GgufValue::String).collect()),
    );

    Content {
        magic: VersionedMagic::GgufV3,
        metadata,
        tensor_infos: HashMap::new(),
        tensor_data_offset: 0,
    }
}

fn unsupported_content() -> Content {
    let mut metadata = HashMap::new();
    metadata.insert(
        "tokenizer.ggml.model".to_string(),
        GgufValue::String("made-up-model".to_string()),
    );

    Content {
        magic: VersionedMagic::GgufV3,
        metadata,
        tensor_infos: HashMap::new(),
        tensor_data_offset: 0,
    }
}

#[test]
fn gemma4_metaspace_tokenizer_builds_and_round_trips() {
    let content = gemma4_content();
    let tokenizer = tokenizer_from_gguf_content(&content)
        .expect("gemma4 metaspace tokenizer should build from synthetic GGUF metadata");

    let encoding = tokenizer
        .encode("hi there", false)
        .expect("encoding should succeed");
    let ids = encoding.get_ids();
    assert!(!ids.is_empty(), "encoding should produce at least one id");

    // With add_space_prefix = false (PrependScheme::Never), the first token
    // must not gain a leading metaspace marker that wasn't in the input.
    let first_token = tokenizer
        .id_to_token(ids[0])
        .expect("first id maps to a token");
    assert!(
        !first_token.starts_with('▁'),
        "leading token `{first_token}` should not carry a metaspace prefix \
             when add_space_prefix is false"
    );

    let decoded = tokenizer
        .decode(ids, true)
        .expect("decoding should succeed");
    assert_eq!(decoded, "hi there");
}

#[test]
fn gguf_bos_token_to_prepend_reads_gemma4_add_bos_token_metadata() {
    let content = gemma4_content();
    assert_eq!(gguf_bos_token_to_prepend(&content), Some(2));
}

#[test]
fn gguf_bos_token_to_prepend_is_none_when_add_bos_token_is_absent() {
    let content = gpt2_content();
    assert_eq!(gguf_bos_token_to_prepend(&content), None);
}

#[test]
fn prepend_bos_if_configured_inserts_missing_bos() {
    let mut input_ids = vec![10, 11, 12];
    prepend_bos_if_configured(&mut input_ids, Some(2));
    assert_eq!(input_ids, vec![2, 10, 11, 12]);
}

#[test]
fn prepend_bos_if_configured_is_noop_when_bos_already_present() {
    let mut input_ids = vec![2, 10, 11, 12];
    prepend_bos_if_configured(&mut input_ids, Some(2));
    assert_eq!(input_ids, vec![2, 10, 11, 12]);
}

#[test]
fn prepend_bos_if_configured_is_noop_when_bos_token_id_is_none() {
    let mut input_ids = vec![10, 11, 12];
    prepend_bos_if_configured(&mut input_ids, None);
    assert_eq!(input_ids, vec![10, 11, 12]);
}

#[test]
fn unsupported_tokenizer_model_is_rejected() {
    let content = unsupported_content();
    let err = tokenizer_from_gguf_content(&content)
        .expect_err("unrecognized tokenizer.ggml.model must be rejected");
    let message = err.to_string();
    assert!(
        message.contains("made-up-model"),
        "error message `{message}` should mention the unsupported model kind"
    );
}

#[test]
fn gpt2_tokenizer_model_delegates_to_candle() {
    let content = gpt2_content();
    // The gpt2 branch must delegate to candle's own
    // `TokenizerFromGguf::from_gguf`, which builds successfully for this
    // minimal-but-valid gpt2 metadata.
    let tokenizer = tokenizer_from_gguf_content(&content)
        .expect("gpt2 metadata should be handled by candle's existing implementation");
    assert!(tokenizer.get_vocab_size(false) >= 4);
}

#[test]
#[ignore = "requires the real gemma-4-12b-it-Q4_K_M.gguf model file on disk"]
fn real_gemma4_gguf_round_trips_hello_world() {
    let path = Path::new("/home/morgan/.local/share/milliways/models/gemma-4-12b-it-Q4_K_M.gguf");
    if !path.exists() {
        eprintln!("skipping: {} not present", path.display());
        return;
    }

    let mut file = fs::File::open(path).expect("open real gemma4 gguf");
    let content = candle_core::quantized::gguf_file::Content::read(&mut file)
        .expect("read real gemma4 gguf metadata");

    let tokenizer = tokenizer_from_gguf_content(&content)
        .expect("build tokenizer from real gemma4 gguf metadata");

    let encoding = tokenizer
        .encode("Hello, world!", false)
        .expect("encode real text");
    let ids = encoding.get_ids().to_vec();
    let decoded = tokenizer.decode(&ids, true).expect("decode real text");

    eprintln!("ids: {ids:?}");
    eprintln!("decoded: {decoded:?}");
    assert_eq!(decoded, "Hello, world!");
}

#[test]
#[ignore = "requires the real gemma-4-12b-it-Q4_K_M.gguf model file on disk"]
fn real_gemma4_gguf_chat_input_ids_begin_with_bos_token() {
    let path = Path::new("/home/morgan/.local/share/milliways/models/gemma-4-12b-it-Q4_K_M.gguf");
    if !path.exists() {
        eprintln!("skipping: {} not present", path.display());
        return;
    }

    let mut file = fs::File::open(path).expect("open real gemma4 gguf");
    let content = candle_core::quantized::gguf_file::Content::read(&mut file)
        .expect("read real gemma4 gguf metadata");

    let tokenizer = tokenizer_from_gguf_content(&content)
        .expect("build tokenizer from real gemma4 gguf metadata");
    let bos_token_id = gguf_bos_token_to_prepend(&content);
    assert_eq!(
        bos_token_id,
        Some(2),
        "gemma4 GGUF should configure bos_token_id=2 with add_bos_token=true"
    );

    let messages = vec![NativeChatMessage {
        role: "user".to_string(),
        content: Some(Value::String(
            "Say hello in one short sentence.".to_string(),
        )),
        tool_calls: None,
        tool_call_id: None,
    }];
    let prompt = gemma_chat_input(&messages);
    let encoding = tokenizer.encode(prompt, false).expect("encode chat prompt");
    let mut input_ids = encoding.get_ids().to_vec();
    prepend_bos_if_configured(&mut input_ids, bos_token_id);

    eprintln!("input_ids: {input_ids:?}");
    assert_eq!(
        input_ids.first().copied(),
        Some(2),
        "constructed input_ids should begin with the gemma BOS token id (2)"
    );
}

#[test]
#[ignore = "requires the real gemma-4-12b-it-Q4_K_M.gguf model file on disk and runs a full 12B forward pass"]
fn real_gemma4_generation_produces_non_garbage_output() {
    let model_path =
        Path::new("/home/morgan/.local/share/milliways/models/gemma-4-12b-it-Q4_K_M.gguf");
    if !model_path.exists() {
        eprintln!("skipping: {} not present", model_path.display());
        return;
    }

    let artifacts = CandleArtifactValidation {
        model_family: CandleModelFamily::Gemma4,
        model_format: NativeModelFormat::Gguf,
        layout: CandleArtifactLayout::for_format(NativeModelFormat::Gguf),
        weight_files: vec![artifact_file_name(model_path)],
        tokenizer_file: None,
        config_file: None,
    };

    let decoder = load_real_candle_decoder(CandleModelFamily::Gemma4, model_path, &artifacts)
        .expect("load real gemma4 candle decoder from GGUF");

    let request = NativeChatRequest {
        model: "gemma4:12b".to_string(),
        messages: vec![NativeChatMessage {
            role: "user".to_string(),
            content: Some(Value::String(
                "Say hello in one short sentence.".to_string(),
            )),
            tool_calls: None,
            tool_call_id: None,
        }],
        temperature: None,
        max_tokens: Some(32),
        top_p: None,
        top_k: None,
        seed: None,
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        tools: None,
        tool_choice: None,
        metadata: BTreeMap::new(),
    };

    let output = decoder
        .generate(&request)
        .expect("real gemma4 generation should succeed");

    eprintln!("decoded output: {output:?}");
    assert!(!output.is_empty(), "decoded output should be non-empty");
    assert!(
        output.chars().any(|ch| ch.is_ascii_alphabetic()),
        "decoded output `{output}` should contain at least one ASCII letter, \
             got what looks like garbage/replacement-character output"
    );
}
