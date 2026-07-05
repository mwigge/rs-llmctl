//! Model download, checksum, and artifact-layout/validation helpers.
use super::*;

pub(crate) async fn copy_dir(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir_all(destination)
        .await
        .with_context(|| format!("create {}", destination.display()))?;
    let mut entries = fs::read_dir(source)
        .await
        .with_context(|| format!("read {}", source.display()))?;
    while let Some(entry) = entries.next_entry().await? {
        let entry_type = entry.file_type().await?;
        let target = destination.join(entry.file_name());
        if entry_type.is_dir() {
            Box::pin(copy_dir(&entry.path(), &target)).await?;
        } else if entry_type.is_file() {
            fs::copy(entry.path(), &target).await.with_context(|| {
                format!("copy {} to {}", entry.path().display(), target.display())
            })?;
        }
    }
    Ok(())
}

pub async fn download_model(
    url: &str,
    cache_dir: &Path,
    name_hint: &str,
    expected_sha256: &str,
) -> Result<PathBuf> {
    validate_direct_download_url(url)?;
    validate_direct_download_resolves_public(url)?;
    let expected_sha256 = normalized_sha256(expected_sha256)?;
    fs::create_dir_all(cache_dir)
        .await
        .with_context(|| format!("create model cache {}", cache_dir.display()))?;
    let filename = download_filename(url, name_hint)?;
    let destination = unique_destination(cache_dir, filename.as_ref());
    let partial = destination.with_extension("part");

    let partial_len = match fs::metadata(&partial).await {
        Ok(metadata) if metadata.is_file() => metadata.len(),
        _ => 0,
    };
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(MODEL_DOWNLOAD_TIMEOUT)
        .build()
        .context("build model download client")?;
    let mut request = client.get(url);
    if partial_len > 0 {
        request = request.header(
            RANGE,
            HeaderValue::from_str(&format!("bytes={partial_len}-"))
                .context("build resumable download range header")?,
        );
    }

    let response = request
        .send()
        .await
        .with_context(|| format!("download model from {url}"))?
        .error_for_status()
        .with_context(|| format!("download model from {url}"))?;
    let append_partial =
        partial_len > 0 && response.status() == reqwest::StatusCode::PARTIAL_CONTENT;
    let mut output = if append_partial {
        fs::OpenOptions::new()
            .append(true)
            .open(&partial)
            .await
            .with_context(|| format!("open partial download {}", partial.display()))?
    } else {
        fs::File::create(&partial)
            .await
            .with_context(|| format!("create partial download {}", partial.display()))?
    };
    let mut stream = response.bytes_stream();
    let write_result: Result<()> = async {
        while let Some(chunk) = stream.next().await {
            output
                .write_all(&chunk.with_context(|| format!("read model download from {url}"))?)
                .await
                .with_context(|| format!("write partial download {}", partial.display()))?;
        }
        output
            .flush()
            .await
            .with_context(|| format!("flush partial download {}", partial.display()))?;
        Ok(())
    }
    .await;
    drop(output);
    if let Err(err) = write_result {
        let _ = fs::remove_file(&partial).await;
        return Err(err);
    }

    verify_downloaded_model(&partial, &destination, &expected_sha256).await?;
    Ok(destination)
}

pub(crate) async fn verify_downloaded_model(
    partial: &Path,
    destination: &Path,
    expected_sha256: &str,
) -> Result<()> {
    let Some(name) = destination.file_name().and_then(|name| name.to_str()) else {
        bail!("model path has no filename: {}", destination.display());
    };
    ensure_supported_model_name(name)?;
    anyhow::ensure!(
        !name.to_ascii_lowercase().ends_with(".safetensors"),
        "direct safetensors downloads are not supported without config.json and tokenizer.json sidecars; use an offline manifest or local safetensors directory"
    );
    let expected_sha256 = normalized_sha256(expected_sha256)?;
    let sha256 = sha256_file(partial).await?;
    if sha256 != expected_sha256 {
        let _ = fs::remove_file(partial).await;
        bail!(
            "sha256 mismatch for {}: expected {expected_sha256}, got {sha256}",
            destination.display()
        );
    }
    fs::rename(partial, destination)
        .await
        .with_context(|| format!("move {} to {}", partial.display(), destination.display()))?;
    Ok(())
}

pub async fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)
        .await
        .with_context(|| format!("open model for checksum {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .await
            .with_context(|| format!("read model for checksum {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

pub(crate) fn download_filename(url: &str, name_hint: &str) -> Result<String> {
    let without_query = url.split('?').next().unwrap_or(url);
    let from_url = without_query.rsplit('/').next().unwrap_or("");
    let candidate = if is_supported_model_filename(from_url) {
        from_url
    } else {
        name_hint
    };
    ensure_supported_model_name(candidate)?;
    Ok(candidate.to_string())
}

pub(crate) fn unique_destination(cache_dir: &Path, filename: &std::ffi::OsStr) -> PathBuf {
    let destination = cache_dir.join(filename);
    if !destination.exists() {
        return destination;
    }
    let path = Path::new(filename);
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("gguf");
    for idx in 1.. {
        let candidate = cache_dir.join(format!("{stem}-{idx}.{ext}"));
        if !candidate.exists() {
            return candidate;
        }
    }
    unreachable!("unbounded destination search should always return")
}

pub(crate) fn validate_alias(alias: &str) -> Result<()> {
    anyhow::ensure!(!alias.trim().is_empty(), "model alias cannot be empty");
    anyhow::ensure!(
        alias
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.')),
        "model alias may only contain ASCII letters, numbers, '.', '-', and '_'"
    );
    Ok(())
}

pub(crate) async fn ensure_supported_artifact_path(path: &Path) -> Result<()> {
    let metadata = fs::metadata(path)
        .await
        .with_context(|| format!("stat model artifact {}", path.display()))?;
    if metadata.is_dir() {
        ensure_safetensors_layout(path).await?;
        return Ok(());
    }
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        bail!("model path has no filename: {}", path.display());
    };
    ensure_supported_model_name(name)?;
    if is_safetensors_path(path) {
        ensure_safetensors_file_layout(path).await?;
    }
    Ok(())
}

fn ensure_supported_model_name(name: &str) -> Result<()> {
    anyhow::ensure!(
        is_supported_model_filename(name),
        "model artifact must use .gguf or .safetensors extension: {name}"
    );
    Ok(())
}

fn is_supported_model_filename(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".gguf") || lower.ends_with(".safetensors")
}

pub(crate) fn is_safetensors_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.to_ascii_lowercase().ends_with(".safetensors"))
}

async fn ensure_safetensors_file_layout(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("native safetensors file has no parent directory"))?;
    for required in ["config.json", "tokenizer.json"] {
        let required_path = parent.join(required);
        anyhow::ensure!(
            required_path.is_file(),
            "native safetensors file {} must have sibling {required}",
            path.display()
        );
    }
    Ok(())
}

pub(crate) async fn copy_safetensors_file_layout(
    source: &Path,
    destination_dir: &Path,
) -> Result<()> {
    ensure_safetensors_file_layout(source).await?;
    let filename = source
        .file_name()
        .ok_or_else(|| anyhow!("native safetensors file has no filename"))?;
    fs::copy(source, destination_dir.join(filename))
        .await
        .with_context(|| format!("copy {} to {}", source.display(), destination_dir.display()))?;
    let parent = source
        .parent()
        .ok_or_else(|| anyhow!("native safetensors file has no parent directory"))?;
    for sidecar in ["config.json", "tokenizer.json"] {
        fs::copy(parent.join(sidecar), destination_dir.join(sidecar))
            .await
            .with_context(|| format!("copy safetensors sidecar {sidecar}"))?;
    }
    Ok(())
}

async fn ensure_safetensors_layout(path: &Path) -> Result<()> {
    for required in ["config.json", "tokenizer.json"] {
        let required_path = path.join(required);
        anyhow::ensure!(
            required_path.is_file(),
            "native safetensors directory {} must include {required}",
            path.display()
        );
    }
    let mut entries = fs::read_dir(path)
        .await
        .with_context(|| format!("read native safetensors directory {}", path.display()))?;
    let mut has_weights = false;
    while let Some(entry) = entries.next_entry().await? {
        if entry
            .file_name()
            .to_str()
            .is_some_and(|name| name.to_ascii_lowercase().ends_with(".safetensors"))
        {
            has_weights = true;
            break;
        }
    }
    anyhow::ensure!(
        has_weights,
        "native safetensors directory {} must include at least one .safetensors weight file",
        path.display()
    );
    Ok(())
}

pub(crate) async fn sha256_model_artifact(path: &Path) -> Result<String> {
    let metadata = fs::metadata(path)
        .await
        .with_context(|| format!("stat model artifact {}", path.display()))?;
    if metadata.is_file() {
        return sha256_file(path).await;
    }
    ensure_safetensors_layout(path).await?;
    let mut files = Vec::new();
    let mut entries = fs::read_dir(path)
        .await
        .with_context(|| format!("read model artifact directory {}", path.display()))?;
    while let Some(entry) = entries.next_entry().await? {
        let entry_path = entry.path();
        if entry.file_type().await?.is_file() {
            files.push(entry_path);
        }
    }
    files.sort();
    let mut hasher = Sha256::new();
    for file in files {
        let name = file
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_string();
        hasher.update(name.as_bytes());
        hasher.update(b"\0");
        let digest = sha256_file(&file).await?;
        hasher.update(digest.as_bytes());
        hasher.update(b"\0");
    }
    Ok(hex::encode(hasher.finalize()))
}

pub(crate) async fn model_artifact_len(path: &Path) -> Result<u64> {
    let metadata = fs::metadata(path)
        .await
        .with_context(|| format!("stat model artifact {}", path.display()))?;
    if metadata.is_file() {
        return Ok(metadata.len());
    }
    let mut total = 0_u64;
    let mut entries = fs::read_dir(path)
        .await
        .with_context(|| format!("read model artifact directory {}", path.display()))?;
    while let Some(entry) = entries.next_entry().await? {
        let metadata = entry.metadata().await?;
        if metadata.is_file() {
            total = total.saturating_add(metadata.len());
        }
    }
    Ok(total)
}

pub(crate) fn validate_direct_download_url(url: &str) -> Result<()> {
    let parsed = reqwest::Url::parse(url).with_context(|| format!("parse model URL {url}"))?;
    anyhow::ensure!(
        parsed.scheme() == "https",
        "direct model downloads require https URLs"
    );
    let host = parsed
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("direct model download URL must include a host"))?;
    anyhow::ensure!(
        !matches!(host, "localhost" | "metadata.google.internal"),
        "direct model download URL host is not allowed"
    );
    if let Ok(ip) = host.parse::<IpAddr>() {
        anyhow::ensure!(
            !is_blocked_download_ip(ip),
            "direct model download URL must not target local or private addresses"
        );
    }
    Ok(())
}

pub(crate) fn validate_direct_download_resolves_public(url: &str) -> Result<()> {
    let parsed = reqwest::Url::parse(url).with_context(|| format!("parse model URL {url}"))?;
    let host = parsed
        .host_str()
        .ok_or_else(|| anyhow!("direct model download URL must include a host"))?;
    let port = parsed.port_or_known_default().unwrap_or(443);
    let resolved = (host, port)
        .to_socket_addrs()
        .with_context(|| format!("resolve direct model download host {host}"))?
        .collect::<Vec<_>>();
    anyhow::ensure!(
        !resolved.is_empty(),
        "direct model download URL host did not resolve"
    );
    for addr in resolved {
        anyhow::ensure!(
            !is_blocked_download_ip(addr.ip()),
            "direct model download URL must not resolve to local or private addresses"
        );
    }
    Ok(())
}

fn is_blocked_download_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            ip.is_loopback()
                || ip.is_private()
                || ip.is_link_local()
                || ip.is_unspecified()
                || ip.octets() == [169, 254, 169, 254]
        }
        IpAddr::V6(ip) => {
            ip.is_loopback()
                || ip.is_unspecified()
                || ip.is_unique_local()
                || ip.is_unicast_link_local()
        }
    }
}

fn validate_sha256(value: &str) -> Result<()> {
    anyhow::ensure!(
        value.len() == 64 && value.chars().all(|ch| ch.is_ascii_hexdigit()),
        "sha256 must be 64 hexadecimal characters"
    );
    Ok(())
}

pub(crate) fn normalized_sha256(value: &str) -> Result<String> {
    let normalized = value.to_ascii_lowercase();
    validate_sha256(&normalized)?;
    Ok(normalized)
}

pub(crate) fn resolve_manifest_path(base_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    }
}

pub(crate) fn ensure_relative_component(value: &str, label: &str) -> Result<()> {
    anyhow::ensure!(!value.trim().is_empty(), "{label} cannot be empty");
    anyhow::ensure!(
        !value.contains("..") && !value.starts_with('/') && !value.starts_with('\\'),
        "{label} must be a relative HuggingFace path component"
    );
    Ok(())
}
