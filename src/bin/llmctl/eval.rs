use crate::*;

pub(crate) async fn eval_command(path: &Path, command: EvalCommand, as_json: bool) -> Result<()> {
    let cfg = load_config(path).await?;
    let path = state_file(&cfg, "eval-runs.jsonl")?;
    match command {
        EvalCommand::Run(args) => {
            let record = json!({
                "schema_version": 1,
                "id": Uuid::new_v4(),
                "at": Utc::now(),
                "model": args.model,
                "suite": args.suite,
                "score": args.score,
                "baseline": args.baseline,
                "delta": args.baseline.map(|baseline| args.score - baseline),
                "notes": args.notes
            });
            append_jsonl(&path, &record).await?;
            emit(as_json, &record)
        }
        EvalCommand::RunSuite(args) => {
            let record = run_eval_suite(&cfg, args).await?;
            append_jsonl(&path, &record).await?;
            emit(as_json, &record)
        }
        EvalCommand::List => emit(
            as_json,
            &json!({
                "schema_version": 1,
                "path": path,
                "runs": read_jsonl(&path).await?
            }),
        ),
        EvalCommand::Report => {
            let runs = read_jsonl(&path).await?;
            emit(as_json, &eval_report(&runs))
        }
    }
}

async fn run_eval_suite(cfg: &Config, args: EvalRunSuiteArgs) -> Result<serde_json::Value> {
    let manifest = read_eval_manifest(&args.manifest).await?;
    if manifest.cases.is_empty() {
        bail!("eval manifest {} has no cases", args.manifest.display());
    }

    let base_url = args
        .base_url
        .unwrap_or_else(|| format!("http://{}:{}", cfg.server.host, cfg.server.port));
    let endpoint = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));
    let api_key = match args.api_key_env.as_deref() {
        Some(env) => Some(std::env::var(env).with_context(|| format!("read API key env {env}"))?),
        None => None,
    };
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .context("build eval HTTP client")?;
    let mut cases = Vec::with_capacity(manifest.cases.len());

    for case in &manifest.cases {
        validate_expectation(&case.expect)
            .with_context(|| format!("validate eval case {}", case.id))?;
        let output = if base_url.starts_with("mock://") {
            mock_eval_output(&case.expect, &case.prompt)
        } else {
            execute_eval_case(&client, &endpoint, api_key.as_deref(), &manifest, case)
                .await
                .with_context(|| format!("execute eval case {}", case.id))?
        };
        let checks = score_eval_case(&case.expect, &output)
            .with_context(|| format!("score eval case {}", case.id))?;
        let passed = checks.values().all(|passed| *passed);
        cases.push(json!({
            "id": case.id,
            "passed": passed,
            "checks": checks,
            "output": output
        }));
    }

    let passed = cases
        .iter()
        .filter(|case| case.get("passed").and_then(serde_json::Value::as_bool) == Some(true))
        .count();
    let total = cases.len();
    let score = passed as f64 / total as f64;
    Ok(json!({
        "schema_version": 1,
        "kind": "eval-suite-run",
        "id": Uuid::new_v4(),
        "at": Utc::now(),
        "manifest": args.manifest,
        "base_url": base_url,
        "model": manifest.model,
        "suite": manifest.suite,
        "score": score,
        "passed": passed,
        "failed": total - passed,
        "total": total,
        "cases": cases
    }))
}

fn mock_eval_output(expect: &EvalExpectation, prompt: &str) -> String {
    let mut output = expect
        .exact
        .clone()
        .or_else(|| {
            if expect.contains.is_empty() {
                None
            } else {
                Some(expect.contains.join(" "))
            }
        })
        .unwrap_or_else(|| prompt.to_string());
    if expect.regex.is_some() && !output.contains("score=") {
        output.push_str(" score=1");
    }
    output
}

async fn read_eval_manifest(path: &Path) -> Result<EvalSuiteManifest> {
    let body = fs::read_to_string(path)
        .await
        .with_context(|| format!("read eval manifest {}", path.display()))?;
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("toml") => toml::from_str(&body)
            .with_context(|| format!("parse TOML eval manifest {}", path.display())),
        _ => serde_json::from_str(&body)
            .with_context(|| format!("parse JSON eval manifest {}", path.display())),
    }
}

fn validate_expectation(expect: &EvalExpectation) -> Result<()> {
    if expect.exact.is_none() && expect.contains.is_empty() && expect.regex.is_none() {
        bail!("expectation must set exact, contains, or regex");
    }
    if let Some(pattern) = &expect.regex {
        Regex::new(pattern).with_context(|| format!("compile regex {pattern:?}"))?;
    }
    Ok(())
}

async fn execute_eval_case(
    client: &reqwest::Client,
    endpoint: &str,
    api_key: Option<&str>,
    manifest: &EvalSuiteManifest,
    case: &EvalCaseManifest,
) -> Result<String> {
    let mut messages = Vec::new();
    if let Some(system) = &manifest.system {
        messages.push(json!({"role": "system", "content": system}));
    }
    messages.push(json!({"role": "user", "content": &case.prompt}));

    let mut request = json!({
        "model": &manifest.model,
        "messages": messages,
        "stream": false
    });
    if let Some(temperature) = manifest.temperature {
        request["temperature"] = json!(temperature);
    }
    if let Some(max_tokens) = manifest.max_tokens {
        request["max_tokens"] = json!(max_tokens);
    }

    let mut builder = client.post(endpoint).json(&request);
    if let Some(api_key) = api_key {
        builder = builder.bearer_auth(api_key);
    }
    let response = builder
        .send()
        .await
        .with_context(|| format!("POST {endpoint}"))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .with_context(|| format!("read response from {endpoint}"))?;
    if !status.is_success() {
        bail!("endpoint returned {status}: {body}");
    }
    let value: serde_json::Value =
        serde_json::from_str(&body).context("parse chat completion response")?;
    value
        .pointer("/choices/0/message/content")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| {
            anyhow::anyhow!("chat completion response missing choices[0].message.content")
        })
}

fn score_eval_case(expect: &EvalExpectation, output: &str) -> Result<BTreeMap<String, bool>> {
    let mut checks = BTreeMap::new();
    if let Some(exact) = &expect.exact {
        checks.insert("exact".to_string(), output == exact);
    }
    if !expect.contains.is_empty() {
        checks.insert(
            "contains".to_string(),
            expect.contains.iter().all(|needle| output.contains(needle)),
        );
    }
    if let Some(pattern) = &expect.regex {
        checks.insert("regex".to_string(), Regex::new(pattern)?.is_match(output));
    }
    Ok(checks)
}

fn eval_report(runs: &[serde_json::Value]) -> serde_json::Value {
    let mut by_model: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for run in runs {
        if let (Some(model), Some(score)) = (
            run.get("model").and_then(serde_json::Value::as_str),
            run.get("score").and_then(serde_json::Value::as_f64),
        ) {
            by_model.entry(model.to_string()).or_default().push(score);
        }
    }
    let models = by_model
        .into_iter()
        .map(|(model, scores)| {
            let count = scores.len() as f64;
            let average_score = if count == 0.0 {
                None
            } else {
                Some(scores.iter().sum::<f64>() / count)
            };
            json!({
                "model": model,
                "runs": scores.len(),
                "average_score": average_score
            })
        })
        .collect::<Vec<_>>();
    json!({
        "schema_version": 1,
        "kind": "eval-report",
        "generated_at": Utc::now(),
        "run_count": runs.len(),
        "models": models
    })
}
