use super::error_response;
use crate::config::{Config, ExternalProviderKind, Mode, ModelConfig};
use crate::native;
use axum::body::Bytes;
use axum::http::StatusCode;
use axum::response::Response;
use serde_json::Value;
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ResolvedModelRoute {
    pub(super) requested_alias: String,
    pub(super) upstream_alias: String,
    pub(super) fallback_aliases: Vec<String>,
    pub(super) external_provider: Option<ResolvedExternalProvider>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ResolvedExternalProvider {
    pub(super) id: String,
    pub(super) kind: ExternalProviderKind,
    pub(super) base_url: String,
    pub(super) api_key_env: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ModelRouteError {
    UnknownAlias(String),
    NoConfiguredModels,
    ExternalProviderRoutingDisabled(String),
}

impl std::fmt::Display for ModelRouteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownAlias(alias) => write!(f, "unknown model alias: {alias}"),
            Self::NoConfiguredModels => write!(f, "no models are configured"),
            Self::ExternalProviderRoutingDisabled(provider) => write!(
                f,
                "external provider routing is disabled for provider {provider}"
            ),
        }
    }
}

pub(super) fn model_route_error_response(err: &ModelRouteError) -> Response {
    match err {
        ModelRouteError::UnknownAlias(_) | ModelRouteError::NoConfiguredModels => {
            error_response(StatusCode::NOT_FOUND, "model_not_found", err.to_string())
        }
        ModelRouteError::ExternalProviderRoutingDisabled(_) => {
            error_response(StatusCode::BAD_REQUEST, "bad_request", err.to_string())
        }
    }
}

pub(super) fn routed_models(cfg: &Config) -> Vec<&ModelConfig> {
    let mut models: Vec<_> = cfg
        .models
        .iter()
        .filter(|model| model_is_routed_locally(cfg, model))
        .collect();
    models.sort_by(|left, right| left.alias.cmp(&right.alias));
    models
}

pub(super) fn active_routed_models(cfg: &Config) -> Vec<&ModelConfig> {
    routed_models(cfg)
        .into_iter()
        .filter(|model| model.weight > 0)
        .collect()
}

fn model_is_routed_locally(cfg: &Config, model: &ModelConfig) -> bool {
    let placement = native::placement_plan_from_config(cfg);
    placement
        .nodes
        .iter()
        .find(|node| node.id == placement.local_node)
        .map(|node| node.model_aliases.iter().any(|alias| alias == &model.alias))
        .unwrap_or(false)
}

pub(super) fn resolve_model_route(
    cfg: &Config,
    requested_alias: &str,
    request_id: Uuid,
) -> std::result::Result<ResolvedModelRoute, ModelRouteError> {
    if cfg.models.is_empty() {
        if requested_alias.trim().is_empty() {
            return Err(ModelRouteError::NoConfiguredModels);
        }
        return Ok(ResolvedModelRoute {
            requested_alias: requested_alias.to_string(),
            upstream_alias: requested_alias.to_string(),
            fallback_aliases: Vec::new(),
            external_provider: None,
        });
    }

    let requested = cfg
        .models
        .iter()
        .find(|model| model.alias == requested_alias && model_is_routed_locally(cfg, model))
        .ok_or_else(|| ModelRouteError::UnknownAlias(requested_alias.to_string()))?;

    let upstream = match cfg.mode {
        Mode::Single => cfg
            .models
            .iter()
            .find(|model| model_is_routed_locally(cfg, model))
            .ok_or(ModelRouteError::NoConfiguredModels)?,
        Mode::ColdSwap | Mode::HotSwap => requested,
        Mode::Weighted => weighted_model_for_request(cfg, request_id).unwrap_or(requested),
        Mode::Fallback => {
            if requested.weight > 0 {
                requested
            } else {
                weighted_model_for_request(cfg, request_id).unwrap_or(requested)
            }
        }
    };
    let fallback_aliases = if matches!(cfg.mode, Mode::Fallback) {
        fallback_aliases(cfg, &upstream.alias)
    } else {
        Vec::new()
    };
    if let Some(route) = cfg.external_providers.route_for_model(&upstream.alias) {
        return Err(ModelRouteError::ExternalProviderRoutingDisabled(
            route.provider.clone(),
        ));
    }

    Ok(ResolvedModelRoute {
        requested_alias: requested_alias.to_string(),
        upstream_alias: upstream.alias.clone(),
        fallback_aliases,
        external_provider: None,
    })
}

fn weighted_model_for_request(cfg: &Config, request_id: Uuid) -> Option<&ModelConfig> {
    let weighted = cfg
        .models
        .iter()
        .filter(|model| model.weight > 0 && model_is_routed_locally(cfg, model))
        .collect::<Vec<_>>();
    let total = weighted.iter().fold(0u64, |total, model| {
        total.saturating_add(u64::from(model.weight))
    });
    if total == 0 {
        return None;
    }

    let mut slot = request_id.as_u128() % u128::from(total);
    for model in weighted {
        let weight = u128::from(model.weight);
        if slot < weight {
            return Some(model);
        }
        slot -= weight;
    }

    None
}

fn fallback_aliases(cfg: &Config, selected_alias: &str) -> Vec<String> {
    let mut models = routed_models(cfg)
        .into_iter()
        .filter(|model| model.alias != selected_alias)
        .collect::<Vec<_>>();
    models.sort_by(|left, right| {
        right
            .weight
            .cmp(&left.weight)
            .then_with(|| left.alias.cmp(&right.alias))
    });
    models
        .into_iter()
        .map(|model| model.alias.clone())
        .collect()
}

pub(super) fn rewrite_chat_model(
    body: &[u8],
    route: &ResolvedModelRoute,
) -> std::result::Result<Bytes, String> {
    if route.requested_alias == route.upstream_alias {
        return Ok(Bytes::copy_from_slice(body));
    }

    let mut value: Value =
        serde_json::from_slice(body).map_err(|_| "request body must be valid JSON".to_string())?;
    let Some(object) = value.as_object_mut() else {
        return Err("request body must be a JSON object".to_string());
    };
    object.insert(
        "model".to_string(),
        Value::String(route.upstream_alias.clone()),
    );
    serde_json::to_vec(&value)
        .map(Bytes::from)
        .map_err(|err| err.to_string())
}
