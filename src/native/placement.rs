//! Placement, routing, and heartbeat planning for native cluster nodes.
use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativePlacementPlan {
    pub routing_mode: String,
    pub local_node: String,
    pub nodes: Vec<NativePlacementNode>,
    pub unassigned_models: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativePlacementNode {
    pub id: String,
    pub base_url: String,
    pub roles: Vec<String>,
    pub model_aliases: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeRouteSelection {
    pub query: String,
    pub candidates: Vec<NativePlacementNode>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeHeartbeat {
    pub node_id: String,
    pub runtime: RuntimeBackend,
    pub routing_mode: String,
    pub healthy: bool,
    pub models: usize,
    pub assigned_models: usize,
    pub unassigned_models: Vec<String>,
    pub budget_fraction: f64,
    pub heartbeat_interval_seconds: u64,
    pub telemetry_event: String,
}

impl NativeHeartbeat {
    pub fn safe_telemetry_attributes(&self) -> BTreeMap<String, Value> {
        BTreeMap::from([
            ("cluster.node_id".to_string(), json_value(&self.node_id)),
            ("runtime.backend".to_string(), json_value(self.runtime)),
            (
                "runtime.routing_mode".to_string(),
                json_value(&self.routing_mode),
            ),
            ("runtime.healthy".to_string(), Value::Bool(self.healthy)),
            (
                "runtime.models".to_string(),
                Value::from(self.models as u64),
            ),
            (
                "runtime.assigned_models".to_string(),
                Value::from(self.assigned_models as u64),
            ),
            (
                "runtime.resource.budget_fraction".to_string(),
                Value::from(self.budget_fraction),
            ),
            (
                "runtime.heartbeat_interval_seconds".to_string(),
                Value::from(self.heartbeat_interval_seconds),
            ),
        ])
    }
}

pub fn heartbeat_from_config(cfg: &Config) -> NativeHeartbeat {
    let placement = placement_plan_from_config(cfg);
    let assigned_models = placement
        .nodes
        .iter()
        .map(|node| node.model_aliases.len())
        .sum();
    let healthy = validate_placement_plan(&placement).is_ok();
    NativeHeartbeat {
        node_id: cfg.cluster.node_id.clone(),
        runtime: cfg.runtime.backend,
        routing_mode: placement.routing_mode,
        healthy,
        models: cfg.models.len(),
        assigned_models,
        unassigned_models: placement.unassigned_models,
        budget_fraction: cfg.resources.budget,
        heartbeat_interval_seconds: cfg.runtime.heartbeat_interval_seconds,
        telemetry_event: "llmctl.runtime.heartbeat".to_string(),
    }
}

pub fn placement_plan_from_config(cfg: &Config) -> NativePlacementPlan {
    let nodes = if cfg.cluster.nodes.is_empty() {
        vec![NativePlacementNode {
            id: cfg.cluster.node_id.clone(),
            base_url: format!("http://{}:{}/v1", cfg.server.host, cfg.server.port),
            roles: sorted_roles(&cfg.models),
            model_aliases: cfg.models.iter().map(|model| model.alias.clone()).collect(),
        }]
    } else {
        cfg.cluster
            .nodes
            .iter()
            .map(|node| placement_node(node, &cfg.models))
            .collect()
    };

    let assigned = nodes
        .iter()
        .flat_map(|node| node.model_aliases.iter().cloned())
        .collect::<std::collections::BTreeSet<_>>();
    let unassigned_models = cfg
        .models
        .iter()
        .filter(|model| !assigned.contains(&model.alias))
        .map(|model| model.alias.clone())
        .collect();

    NativePlacementPlan {
        routing_mode: if cfg.cluster.nodes.is_empty() {
            "single-node".to_string()
        } else {
            "cluster-role-placement".to_string()
        },
        local_node: cfg.cluster.node_id.clone(),
        nodes,
        unassigned_models,
    }
}

pub fn validate_placement_plan(plan: &NativePlacementPlan) -> Result<()> {
    if !plan.unassigned_models.is_empty() {
        bail!(
            "native placement leaves model aliases unassigned: {}",
            plan.unassigned_models.join(", ")
        );
    }

    let mut owners: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for node in &plan.nodes {
        if node.id.trim().is_empty() {
            bail!("native placement contains a node with an empty id");
        }
        if node.base_url.trim().is_empty() {
            bail!("native placement node '{}' has an empty base_url", node.id);
        }
        for alias in &node.model_aliases {
            owners
                .entry(alias.as_str())
                .or_default()
                .push(node.id.as_str());
        }
    }

    let duplicate = owners
        .iter()
        .find(|(_, node_ids)| node_ids.len() > 1)
        .map(|(alias, node_ids)| ((*alias).to_string(), node_ids.join(", ")));
    if let Some((alias, node_ids)) = duplicate {
        bail!("native placement assigns model alias '{alias}' to multiple nodes: {node_ids}");
    }

    Ok(())
}

pub fn route_selection_for_model(
    plan: &NativePlacementPlan,
    model_alias: &str,
) -> Result<NativeRouteSelection> {
    let candidates = plan
        .nodes
        .iter()
        .filter(|node| {
            node.model_aliases
                .iter()
                .any(|alias| alias.as_str() == model_alias)
        })
        .cloned()
        .collect::<Vec<_>>();

    if candidates.is_empty() {
        bail!("native placement has no node for model alias '{model_alias}'");
    }
    if candidates.len() > 1 {
        bail!("native placement has multiple nodes for model alias '{model_alias}'");
    }

    Ok(NativeRouteSelection {
        query: format!("model:{model_alias}"),
        candidates,
    })
}

pub fn route_selection_for_role(
    plan: &NativePlacementPlan,
    role: &str,
) -> Result<NativeRouteSelection> {
    let normalized = normalize_role(role);
    let candidates = plan
        .nodes
        .iter()
        .filter(|node| node.roles.iter().any(|node_role| node_role == normalized))
        .cloned()
        .collect::<Vec<_>>();

    if candidates.is_empty() {
        bail!("native placement has no node for role '{normalized}'");
    }

    Ok(NativeRouteSelection {
        query: format!("role:{normalized}"),
        candidates,
    })
}

pub(crate) fn json_value<T: Serialize>(value: T) -> Value {
    serde_json::to_value(value).unwrap_or(Value::Null)
}

fn placement_node(node: &ClusterNodeConfig, models: &[ModelConfig]) -> NativePlacementNode {
    let explicit_aliases = node
        .model_aliases
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let role_set = node
        .roles
        .iter()
        .map(|role| normalize_role(role).to_string())
        .collect::<std::collections::BTreeSet<_>>();
    let model_aliases = models
        .iter()
        .filter(|model| {
            explicit_aliases.contains(&model.alias)
                || role_set.contains(normalize_role(&model.role))
        })
        .map(|model| model.alias.clone())
        .collect();

    NativePlacementNode {
        id: node.id.clone(),
        base_url: node.base_url.clone(),
        roles: node
            .roles
            .iter()
            .map(|role| normalize_role(role).to_string())
            .collect(),
        model_aliases,
    }
}

fn sorted_roles(models: &[ModelConfig]) -> Vec<String> {
    models
        .iter()
        .map(|model| normalize_role(&model.role).to_string())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}
