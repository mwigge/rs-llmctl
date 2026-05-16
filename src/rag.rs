use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SearchDocument {
    pub id: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchHit {
    pub id: String,
    pub title: Option<String>,
    pub path: Option<String>,
    pub score: f64,
    pub snippet: String,
}

pub fn lexical_search(query: &str, documents: &[SearchDocument], limit: usize) -> Vec<SearchHit> {
    let query_terms = tokenize(query);
    if query_terms.is_empty() || limit == 0 {
        return Vec::new();
    }

    let mut hits = documents
        .iter()
        .filter_map(|document| {
            let content_terms = tokenize(&document.content);
            let title_terms = tokenize(document.title.as_deref().unwrap_or_default());
            let mut score = 0.0;
            for term in &query_terms {
                if title_terms.contains(term) {
                    score += 3.0;
                }
                if content_terms.contains(term) {
                    score += 1.0;
                }
            }
            if score <= 0.0 {
                return None;
            }
            Some(SearchHit {
                id: document.id.clone(),
                title: document.title.clone(),
                path: document.path.clone(),
                score,
                snippet: snippet(&document.content, &query_terms),
            })
        })
        .collect::<Vec<_>>();

    hits.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.id.cmp(&right.id))
    });
    hits.truncate(limit);
    hits
}

fn tokenize(text: &str) -> BTreeSet<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '_')
        .map(str::trim)
        .filter(|term| term.len() >= 2)
        .map(str::to_ascii_lowercase)
        .collect()
}

fn snippet(content: &str, terms: &BTreeSet<String>) -> String {
    let lower = content.to_ascii_lowercase();
    let start = terms
        .iter()
        .filter_map(|term| lower.find(term))
        .min()
        .unwrap_or(0);
    let start = content[..start].rfind('\n').map(|idx| idx + 1).unwrap_or(0);
    let end = content[start..]
        .find('\n')
        .map(|idx| start + idx)
        .unwrap_or_else(|| content.len());
    content[start..end].chars().take(240).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexical_search_ranks_title_and_content_matches() {
        let documents = vec![
            SearchDocument {
                id: "ops".to_string(),
                title: Some("Operations Runbook".to_string()),
                path: Some("docs/operations.md".to_string()),
                content: "restart workers and check readiness".to_string(),
            },
            SearchDocument {
                id: "security".to_string(),
                title: Some("Security".to_string()),
                path: Some("docs/security.md".to_string()),
                content: "api keys and audit evidence".to_string(),
            },
        ];

        let hits = lexical_search("operations readiness", &documents, 10);

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "ops");
        assert!(hits[0].score > 1.0);
        assert_eq!(hits[0].snippet, "restart workers and check readiness");
    }
}
