use futures_util::Stream;
use std::pin::Pin;
use std::time::Duration;

pub(crate) const DEFAULT_CLIENT_TIMEOUT: Duration = Duration::from_secs(300);

pub type Result<T> = std::result::Result<T, LlmctlError>;
pub type ChatStream = Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk>> + Send>>;

mod contracts;
pub use contracts::*;
mod types;
pub use types::*;
mod error;
pub use error::*;
mod client;
pub use client::*;
mod transport;
pub(crate) use transport::*;

#[cfg(test)]
mod tests;
