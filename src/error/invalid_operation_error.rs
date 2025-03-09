use thiserror::Error;

#[derive(Error, Debug)]
#[error("invalid operation: {message}")]
pub struct InvalidOperationError {
    message: String,
    #[source]
    source: Option<anyhow::Error>,
}

impl InvalidOperationError {
    pub fn with_message(message: String) -> Self {
        InvalidOperationError {
            message,
            source: None,
        }
    }
}
