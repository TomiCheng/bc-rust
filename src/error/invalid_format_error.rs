use thiserror::Error;

#[derive(Error, Debug)]
#[error("invalid format: {message}")]
pub struct InvalidFormatError {
    message: String,
    #[source]
    source: Option<anyhow::Error>,
}

impl InvalidFormatError {
    pub fn with_message(message: String) -> Self {
        InvalidFormatError {
            message,
            source: None,
        }
    }
}
