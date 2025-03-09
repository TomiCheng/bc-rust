use thiserror::Error;

#[derive(Error, Debug)]
#[error("arithmetic error: {message}")]
pub struct ArithmeticError {
    message: String,
    #[source]
    source: Option<anyhow::Error>,
}

impl ArithmeticError {
    pub fn with_message(message: String) -> Self {
        ArithmeticError {
            message,
            source: None,
        }
    }
}
