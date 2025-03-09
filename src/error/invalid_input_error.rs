use thiserror::Error;

#[derive(Error, Debug)]
#[error("invalid input {parameter}: {message}")]
pub struct InvalidInputError {
    parameter: String,
    message: String,
    #[source]
    source: Option<anyhow::Error>,
}

impl InvalidInputError {
    pub fn with_message(message: String) -> Self {
        InvalidInputError {
            parameter: "".to_string(),
            message,
            source: None,
        }
    }
    pub fn with_parameter_and_message(parameter: String, message: String) -> Self {
        InvalidInputError {
            parameter,
            message,
            source: None,
        }
    }
}