use std::fmt;

/// Top-level error type for the bc-rust library.
#[derive(Debug)]
pub enum BcError {
    /// An argument passed to a function was invalid.
    InvalidArgument(String),
    /// The input data buffer is too short or has an unexpected length.
    DataLength(String),
    /// The output buffer is too small to hold the result.
    OutputLength(String),
}

impl fmt::Display for BcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BcError::InvalidArgument(msg) => write!(f, "Invalid argument: {}", msg),
            BcError::DataLength(msg) => write!(f, "Data length error: {}", msg),
            BcError::OutputLength(msg) => write!(f, "Output length error: {}", msg),
        }
    }
}

impl std::error::Error for BcError {}

/// Convenience alias for `Result<T, BcError>`.
pub type BcResult<T> = Result<T, BcError>;
