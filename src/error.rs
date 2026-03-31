use std::fmt;

#[derive(Debug)]
pub enum BcError {
    InvalidArgument(String),
    DataLength(String),
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

pub type BcResult<T> = Result<T, BcError>;
