use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BcError {
    #[error("{0}")]
    InvalidArgument(String),
    #[error("{0}")]
    ArithmeticError(String),
    #[error("{0}")]
    ParseIntError(#[from] std::num::ParseIntError),
}

impl BcError {
    pub fn invalid_argument(message: impl AsRef<str>) -> Self {
        BcError::InvalidArgument(message.as_ref().to_string())
    }
    pub fn arithmetic_error(message: impl AsRef<str>) -> Self {
        BcError::ArithmeticError(message.as_ref().to_string())
    }
}
