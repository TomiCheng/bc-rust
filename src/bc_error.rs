use std::fmt::{Debug};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BcError {
    #[error("{message}, argument: {argument}")]
    InvalidArgument { message: String, argument: String },
    #[error("{0}")]
    ArithmeticError(String),
    #[error("{0}")]
    InvalidData(String),
    #[error("{0}")]
    InvalidOperation(String),
    #[error("{0}")]
    ParseIntError(#[from] std::num::ParseIntError),
    #[error("{0}")]
    IoError(#[from] std::io::Error),
}

impl BcError {
    pub fn invalid_argument(message: impl AsRef<str>) -> Self {
        BcError::InvalidArgument {
            message: message.as_ref().to_string(),
            argument: "".to_string()
        }
    }
    pub fn arithmetic_error(message: impl AsRef<str>) -> Self {
        BcError::ArithmeticError(message.as_ref().to_string())
    }
    pub fn invalid_data(message: impl AsRef<str>) -> Self {
        BcError::InvalidData(message.as_ref().to_string())
    }
    pub fn invalid_operation(message: impl AsRef<str>) -> Self {
        BcError::InvalidOperation(message.as_ref().to_string())
    }
}

#[macro_export]
macro_rules! err_invalid_arg {
    ($cond:expr, $msg:expr, $arg:expr) => {
        if $cond {
            return Err(crate::BcError::InvalidArgument {
                message: $msg.to_string(),
                argument: $arg.to_string(),
            });
        }
    };
}
