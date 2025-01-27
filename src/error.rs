use thiserror::Error;
use std::num::ParseIntError;

#[derive(Error, Debug)]
pub enum BcError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid format: {0}")]
    InvalidFormat(String),
    #[error("invalid format: {source}")]
    ParseIntError {
        msg: String,
        #[source]
        source: ParseIntError,
    },
    #[error("arithmetic error: {0}")]
    ArithmeticError(String),
    #[error("invalid operation: {0}")]
    InvalidOperation(String),
}