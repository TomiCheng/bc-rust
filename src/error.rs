use thiserror::Error;
use std::num::ParseIntError;
use std::io::Error as IoError;
use chrono::ParseError;

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
    #[error("io error: {source}")]
    IoError {
        msg: String,
        #[source]
        source: IoError,
    },
    #[error("arithmetic error: {0}")]
    ArithmeticError(String),
    #[error("invalid operation: {0}")]
    InvalidOperation(String),
    #[error("invalid case: {0}")]
    InvalidCase(String),
    #[error("io error: {source}")]
    ParseError {
        msg: String,
        #[source]
        source: ParseError,
    },
    #[error("asn1 error: {msg}")]
    Asn1Error {
        msg: String,
        source: Option<Box<BcError>>,
    }, 
    #[error("eod of read error: {msg}")]
    EndOfReadError {
        msg: String,
    },
}