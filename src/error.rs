
use std::error::Error;
use std::fmt::{Debug, Formatter};
use std::num::ParseIntError;

pub struct BcError {
    error: Box<dyn Error + Send + Sync>,
    kind: ErrorKind,
}

#[derive(Debug, Clone)]
pub enum ErrorKind {
    InvalidInput,
    Overflow,
    ParsingIntError,
    ArithmeticError,
}

impl BcError {
    pub fn with_invalid_argument<E>(error: E) -> Self
        where E: Into<Box<dyn Error + Send + Sync>> {
        BcError {
            error: error.into(),
            kind: ErrorKind::InvalidInput,
        }
    }
    pub fn with_overflow<E>(error: E) -> Self
        where E: Into<Box<dyn Error + Send + Sync>> {
        BcError {
            error: error.into(),
            kind: ErrorKind::Overflow,
        }
    }
    pub fn with_arithmetic_error<E>(error: E) -> Self
        where E: Into<Box<dyn Error + Send + Sync>> {
        BcError {
            error: error.into(),
            kind: ErrorKind::ArithmeticError,
        }
    }
}

impl Debug for BcError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "BcError: {:?}, kind: {:?}", self.error, self.kind)
    }
}

impl From<ParseIntError> for BcError {
    fn from(error: ParseIntError) -> Self {
        BcError {
            error: Box::new(error),
            kind: ErrorKind::ParsingIntError,
        }
    }
}