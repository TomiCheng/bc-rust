
use std::error::Error;
use std::fmt::{Debug, Formatter};
use std::num::ParseIntError;

pub struct BcError {
    error: Box<dyn Error + Send + Sync>,
    kind: ErrorKind,
}

#[derive(Debug, Clone)]
pub enum ErrorKind {
    ArgumentOutOfRange,
    InvalidInput,
    Overflow,
    ParsingIntError,
    ArithmeticError,
    InvalidCast,
    MemoableReset,
    OutputLength,
}

macro_rules! define_error_fn {
    ($(($fn_name:ident, $kind:expr)),* $(,)?) => {
        $(
            pub fn $fn_name<E>(error: E) -> Self
            where
                E: Into<Box<dyn Error + Send + Sync>>,
            {
                BcError {
                    error: error.into(),
                    kind: $kind,
                }
            }
        )*
    };
}

impl BcError {
    define_error_fn!(
        (with_invalid_argument, ErrorKind::InvalidInput),
        (with_overflow, ErrorKind::Overflow),
        (with_arithmetic_error, ErrorKind::ArithmeticError),
        (with_invalid_operation, ErrorKind::InvalidInput),
        (with_argument_out_of_range, ErrorKind::ArgumentOutOfRange),
        (with_output_length, ErrorKind::OutputLength),
        (with_invalid_cast, ErrorKind::InvalidCast),
        (with_memoable_reset, ErrorKind::MemoableReset),
    );
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