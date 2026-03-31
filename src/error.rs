use std::fmt;

/// Top-level error type for the bc-rust library.
#[derive(Debug)]
#[non_exhaustive]
pub enum BcError {
    /// An argument passed to a function was invalid.
    InvalidArgument { param: Option<String>, msg: String },
}

impl fmt::Display for BcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BcError::InvalidArgument { param: Some(p), msg } => write!(f, "Invalid argument '{}': {}", p, msg),
            BcError::InvalidArgument { param: None, msg } => write!(f, "Invalid argument: {}", msg),
        }
    }
}

impl std::error::Error for BcError {}

/// Convenience alias for `Result<T, BcError>`.
pub type BcResult<T> = Result<T, BcError>;

/// Creates an `Err(BcError::InvalidArgument)`.
///
/// Usage:
/// - `invalid_arg!("msg")`             — no param name
/// - `invalid_arg!("key", "msg")`      — with param name
#[macro_export]
macro_rules! invalid_arg {
    ($msg:expr) => {
        Err($crate::error::BcError::InvalidArgument { param: None, msg: $msg.to_string() })
    };
    ($param:expr, $msg:expr) => {
        Err($crate::error::BcError::InvalidArgument { param: Some($param.to_string()), msg: $msg.to_string() })
    };
}
