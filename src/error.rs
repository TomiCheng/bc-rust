use std::fmt;

/// Top-level error type for the bc-rust library.
#[derive(Debug)]
#[non_exhaustive]
pub enum BcError {
    /// An argument passed to a function was invalid.
    InvalidArgument { param: Option<String>, msg: String },
    /// An I/O error occurred, with an optional source and a message.
    IoError { source: Option<std::io::Error>, msg: String },
}

impl fmt::Display for BcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BcError::InvalidArgument { param: Some(p), msg } => write!(f, "Invalid argument '{}': {}", p, msg),
            BcError::InvalidArgument { param: None, msg } => write!(f, "Invalid argument: {}", msg),
            BcError::IoError { source: Some(s), msg } => write!(f, "I/O error: {} ({})", msg, s),
            BcError::IoError { source: None, msg } => write!(f, "I/O error: {}", msg),
        }
    }
}

impl std::error::Error for BcError {}

impl From<std::io::Error> for BcError {
    fn from(e: std::io::Error) -> Self {
        BcError::IoError { source: Some(e), msg: "I/O operation failed".to_string() }
    }
}

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

/// Creates an `Err(BcError::IoError)`.
///
/// Usage:
/// - `io_error!("msg")`               — no source
/// - `io_error!(source, "msg")`       — with source
#[macro_export]
macro_rules! io_error {
    ($msg:expr) => {
        Err($crate::error::BcError::IoError { source: None, msg: $msg.to_string() })
    };
    ($source:expr, $msg:expr) => {
        Err($crate::error::BcError::IoError { source: Some($source), msg: $msg.to_string() })
    };
}
