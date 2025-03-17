// use chrono;
// use std::backtrace::Backtrace;
// use std::collections::HashMap;
// use std::error;
// use std::fmt;
// use std::io;
// use std::num;

// pub struct Error {
//     kind: ErrorKind,
//     message: String,
//     source: Option<Box<dyn error::Error>>,
//     backtrace: Backtrace,
//     extensions: HashMap<String, String>,
// }

// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
// pub enum ErrorKind {
//     InvalidInput,
//     InvalidFormat,
//     ArithmeticError,
//     ParseIntError,
//     InvalidOperation,
//     IoError,
//     EndOfReadError,
//     DateTimeParseError,
//     Other,
// }

// impl Error {
//     pub fn with_kind(kind: ErrorKind) -> Error {
//         Error {
//             kind,
//             message: format!("Error: {:?}", kind),
//             source: None,
//             backtrace: Backtrace::capture(),
//             extensions: HashMap::new(),
//         }
//     }
//     pub fn with_message(kind: ErrorKind, message: String) -> Error {
//         Error {
//             kind,
//             message,
//             source: None,
//             backtrace: Backtrace::capture(),
//             extensions: HashMap::new(),
//         }
//     }

//     pub fn with_invalid_input(message: String, parameter: String) -> Error {
//         let mut extensions = HashMap::new();
//         extensions.insert("parameter".to_string(), parameter);
//         Error {
//             kind: ErrorKind::InvalidInput,
//             message,
//             source: None,
//             backtrace: Backtrace::capture(),
//             extensions,
//         }
//     }

//     pub fn with_io_error(message: String, source: io::Error) -> Error {
//         Error {
//             kind: ErrorKind::IoError,
//             message,
//             source: Some(Box::new(source)),
//             backtrace: Backtrace::capture(),
//             extensions: HashMap::new(),
//         }
//     }

//     pub fn kind(&self) -> ErrorKind {
//         self.kind
//     }
//     pub fn message(&self) -> &str {
//         &self.message
//     }

//     pub fn backtrace(&self) -> &Backtrace {
//         &self.backtrace
//     }

//     pub fn extensions(&self) -> &HashMap<String, String> {
//         &self.extensions
//     }
// }

// impl error::Error for Error {
//     fn source(&self) -> Option<&(dyn error::Error + 'static)> {
//         self.source.as_ref().map(|e| &**e)
//     }
// }

// impl fmt::Display for Error {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(f, "{}", self.message)
//     }
// }

// impl fmt::Debug for Error {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         //f.debug_struct("Error")
//         f.debug_struct("Error")
//             .field("kind", &self.kind)
//             .field("message", &self.message)
//             .field("source", &self.source)
//             .field("backtrace", &self.backtrace)
//             .field("extensions", &self.extensions)
//             .finish()
//     }
// }

// impl From<io::Error> for Error {
//     fn from(error: io::Error) -> Self {
//         Error {
//             kind: ErrorKind::IoError,
//             message: error.to_string(),
//             source: Some(Box::new(error)),
//             backtrace: Backtrace::capture(),
//             extensions: HashMap::new(),
//         }
//     }
// }

// impl From<num::ParseIntError> for Error {
//     fn from(error: num::ParseIntError) -> Self {
//         Error {
//             kind: ErrorKind::ParseIntError,
//             message: error.to_string(),
//             source: Some(Box::new(error)),
//             backtrace: Backtrace::capture(),
//             extensions: HashMap::new(),
//         }
//     }
// }

// impl From<chrono::ParseError> for Error {
//     fn from(error: chrono::ParseError) -> Self {
//         Error {
//             kind: ErrorKind::DateTimeParseError,
//             message: error.to_string(),
//             source: Some(Box::new(error)),
//             backtrace: Backtrace::capture(),
//             extensions: HashMap::new(),
//         }
//     }
// }

use thiserror::Error;

#[derive(Debug, Error)]
pub enum BcError {
    #[error("Invalid argument {parameter}: {msg}")]
    InvalidArgument { msg: String, parameter: String },
    #[error("Invalid operation: {msg}")]
    InvalidOperation { msg: String },
    #[error("Invalid format: {msg}")]
    InvalidFormat { msg: String },
    #[error("Arithmetic Error: {msg}")]
    ArithmeticError { msg: String },
    #[error("End of read error: {msg}")]
    EndOfReadError { msg: String },
}

impl BcError {
    pub fn invalid_argument(msg: &str, parameter: &str) -> BcError {
        BcError::InvalidArgument {
            msg: msg.to_owned(),
            parameter: parameter.to_owned(),
        }
    }
    pub fn invalid_operation(msg: &str) -> BcError {
        BcError::InvalidOperation {
            msg: msg.to_owned(),
        }
    }
    pub fn arithmetic_error(msg: &str) -> BcError {
        BcError::ArithmeticError {
            msg: msg.to_owned(),
        }
    }

    pub fn invalid_format(msg: &str) -> BcError {
        BcError::InvalidFormat {
            msg: msg.to_owned(),
        }
    }

    pub fn eof_of_read(msg: &str) -> BcError {
        BcError::EndOfReadError {
            msg: msg.to_owned(),
        }
    }
}
