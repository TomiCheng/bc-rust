use std::backtrace::Backtrace;
use std::collections::HashMap;
use std::error;
use std::fmt;
use std::io;
use std::num;

#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    message: String,
    source: Option<Box<dyn error::Error>>,
    backtrace: Backtrace,
    extensions: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    InvalidInput,
    InvalidFormat,
    ArithmeticError,
    ParseIntError,
    InvalidOperation,
    IoError,
    EndOfReadError,
    Other,
}

impl Error {
    pub fn with_message(kind: ErrorKind, message: String) -> Error {
        Error {
            kind,
            message,
            source: None,
            backtrace: Backtrace::capture(),
            extensions: HashMap::new(),
        }
    }

    pub fn with_invalid_input(message: String, parameter: String) -> Error {
        let mut extensions = HashMap::new();
        extensions.insert("parameter".to_string(), parameter);
        Error {
            kind: ErrorKind::InvalidInput,
            message,
            source: None,
            backtrace: Backtrace::capture(),
            extensions,
        }
    }

    pub fn with_io_error(message: String, source: io::Error) -> Error {
        Error {
            kind: ErrorKind::IoError,
            message,
            source: Some(Box::new(source)),
            backtrace: Backtrace::capture(),
            extensions: HashMap::new(),
        }
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }
    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn backtrace(&self) -> &Backtrace {
        &self.backtrace
    }
    
    pub fn extensions(&self) -> &HashMap<String, String> {
        &self.extensions
    }
   
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        self.source.as_ref().map(|e| &**e)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.message)
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Error {
            kind: ErrorKind::IoError,
            message: error.to_string(),
            source: Some(Box::new(error)),
            backtrace: Backtrace::capture(),
            extensions: HashMap::new(),
        }
    }
}

impl From<num::ParseIntError> for Error {
    fn from(error: num::ParseIntError) -> Self {
        Error {
            kind: ErrorKind::ParseIntError,
            message: error.to_string(),
            source: Some(Box::new(error)),
            backtrace: Backtrace::capture(),
            extensions: HashMap::new(),
        }
    }
}


