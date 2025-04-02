#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Invalid argument {parameter}: {msg}")]
    InvalidInput {
        msg: String,
        parameter: String,
    },
    #[error("Arithmetic Error: {msg}")]
    ArithmeticError {
        msg: String,
    },
    #[error("Invalid Operation: {msg}")]
    InvalidOperation {
        msg: String,
    },
    #[error("Invalid Format: {msg}")]
    InvalidFormat {
        msg: String,
    },
    #[error("IO Error: {msg}")]
    IoError {
        msg: String,
    },
    #[error("End Of Stream: {msg}")]
    EndOfStream {
        msg: String,
    },
    #[error("Invalid Cast: {msg}")]
    InvalidCast {
        msg: String,
    },
//     kind: ErrorKind,
//     message: String,
//     source: Option<Box<dyn error::Error>>,
//     extensions: HashMap<String, String>,
}

impl Error {
    pub(crate) fn invalid_operation(msg: &str) -> Self {
        Error::InvalidOperation {
            msg: msg.to_owned(),
        }
    }
    pub(crate) fn invalid_argument(msg: &str, parameter: &str) -> Self {
        Error::InvalidInput {
            msg: msg.to_owned(),
            parameter: parameter.to_owned(),
        }
    }
    pub(crate) fn invalid_format(msg: &str) -> Self {
        Error::InvalidFormat {
            msg: msg.to_owned(),
        }
    }
    pub fn invalid_cast(msg: &str) -> Self {
        Error::InvalidCast {
            msg: msg.to_owned(),
        }
    }
}
// 
// pub trait Context<T, E> {
//     fn with_context<C, F>(self, context: F) -> Self
//     where
//         F: FnOnce() -> C;
// 
// }
// 
// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
// pub enum ErrorKind {
//     InvalidInput,
//     ArithmeticError,
//     InvalidOperation,
//     InvalidFormat,
//     IoError,
//     ParseIntError,
//     Other,
// }
// 
// impl Error {
//     pub fn invalid_argument(msg: &str, parameter: &str) -> Self {
//         Error {
//             kind: ErrorKind::InvalidInput,
//             message: msg.to_owned(),
//             extensions: HashMap::from([(
//                 "parameter".to_string(),
//                 parameter.to_string(),
//             )]),
//             ..Self::default()
//         }
//     }
//     pub fn arithmetic_error(msg: &str) -> Self {
//         Error {
//             kind: ErrorKind::ArithmeticError,
//             message: msg.to_owned(),
//             ..Self::default()
//         }
//     }
//     
//     pub fn invalid_operation(msg: &str) -> Self {
//         Error {
//             kind: ErrorKind::InvalidOperation,
//             message: msg.to_owned(),
//             ..Self::default()
//         }
//     }
// 
//     pub fn invalid_format(msg: &str) -> Self {
//         Error {
//             kind: ErrorKind::InvalidFormat,
//             message: msg.to_owned(),
//             ..Self::default()
//         }
//     }
//     pub fn kind(&self) -> ErrorKind {
//         self.kind
//     }
//     pub fn message(&self) -> &str {
//         &self.message
//     }
//     pub fn source(&self) -> &Option<Box<dyn error::Error>> {
//         &self.source
//     }
//     pub fn extensions(&self) -> &HashMap<String, String> {
//         &self.extensions
//     }
// }
// 
// impl Default for Error {
//     fn default() -> Self {
//         Error {
//             kind: ErrorKind::Other,
//             message: "Other error".to_owned(),
//             source: None,
//             extensions: HashMap::new(),
//         }
//     }
// }
// 
// impl fmt::Debug for Error {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         f.debug_struct("Error")
//             .field("kind", &self.kind)
//             .field("message", &self.message)
//             .field("source", &self.source)
//             .field("extensions", &self.extensions)
//             .finish()
//     }
// }
// 
// impl error::Error for Error {
// 
// }
// impl fmt::Display for Error {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(f, "{}", self.message)
//     }
// }
// impl From<io::Error> for Error {
//     fn from(value: io::Error) -> Self {
//         Error {
//             kind: ErrorKind::IoError,
//             message: value.to_string(),
//             source: Some(Box::new(value)),
//             ..Self::default()
//         }
//     }
// }
// 
// impl From<std::num::ParseIntError> for Error {
//     fn from(value: std::num::ParseIntError) -> Self {
//         Error {
//             kind: ErrorKind::ParseIntError,
//             message: value.to_string(),
//             source: Some(Box::new(value)),
//             ..Self::default()
//         }
//     }
// }
// 
// impl<T, E> Context<T, E> for Result<T, E> {
//     fn with_context<C, F>(self, context: F) -> Self
//     where F: FnOnce() -> C
//     {
//         match self {
//             Ok(_) => self,
//             Err(e) => {
//                 let rrr = context();
//                 todo!();
//             },
//         }
//     }
// }
// 
// #[macro_export]
// macro_rules! invalid_argument {
//     ($msg:expr, $parameter:expr) => {
//         return Err(Error::invalid_argument($msg, $parameter))
//     };
//     ($condition:expr, $msg:expr, $parameter:expr) => {
//         if $condition {
//             return Err(Error::invalid_argument($msg, $parameter))
//         }
//     };
// }
// 
// #[macro_export]
// macro_rules! arithmetic_error {
//     ($msg:expr) => {
//         return Err(Error::arithmetic_error($msg))
//     };
//     ($condition:expr, $msg:expr) => {
//         if $condition {
//             return Err(Error::arithmetic_error($msg))
//         }
//     };
// }
// 
// #[macro_export]
// macro_rules! invalid_operation {
//     ($msg:expr) => {
//         return Err(Error::invalid_operation($msg))
//     };
//     ($condition:expr, $msg:expr) => {
//         if $condition {
//             return Err(Error::invalid_operation($msg))
//         }
//     };
// }
// 
// #[macro_export]
// macro_rules! invalid_format {
//     ($msg:expr) => {
//         return Err(Error::invalid_format($msg))
//     };
//     ($condition:expr, $msg:expr) => {
//         if $condition {
//             return Err(Error::invalid_format($msg))
//         }
//     };
// }
// 
// //
// // pub enum BcError {
// //     //#[error("Invalid argument {parameter}: {msg}")]
// //     InvalidArgument { msg: String, parameter: String },
// //     //#[error("Invalid operation: {msg}")]
// //     InvalidOperation { msg: String },
// //     //#[error("Invalid format: {msg}")]
// //     InvalidFormat { msg: String },
// //     //#[error("Arithmetic Error: {msg}")]
// //     ArithmeticError { msg: String },
// //     //#[error("End of read error: {msg}")]
// //     EndOfReadError { msg: String },
// // }
// //
// // impl BcError {
// //     pub fn invalid_argument(msg: &str, parameter: &str) -> BcError {
// //         BcError::InvalidArgument {
// //             msg: msg.to_owned(),
// //             parameter: parameter.to_owned(),
// //         }
// //     }
// //     pub fn invalid_operation(msg: &str) -> BcError {
// //         BcError::InvalidOperation {
// //             msg: msg.to_owned(),
// //         }
// //     }
// //     pub fn arithmetic_error(msg: &str) -> BcError {
// //         BcError::ArithmeticError {
// //             msg: msg.to_owned(),
// //         }
// //     }
// //
// //     pub fn invalid_format(msg: &str) -> BcError {
// //         BcError::InvalidFormat {
// //             msg: msg.to_owned(),
// //         }
// //     }
// //
// //     pub fn eof_of_read(msg: &str) -> BcError {
// //         BcError::EndOfReadError {
// //             msg: msg.to_owned(),
// //         }
// //     }
// // }
