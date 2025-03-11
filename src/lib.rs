#![feature(random)]
pub mod asn1;
pub mod crypto;
pub mod error;
pub mod math;
pub mod security;
pub mod util;

pub use error::{Error, ErrorKind};
pub type Result<T> = std::result::Result<T, Error>;
