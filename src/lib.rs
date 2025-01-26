#![feature(random)]
pub mod asn1;
pub mod crypto;
pub mod util;
pub mod math;
mod error;

pub use error::BcError;
pub type Result<T> = std::result::Result<T, BcError>;