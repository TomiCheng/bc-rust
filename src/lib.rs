#![feature(random)]

pub mod math;
mod error;
pub mod util;
pub mod crypto;
mod security;
pub mod asn1;

pub use error::BcError;
pub type Result<T> = std::result::Result<T, BcError>;