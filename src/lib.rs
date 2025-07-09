pub mod math;
mod error;
pub mod util;
pub mod crypto;
mod security;
pub mod asn1;
mod global;

pub use error::BcError;
pub use global::{Global, GLOBAL};

pub type Result<T> = std::result::Result<T, BcError>;