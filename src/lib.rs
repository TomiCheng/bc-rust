pub mod asn1;
pub mod crypto;
mod error;
mod global;
pub mod math;
mod security;
pub mod util;

pub use error::BcError;
pub use global::{GLOBAL, Global};

pub type Result<T> = std::result::Result<T, BcError>;
