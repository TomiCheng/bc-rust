#![feature(random)]
pub mod math;
mod error;
mod util;

pub use error::BcError;
pub type Result<T> = std::result::Result<T, BcError>;