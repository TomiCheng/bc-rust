#![feature(random)]
#![feature(error_generic_member_access)]

pub mod asn1;
pub mod crypto;
pub mod error;
pub mod math;
pub mod security;
pub mod util;

pub use error::BcError;
pub type Error = anyhow::Error;
pub type Result<T> = anyhow::Result<T, Error>;
