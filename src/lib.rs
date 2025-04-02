#![feature(random)]
#![feature(error_generic_member_access)]
#![feature(associated_type_defaults)]

pub mod asn1;
pub mod crypto;
pub mod error;
pub mod math;
pub mod security;
pub mod util;

pub type Result<T> = anyhow::Result<T>;
pub use error::Error;
