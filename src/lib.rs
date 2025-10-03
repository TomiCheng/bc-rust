//#![cfg_attr(debug_assertions, allow(unused))]

pub mod math;
pub mod bc_error;
pub mod crypto;
pub mod util;
pub mod security;

pub use bc_error::BcError;
pub type BcResult<T> = Result<T, BcError>;