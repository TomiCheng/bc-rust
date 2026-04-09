//! Trait for parsing PEM objects into concrete types.
//!
//! Port of `PemObjectParser.cs` from bc-csharp.

use super::pem_object::PemObject;
use crate::error::BcResult;

/// Trait for types that can parse a [`PemObject`] into a concrete type.
///
/// Equivalent to bc-csharp's `PemObjectParser` interface.
///
/// Unlike bc-csharp which returns `object`, Rust uses an associated type
/// to specify the concrete output type at compile time.
///
/// # Note
///
/// Implementations will be added when concrete cryptographic types
/// (certificates, keys, etc.) are available.
pub trait PemObjectParser {
    /// The concrete type produced by parsing.
    type Output;

    /// Parses a [`PemObject`] into [`Self::Output`].
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::BcError::PemError`] if parsing fails.
    fn parse_object(&self, obj: &PemObject) -> BcResult<Self::Output>;
}
