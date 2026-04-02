//! Trait for generating PEM objects.
//!
//! Port of `PemObjectGenerator.cs` from bc-csharp.

use crate::error::BcResult;
use super::pem_object::PemObject;

/// Trait for types that can generate a [`PemObject`].
///
/// Equivalent to bc-csharp's `PemObjectGenerator` interface.
pub trait PemObjectGenerator {
    /// Generates a [`PemObject`].
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::BcError::PemError`] if generation fails.
    fn generate(&self) -> BcResult<PemObject>;
}
