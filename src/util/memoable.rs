use crate::Result;

/// Utility module: defines the trait for objects whose state can be backed up (memoized).
pub trait Memoable {
    /// Produce a copy of this object with its configuration and in its current state.
    ///
    /// # Remarks
    ///
    /// The returned object may be used simply to store the state, or may be used as a similar object
    /// starting from the copied state.
    fn copy(&self) -> Self;

    /// Restore a copied object state into this object.
    ///
    /// # Remarks
    ///
    /// Implementations of this method **should** try to avoid or minimize memory allocation to perform the reset.
    ///
    /// # Arguments
    ///
    /// * `other` - an object originally [`Memoable::copy`] from an object of the same type as this instance.
    ///
    /// # Errors
    /// * `InvalidCast` - if the provided object is not of the correct type.
    /// * `MemoableReset` - if the `other` parameter is in some other way invalid.
    fn restore(&mut self, other: &Self) -> Result<()>;
}