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
    /// Implementations of this method **should** try to avoid or minimise memory allocation to perform the reset.
    /// 
    /// # Arguments
    /// 
    /// * `other` - an object originally {@link #copy() copied} from an object of the same type as this instance.
    /// 
    fn reset(&mut self, other: &Self);
}