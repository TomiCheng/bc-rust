use crate::math::BigInteger;

pub trait FiniteField {
    /// Returns the characteristic of the field.
    fn characteristic(&self) -> &BigInteger;

    /// Returns the dimension of the field.
    fn dimension(&self) -> usize;
}