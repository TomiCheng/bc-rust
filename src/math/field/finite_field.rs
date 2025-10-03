use crate::math::BigInteger;

pub trait FiniteField {
    fn characteristic(&self) -> &BigInteger;
    fn dimension(&self) -> u32;
}