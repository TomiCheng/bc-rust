use crate::math::BigInteger;
use crate::math::field::FiniteField;

#[derive(Debug, Hash, PartialEq)]
pub(crate) struct PrimeField {
    characteristic: BigInteger
}

impl PrimeField {
    pub fn new(characteristic: BigInteger) -> Self {
        PrimeField { characteristic }
    }
}

impl FiniteField for PrimeField {
    fn characteristic(&self) -> &BigInteger {
        &self.characteristic
    }
    fn dimension(&self) -> usize {
        1
    }
}