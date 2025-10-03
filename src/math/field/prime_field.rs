use super::*;
use crate::math::BigInteger;
use std::hash::{Hash, Hasher};

#[derive(Clone)]
pub(crate) struct PrimeField {
    characteristic: BigInteger,
}
impl PrimeField {
    pub fn new(characteristic: BigInteger) -> Self {
        Self { characteristic }
    }
}
impl FiniteField for PrimeField {
    fn characteristic(&self) -> &BigInteger {
        &self.characteristic
    }

    fn dimension(&self) -> u32 {
        1
    }
}
impl PartialEq for PrimeField {
    fn eq(&self, other: &Self) -> bool {
        self.characteristic == other.characteristic
    }
}
impl Hash for PrimeField {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.characteristic.hash(state);
    }
}
