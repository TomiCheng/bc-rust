// TODO

use std::hash::{Hash, Hasher};
use crate::math::big_integer::{ONE, ZERO};
use crate::math::BigInteger;
use crate::math::ec::EcFieldElement;
use crate::util::big_integers;
use crate::Result;

pub struct FpFieldElement {
    q: BigInteger,
    r: Option<BigInteger>,
    x: BigInteger,
}

impl FpFieldElement {
    pub(crate) fn new(q: BigInteger, r: Option<BigInteger>, x: BigInteger) -> Self {
        FpFieldElement { q, r, x }
    }
    fn mod_add(&self, x1: &BigInteger, x2: &BigInteger) -> BigInteger {
        todo!();
    }
    fn mod_reduce(&self, x: &BigInteger) -> BigInteger {
        todo!();
    }
    fn mod_subtract(&self, x1: &BigInteger, x2: &BigInteger) -> BigInteger {
        let mut x3 = x1.subtract(x2);
        if x3.sign() < 0 {
            x3 = x3.add(&self.q)
        }
        x3
    }
    fn mod_multiply(&self, x1: &BigInteger, x2: &BigInteger) -> BigInteger {
        self.mod_reduce(&x1.multiply(x2))
    }
    fn mod_inverse(&self, x: &BigInteger) -> Result<BigInteger> {
        big_integers::mod_odd_inverse(&self.q, x)?;
        todo!();
    }
}

impl EcFieldElement for FpFieldElement {
    fn big_integer(&self) -> &BigInteger {
        &self.x
    }
    fn field_name(&self) -> String {
        "Fp".to_string()
    }
    fn field_size(&self) -> usize {
        self.q.bit_length()
    }
    fn add(&self, b: &Self) -> Self {
        Self::new(self.q.clone(), self.r.clone(), self.mod_add(&self.x, b.big_integer()))
    }
    fn add_one(&self) -> Self {
        let mut x2 = self.x.add(&(*ONE));
        if x2 == self.q {
            x2 = (*ZERO).clone();
        }
        Self::new(self.q.clone(), self.r.clone(), x2)
    }
    fn subtract(&self, b: &Self) -> Self {
        Self::new(self.q.clone(), self.r.clone(), self.mod_subtract(&self.x, b.big_integer()))
    }
    fn multiply(&self, b: &Self) -> Self {
        Self::new(self.q.clone(), self.r.clone(), self.mod_multiply(&self.x, b.big_integer()))
    }
    fn divide(&self, b: &Self) -> Result<Self> {
        Ok(Self::new(self.q.clone(), self.r.clone(), self.mod_multiply(&self.x, &self.mod_inverse(b.big_integer())?)))
    }
}
impl PartialEq for FpFieldElement {
    fn eq(&self, other: &Self) -> bool {
        self.q == other.q && self.x == other.x
    }
}
impl Hash for FpFieldElement {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.q.hash(state);
        self.x.hash(state);
    }
}