use crate::math::BigInteger;
use crate::Result;

pub trait EcFieldElement {
    fn big_integer(&self) -> &BigInteger;
    fn field_name(&self) -> String;
    fn field_size(&self) -> usize;
    fn add(&self, b: &Self) -> Self;
    fn add_one(&self) -> Self;
    fn subtract(&self, b: &Self) -> Self;
    fn multiply(&self, b: &Self) -> Self;
    fn divide(&self, b: &Self) -> Result<Self> where Self: Sized;
}

// TODO