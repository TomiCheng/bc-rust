use crate::Result;
use crate::math::BigInteger;
use crate::util::big_integers;
use std::fmt::Display;
use std::hash::Hash;

pub trait EcFieldElement: PartialEq + Hash + Clone + Display {
    fn big_integer(&self) -> &BigInteger;
    fn field_name(&self) -> String;
    fn field_size(&self) -> usize;
    fn add(&self, b: &Self) -> Self;
    fn add_one(&self) -> Self;
    fn subtract(&self, b: &Self) -> Self;
    fn multiply(&self, b: &Self) -> Result<Self>
    where
        Self: Sized;
    fn divide(&self, b: &Self) -> Result<Self>
    where
        Self: Sized;
    fn negate(&self) -> Self;
    fn square(&self) -> Result<Self>;
    fn invert(&self) -> Result<Self>
    where
        Self: Sized;
    fn sqrt(&self) -> Result<Option<Self>>
    where
        Self: Sized;
    fn bit_length(&self) -> usize {
        self.big_integer().bit_length()
    }
    fn is_one(&self) -> bool {
        self.bit_length() == 1
    }
    fn is_zero(&self) -> bool {
        self.big_integer().sign() == 0
    }
    fn multiply_minus_product(&self, b: &Self, x: &Self, y: &Self) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(self.multiply(b)?.subtract(&x.multiply(y)?))
    }
    fn multiply_plus_product(&self, b: &Self, x: &Self, y: &Self) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(self.multiply(b)?.add(&x.multiply(y)?))
    }
    fn square_minus_product(&self, x: &Self, y: &Self) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(self.square()?.subtract(&x.multiply(y)?))
    }
    fn square_plus_product(&self, x: &Self, y: &Self) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(self.square()?.add(&x.multiply(y)?))
    }
    fn square_pow(&self, pow: usize) -> Result<Self>
    where
        Self: Sized,
    {
        let mut result = self.clone();
        for _ in 0..pow {
            result = result.square()?;
        }
        Ok(result)
    }
    fn test_bit_zero(&self) -> bool {
        self.big_integer().test_bit(0)
    }
    fn get_encoded(&self) -> Result<Vec<u8>> {
        big_integers::to_unsigned_bytes(self.get_encoded_length(), self.big_integer())
    }
    fn get_encoded_length(&self) -> usize {
        (self.field_size() + 7) / 8
    }
    fn encode_to(&self, buf: &mut [u8]) -> Result<()> {
        big_integers::to_unsigned_bytes_inplace(self.big_integer(), buf)
    }
}
